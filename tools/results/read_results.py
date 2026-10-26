"""读取 NeuralReflex 发布的每帧结果（扩展机制方案第 5 节）。只依赖 Python 标准库。

宿主（NeuralReflex 开启“结果共享”，或 nr_detect --publish）把每帧结果写入命名共享内存
Local\\NeuralReflex.Results.v1，并通过两个交替的手动重置事件通知。本脚本演示读取方的完整流程：
打开映射与事件、读取元数据、按序号锁读取完整的一帧、等待下一帧。

用法：
  python read_results.py                    每秒输出收到的帧数、漏帧数与送达延迟
  python read_results.py --jsonl out.jsonl  另把每帧结果逐行写入文件
  python read_results.py --print            打印每帧的目标
"""
import argparse
import ctypes
import json
import struct
import sys
import time
from ctypes import wintypes

MAPPING = "Local\\NeuralReflex.Results.v1"
EVENTS = ("Local\\NeuralReflex.Results.v1.Event0", "Local\\NeuralReflex.Results.v1.Event1")
MAGIC, LAYOUT = 0x5352524E, 1

FILE_MAP_READ, SYNCHRONIZE, WAIT_OBJECT_0 = 0x0004, 0x00100000, 0
HEADER = struct.Struct("<IIIIqqIIIIq")      # 56 字节，字段见方案 5.2
FRAME_HEAD = struct.Struct("<Qqqiiiiiiii")  # 帧号、两个时刻、检测区域、目标数、关键点数、候选数、溢出数
DETECTION = struct.Struct("<fffffi")        # x1, y1, x2, y2, score, cls
KEYPOINT = struct.Struct("<fff")            # x, y, score

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.OpenFileMappingW.restype = wintypes.HANDLE
k32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
k32.MapViewOfFile.restype = ctypes.c_void_p
k32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
k32.OpenEventW.restype = wintypes.HANDLE
k32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
k32.WaitForSingleObject.restype = wintypes.DWORD
k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]


def qpc():
    v = ctypes.c_int64()
    k32.QueryPerformanceCounter(ctypes.byref(v))
    return v.value


class Results:
    def __init__(self):
        mapping = k32.OpenFileMappingW(FILE_MAP_READ, False, MAPPING)
        if not mapping:
            raise FileNotFoundError("没有找到结果共享内存：宿主未开启结果共享或尚未开始检测")
        self.base = k32.MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0)
        self.events = [k32.OpenEventW(SYNCHRONIZE, False, name) for name in EVENTS]
        if not self.base or not all(self.events):
            raise OSError("打开结果共享内存或通知事件失败")
        h = HEADER.unpack(ctypes.string_at(self.base, HEADER.size))
        magic, version, self.total, self.frame_offset = h[:4]
        self.meta_offset, self.meta_capacity, self.qpc_frequency = h[6], h[7], h[10]
        if magic != MAGIC or version != LAYOUT:
            raise ValueError(f"布局不兼容：魔数 {magic:#x}，版本 {version}（本脚本支持版本 {LAYOUT}）")

    def _seq(self, offset):
        return ctypes.c_int64.from_address(self.base + offset).value

    def published(self):
        """已发布帧数（帧序号锁 / 2）。"""
        return self._seq(16) // 2

    def metadata(self):
        while True:
            s1 = self._seq(24)
            if s1 & 1:
                continue
            length = ctypes.c_uint32.from_address(self.base + 40).value
            data = ctypes.string_at(self.base + self.meta_offset, length)
            if self._seq(24) == s1:
                return json.loads(data.decode("utf-8")) if data else {}

    def read_frame(self):
        """按序号锁读取完整的一帧；返回 (已发布帧数, 帧字典)。"""
        while True:
            s1 = self._seq(16)
            if s1 & 1:
                continue
            head = FRAME_HEAD.unpack(ctypes.string_at(self.base + self.frame_offset, FRAME_HEAD.size))
            count, num_kpt = max(0, min(head[7], 300)), max(0, min(head[8], 32))
            dets = ctypes.string_at(self.base + self.frame_offset + FRAME_HEAD.size, count * DETECTION.size)
            kpts = ctypes.string_at(self.base + self.frame_offset + FRAME_HEAD.size + 300 * DETECTION.size,
                                    count * num_kpt * KEYPOINT.size)
            if self._seq(16) != s1:
                continue  # 读取期间宿主写入了新的一帧，重读
            frame = {
                "frame": head[0], "acquire_qpc": head[1], "result_qpc": head[2],
                "region": list(head[3:7]), "candidates": head[9], "overflow": head[10],
                "det": [list(DETECTION.unpack_from(dets, i * DETECTION.size)) for i in range(count)],
            }
            if num_kpt:
                pts = [KEYPOINT.unpack_from(kpts, i * KEYPOINT.size) for i in range(count * num_kpt)]
                frame["kpt"] = [[v for p in pts[i * num_kpt:(i + 1) * num_kpt] for v in p] for i in range(count)]
            return s1 // 2, frame

    def wait_next(self, seen, timeout_ms):
        """等待已发布帧数超过 seen。宿主发布第 k 帧时设置 Event[k % 2]，并预先重置 Event[(k + 1) % 2]。"""
        if self.published() > seen:
            return True
        return k32.WaitForSingleObject(self.events[(seen + 1) % 2], timeout_ms) == WAIT_OBJECT_0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="逐帧写入 JSON 行")
    ap.add_argument("--print", action="store_true", help="打印每帧的目标")
    ap.add_argument("--duration", type=float, default=0, help="运行秒数，0 表示直到 Ctrl+C")
    ap.add_argument("--quiet", action="store_true", help="不输出每秒统计")
    args = ap.parse_args()

    while True:
        try:
            results = Results()
            break
        except FileNotFoundError as e:
            print(e, "，1 秒后重试", file=sys.stderr)
            time.sleep(1)
    meta = results.metadata()
    names = meta.get("classNames", {})
    print("元数据：", json.dumps({k: v for k, v in meta.items() if k != "classNames"}, ensure_ascii=False), file=sys.stderr)

    out = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    seen, last_frame, start = results.published(), None, time.time()
    total = missed = 0
    sec_start, sec_frames, sec_latency = time.time(), 0, []
    try:
        while not args.duration or time.time() - start < args.duration:
            if not results.wait_next(seen, 200):
                continue
            seen, frame = results.read_frame()
            latency_ms = (qpc() - frame["result_qpc"]) * 1000.0 / results.qpc_frequency
            if last_frame is not None and frame["frame"] > last_frame + 1:
                missed += frame["frame"] - last_frame - 1
            last_frame = frame["frame"]
            total += 1
            sec_frames += 1
            sec_latency.append(latency_ms)
            if out:
                out.write(json.dumps(frame) + "\n")
            if args.print:
                labels = ", ".join(f"{names.get(str(d[5]), d[5])} {d[4]:.2f}" for d in frame["det"])
                print(f"帧 {frame['frame']}: {labels}")
            now = time.time()
            if not args.quiet and now - sec_start >= 1:
                sec_latency.sort()
                p = lambda q: sec_latency[min(len(sec_latency) - 1, int(q * len(sec_latency)))]
                print(f"[{now - start:6.1f}s] 收到 {sec_frames / (now - sec_start):6.1f} 帧/秒  累计漏帧 {missed}  "
                      f"送达延迟 P50 {p(0.5):.3f} P99 {p(0.99):.3f} ms", file=sys.stderr)
                sec_start, sec_frames, sec_latency = now, 0, []
    except KeyboardInterrupt:
        pass
    finally:
        if out:
            out.close()
    print(f"共收到 {total} 帧，漏帧 {missed}", file=sys.stderr)


if __name__ == "__main__":
    main()
