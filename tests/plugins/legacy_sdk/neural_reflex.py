"""NeuralReflex 外部进程插件 SDK（扩展机制方案 5.5）。只依赖 Python 标准库。

插件由宿主启动，通过环境变量取得设置，通过标准输出向宿主报告状态、指标、日志与提示，
通过共享内存读取每帧检测结果。典型用法：

    import neural_reflex as nr

    cfg = nr.settings()
    for frame, meta in nr.frames():
        names = meta.get("classNames", {})
        ...
        nr.metric("alerts", count, label="告警次数")

宿主关闭插件的标准输入表示请求停止，frames() 随即结束。
"""
import ctypes
import json
import os
import pathlib
import struct
import sys
import threading
import time
from ctypes import wintypes

PROTOCOL = 1  # 插件协议版本（plugin.json 的 api 字段）

MAPPING = os.environ.get("NR_RESULTS_MAPPING", "Local\\NeuralReflex.Results.v1")
EVENTS = (MAPPING + ".Event0", MAPPING + ".Event1")
MAGIC, LAYOUT = 0x5352524E, 1
MAX_DETECTIONS, MAX_KEYPOINTS = 300, 32

# ---------- 宿主传入的信息 ----------


def plugin_id():
    return os.environ.get("NR_PLUGIN_ID", "")


def plugin_dir():
    return pathlib.Path(os.environ.get("NR_PLUGIN_DIR", pathlib.Path.cwd()))


def data_dir():
    """插件可写的数据目录，不存在时创建。"""
    path = pathlib.Path(os.environ.get("NR_PLUGIN_DATA", pathlib.Path.cwd() / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings():
    """用户设置（宿主已按 plugin.json 的声明校验并补全缺省值）。"""
    try:
        return json.loads(os.environ.get("NR_PLUGIN_SETTINGS", "{}"))
    except json.JSONDecodeError:
        return {}


# ---------- 向宿主报告：标准输出每行一个 JSON ----------

_out_lock = threading.Lock()


def _emit(message):
    line = json.dumps(message, ensure_ascii=False)
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def status(text):
    _emit({"type": "status", "text": str(text)})


def metric(key, value, label=None, unit=None):
    message = {"type": "metric", "key": key, "value": value}
    if label is not None:
        message["label"] = label
    if unit is not None:
        message["unit"] = unit
    _emit(message)


def log(text, level="info"):
    _emit({"type": "log", "level": level, "text": str(text)})


def notify(text):
    _emit({"type": "notify", "text": str(text)})


# ---------- 停止请求：宿主关闭标准输入 ----------

_stop = threading.Event()
_watching = False


def _watch_stdin():
    try:
        while sys.stdin.read(4096):
            pass
    except (OSError, ValueError):
        pass
    _stop.set()


def stop_requested():
    """宿主是否已请求停止。仅在宿主托管时（存在 NR_PLUGIN_ID）监视标准输入。"""
    global _watching
    if not _watching and plugin_id():
        _watching = True
        threading.Thread(target=_watch_stdin, daemon=True).start()
    return _stop.is_set()


# ---------- 结果共享内存（布局见扩展机制方案 5.2） ----------

FILE_MAP_READ, SYNCHRONIZE, WAIT_OBJECT_0 = 0x0004, 0x00100000, 0
HEADER = struct.Struct("<IIIIqqIIIIq")      # 56 字节
FRAME_HEAD = struct.Struct("<Qqqiiiiiiii")  # 帧号、两个时刻、检测区域、目标数、关键点数、候选数、溢出数
DETECTION = struct.Struct("<fffffi")        # x1, y1, x2, y2, score, cls
KEYPOINT = struct.Struct("<fff")            # x, y, score

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.OpenFileMappingW.restype = wintypes.HANDLE
_k32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_k32.MapViewOfFile.restype = ctypes.c_void_p
_k32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
_k32.OpenEventW.restype = wintypes.HANDLE
_k32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]


def qpc():
    v = ctypes.c_int64()
    _k32.QueryPerformanceCounter(ctypes.byref(v))
    return v.value


class Results:
    """直接读取共享内存。一般使用 frames()；需要自行控制等待时使用本类。"""

    def __init__(self):
        mapping = _k32.OpenFileMappingW(FILE_MAP_READ, False, MAPPING)
        if not mapping:
            raise FileNotFoundError("没有找到结果共享内存：宿主未开启结果共享或尚未开始检测")
        self.base = _k32.MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0)
        self.events = [_k32.OpenEventW(SYNCHRONIZE, False, name) for name in EVENTS]
        if not self.base or not all(self.events):
            raise OSError("打开结果共享内存或通知事件失败")
        h = HEADER.unpack(ctypes.string_at(self.base, HEADER.size))
        magic, version, self.total, self.frame_offset = h[:4]
        self.meta_offset, self.meta_capacity, self.qpc_frequency = h[6], h[7], h[10]
        if magic != MAGIC or version != LAYOUT:
            raise ValueError(f"布局不兼容：魔数 {magic:#x}，版本 {version}（SDK 支持版本 {LAYOUT}）")

    def _seq(self, offset):
        return ctypes.c_int64.from_address(self.base + offset).value

    def published(self):
        """已发布帧数（帧序号锁 / 2）。"""
        return self._seq(16) // 2

    def metadata_version(self):
        return self._seq(24)

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
            count, num_kpt = max(0, min(head[7], MAX_DETECTIONS)), max(0, min(head[8], MAX_KEYPOINTS))
            dets = ctypes.string_at(self.base + self.frame_offset + FRAME_HEAD.size, count * DETECTION.size)
            kpts = ctypes.string_at(self.base + self.frame_offset + FRAME_HEAD.size + MAX_DETECTIONS * DETECTION.size,
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
        return _k32.WaitForSingleObject(self.events[(seen + 1) % 2], timeout_ms) == WAIT_OBJECT_0

    def latency_ms(self, frame):
        """结果可用到现在的时间。"""
        return (qpc() - frame["result_qpc"]) * 1000.0 / self.qpc_frequency


def open_results(on_wait=None, timeout=None):
    """打开结果共享内存，宿主尚未发布时每 0.2 s 重试；on_wait 在第一次等待时调用。
    宿主请求停止或超过 timeout 秒时返回 None。"""
    waited, deadline = False, None if timeout is None else time.time() + timeout
    while not stop_requested() and (deadline is None or time.time() < deadline):
        try:
            return Results()
        except FileNotFoundError:
            if not waited and on_wait:
                on_wait()
            waited = True
            time.sleep(0.2)
    return None


def frames(on_wait=None, timeout_ms=200):
    """逐帧产出 (frame, meta)。meta 为当前会话的元数据（会话切换时自动更新）。
    宿主请求停止时结束；只取最新帧，处理慢于检测时中间的帧被跳过（frame["frame"] 不连续）。"""
    results = open_results(on_wait)
    if results is None:
        return
    meta_version, meta = results.metadata_version(), results.metadata()
    seen = results.published()
    while not stop_requested():
        if not results.wait_next(seen, timeout_ms):
            continue
        seen, frame = results.read_frame()
        if results.metadata_version() != meta_version:
            meta_version, meta = results.metadata_version(), results.metadata()
        frame["latency_ms"] = results.latency_ms(frame)
        yield frame, meta
