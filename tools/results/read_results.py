"""读取 NeuralReflex 发布的每帧结果（扩展机制方案第 5 节）。只依赖 Python 标准库。

宿主（NeuralReflex 开启“结果共享”，或 nr_detect --publish）把每帧结果写入命名共享内存
Local\\NeuralReflex.Results.v1，并通过两个交替的手动重置事件通知。读取逻辑在
sdk/python/neural_reflex.py（插件 SDK）中，本脚本是它的命令行示例。

用法：
  python read_results.py                    每秒输出收到的帧数、漏帧数与送达延迟
  python read_results.py --jsonl out.jsonl  另把每帧结果逐行写入文件
  python read_results.py --print            打印每帧的目标
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "sdk" / "python"))
import neural_reflex as nr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", help="逐帧写入 JSON 行")
    ap.add_argument("--print", action="store_true", help="打印每帧的目标")
    ap.add_argument("--duration", type=float, default=0, help="运行秒数，0 表示直到 Ctrl+C")
    ap.add_argument("--quiet", action="store_true", help="不输出每秒统计")
    args = ap.parse_args()

    start = time.time()
    results = nr.open_results(on_wait=lambda: print("没有找到结果共享内存：宿主未开启结果共享或尚未开始检测，等待中",
                                                    file=sys.stderr),
                              timeout=args.duration or None)
    if results is None:
        print("等待超时，没有读到结果", file=sys.stderr)
        return
    meta = results.metadata()
    names = meta.get("classNames", {})
    print("元数据：", json.dumps({k: v for k, v in meta.items() if k != "classNames"}, ensure_ascii=False), file=sys.stderr)

    out = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    seen, last_frame = results.published(), None
    total = missed = 0
    sec_start, sec_frames, sec_latency = time.time(), 0, []
    try:
        while not args.duration or time.time() - start < args.duration:
            if not results.wait_next(seen, 200):
                continue
            seen, frame = results.read_frame()
            latency_ms = results.latency_ms(frame)
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
