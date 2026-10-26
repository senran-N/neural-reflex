"""并行负载模拟：本地 AI 推理（设计方案第 1 节的场景之一）。

用 TensorRT for RTX 持续运行一个较大的检测模型，每秒输出一行吞吐（JSON），
供测试脚本统计检测开启前后的变化。
用法：python ai_load.py --onnx yolo26x-1280-fp16.onnx --duration 60
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "model_export"))
from reference_pipeline import TrtRtxBackend  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--duration", type=float, default=60)
    args = parser.parse_args()

    backend = TrtRtxBackend(args.onnx)
    name, (host, _, _) = next((n, b) for n, b in backend.buffers.items() if b[2])
    tensor = np.random.default_rng(0).random(host.shape, dtype=np.float32)
    backend(tensor)  # 预热与即时编译
    print(json.dumps({"event": "start", "input": list(host.shape)}), flush=True)

    end = time.time() + args.duration
    sec_start, count = time.perf_counter(), 0
    while time.time() < end:
        backend(tensor)
        count += 1
        now = time.perf_counter()
        if now - sec_start >= 1.0:
            print(json.dumps({"event": "second", "unix_ms": int(time.time() * 1000), "fps": count / (now - sec_start)}), flush=True)
            sec_start, count = now, 0


if __name__ == "__main__":
    main()
