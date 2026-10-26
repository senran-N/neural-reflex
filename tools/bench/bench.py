"""并行负载下的检测测量（设计方案 7.3–7.5）。

每次试验：合成测试源（144 FPS，真实照片）持续运行；启动并行负载并预热；
阶段 A 只运行负载 D 秒（负载吞吐基线）；阶段 B 同时运行检测 D 秒（检测延迟分布与负载吞吐）。
同一负载下各配置交替执行并重复，避免热状态变化被误判为配置差异。

用法：python bench.py --loads none,ai,gfx --configs old,new,new_if2 --repeats 2 --duration 20 --out cache/bench/r1.jsonl
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = ROOT / "build" / "bin" / "Release"
OLD_BIN = ROOT.parent / "nr-baseline" / "build" / "bin" / "Release"
PY = sys.executable
IMAGE = r"D:\SDK\venvs\nr-export\Lib\site-packages\ultralytics\assets\bus.jpg"
MODEL = str(ROOT / "models" / "reference" / "yolo26n-640-nms-fp16.onnx")

LOADS = {
    "none": None,
    "ai": lambda t: [PY, str(ROOT / "tools" / "bench" / "ai_load.py"), "--onnx",
                     str(ROOT / "models" / "load" / "yolo26x-1280-fp16.onnx"), "--duration", str(t)],
    "gfx": lambda t: [str(BIN / "nr_gpuload.exe"), "--iterations", "160", "--duration", str(t)],
    # 只占容量：常驻 4 GB，着色器仍采样小纹理
    "gfx_vram": lambda t: [str(BIN / "nr_gpuload.exe"), "--iterations", "160", "--vram-gb", "4", "--duration", str(t)],
    # 只占带宽与时长：采样一块 256 MB 常驻纹理，单帧约 20 ms
    "gfx_bw": lambda t: [str(BIN / "nr_gpuload.exe"), "--iterations", "160", "--vram-gb", "0.25", "--sample-resident",
                         "--duration", str(t)],
    # 与 gfx_bw 计算量相同，每帧拆成多个 draw，单次 GPU 工作变短
    "gfx_bw_d16": lambda t: [str(BIN / "nr_gpuload.exe"), "--iterations", "160", "--vram-gb", "0.25", "--sample-resident",
                             "--draws", "16", "--duration", str(t)],
    "gfx_bw_d128": lambda t: [str(BIN / "nr_gpuload.exe"), "--iterations", "160", "--vram-gb", "0.25", "--sample-resident",
                              "--draws", "128", "--duration", str(t)],
    # 两者同时（r1 中的 gfx_vram）
    "gfx_vram_bw": lambda t: [str(BIN / "nr_gpuload.exe"), "--iterations", "160", "--vram-gb", "4", "--sample-resident",
                              "--duration", str(t)],
}

CONFIGS = {
    "old": (OLD_BIN / "nr_detect.exe", []),
    "new": (BIN / "nr_detect.exe", ["--inflight", "1"]),
    "new_if2": (BIN / "nr_detect.exe", ["--inflight", "2"]),
    "prio_above": (BIN / "nr_detect.exe", ["--gpu-priority", "above"]),
    "prio_high": (BIN / "nr_detect.exe", ["--gpu-priority", "high"]),
    "prio_realtime": (BIN / "nr_detect.exe", ["--gpu-priority", "realtime"]),
    "mmcss": (BIN / "nr_detect.exe", ["--mmcss"]),
    "sync_spin": (BIN / "nr_detect.exe", ["--sync", "spin"]),
    "sync_block": (BIN / "nr_detect.exe", ["--sync", "block"]),
    "best": (BIN / "nr_detect.exe", []),  # 由 --best-args 指定
}

PCT = re.compile(r"(\S+?)\s+n=(\d+)\s+P50\s+([\d.]+)\s+P95\s+([\d.]+)\s+P99\s+([\d.]+)\s+最大\s+([\d.]+)")
SECOND = re.compile(r"\[\s*([\d.]+)s\]\s+检测\s+([\d.]+)/s")


class LoadProcess:
    """并行负载进程：后台读取每秒吞吐。"""

    def __init__(self, cmd):
        self.samples, self.started = [], threading.Event()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("event") == "start":
                self.started.set()
            elif msg.get("event") == "second":
                self.samples.append((msg["unix_ms"] / 1000.0, msg["fps"]))

    def mean_between(self, t0, t1):
        vals = [f for t, f in self.samples if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else None

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def gpu_state():
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    keys = ["temp_c", "sm_mhz", "mem_mhz", "power_w", "vram_used_mb"]
    try:
        return dict(zip(keys, [float(x) for x in out.split(",")]))
    except ValueError:
        return {}


def run_detect(exe, args, duration):
    cmd = [str(exe), "--model", MODEL, "--duration", str(duration), *args]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=duration + 180)
    t1 = time.time()
    out = res.stdout + res.stderr
    pct = {m.group(1): {"n": int(m.group(2)), "p50": float(m.group(3)), "p95": float(m.group(4)),
                        "p99": float(m.group(5)), "max": float(m.group(6))} for m in PCT.finditer(out)}
    rates = [float(m.group(2)) for m in SECOND.finditer(out) if float(m.group(1)) > 3.0]
    status = re.search(r"GPU 调度优先级 请求 \S+（状态 (0x[0-9A-Fa-f]+)），生效 (\S+)", out)
    return {
        "ok": res.returncode == 0,
        "error": None if res.returncode == 0 else out.strip()[-400:],
        "wall": (t0, t1),
        "steady_rate": sum(rates) / len(rates) if rates else 0.0,
        "min_rate": min(rates) if rates else 0.0,
        "latency": pct.get("取得→结果可用（主机）"),
        "infer": pct.get("GPU"),  # 第一个以 GPU 开头的是预处理，下面单独取
        "stages": {k: v for k, v in pct.items() if k.startswith("GPU")},
        "priority_status": status.group(1) if status else None,
        "priority_effective": status.group(2) if status else None,
        "stdout_tail": out.strip().splitlines()[-8:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loads", default="none,ai,gfx,gfx_vram")
    ap.add_argument("--configs", default="old,new,new_if2")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--best-args", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.best_args:
        CONFIGS["best"] = (BIN / "nr_detect.exe", args.best_args.split())

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    D = args.duration
    source = subprocess.Popen([str(BIN / "nr_testsource.exe"), "--fps", "144", "--duration", "100000", "--image", IMAGE],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    try:
        for load in args.loads.split(","):
            for rep in range(args.repeats):
                configs = args.configs.split(",")
                if rep % 2:
                    configs.reverse()  # 交替顺序
                for name in configs:
                    exe, extra = CONFIGS[name]
                    lp = None
                    if LOADS[load]:
                        lp = LoadProcess(LOADS[load](2 * D + 60))
                        lp.started.wait(120)
                        time.sleep(4)
                    a0 = time.time()
                    time.sleep(D)
                    a1 = time.time()
                    det = run_detect(exe, extra, D)
                    state = gpu_state()
                    b0, b1 = det["wall"]
                    record = {"load": load, "config": name, "repeat": rep, "duration": D, **det, "gpu": state}
                    if lp:
                        record["load_fps_alone"] = lp.mean_between(a0 + 1, a1)
                        record["load_fps_with_detect"] = lp.mean_between(b0 + 2, b1 - 1)
                        lp.stop()
                        time.sleep(2)
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    lat = det["latency"] or {}
                    print(f"{load:9} {name:13} rep{rep} rate {det['steady_rate']:6.1f} P50 {lat.get('p50', 0):6.2f} "
                          f"P95 {lat.get('p95', 0):6.2f} P99 {lat.get('p99', 0):6.2f} max {lat.get('max', 0):6.2f} "
                          f"load {record.get('load_fps_alone') or 0:7.1f}->{record.get('load_fps_with_detect') or 0:7.1f} "
                          f"{'' if det['ok'] else 'FAILED'} {state.get('temp_c', '')}C", flush=True)
    finally:
        source.terminate()


if __name__ == "__main__":
    main()
