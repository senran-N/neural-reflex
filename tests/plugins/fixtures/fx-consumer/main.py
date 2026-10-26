"""以指标报告收到的帧数、缺口、各会话帧数，以及帧与元数据的会话是否一致。"""
import threading
import time

import neural_reflex as nr

cfg = nr.settings()
stats = {"count": 0, "gaps": 0, "mismatch": 0, "sessions": {}}
lock = threading.Lock()


def report():
    while not nr.wait_stop(0.2):
        with lock:
            snapshot = {k: (dict(v) if isinstance(v, dict) else v) for k, v in stats.items()}
        for key, value in snapshot.items():
            nr.metric(key, value)


threading.Thread(target=report, daemon=True).start()
for frame, meta in nr.frames(mode=cfg["mode"], max_fps=cfg["maxFps"] or None):
    with lock:
        stats["count"] += 1
        stats["gaps"] += frame["gap"]
        key = str(frame["session"])
        stats["sessions"][key] = stats["sessions"].get(key, 0) + 1
        if meta.get("session") != frame["session"]:
            stats["mismatch"] += 1
    if cfg["sleepMs"]:
        time.sleep(cfg["sleepMs"] / 1000)
