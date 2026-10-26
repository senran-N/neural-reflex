"""示例插件：指定类别的目标连续出现一段时间后发出提示（扩展机制方案 5.5）。"""
import csv
import datetime
import time

import neural_reflex as nr

cfg = nr.settings()
watch = {c.strip() for c in str(cfg.get("classes", "person")).split(",") if c.strip()}
min_score = float(cfg.get("minScore", 0.5))
min_count = int(cfg.get("minCount", 1))
hold_frames = int(cfg.get("holdFrames", 30))
cooldown = float(cfg.get("cooldown", 10))
beep = cfg.get("sound") == "beep"
record = bool(cfg.get("record", True))

csv_path = nr.data_dir() / "alerts.csv"
if record and not csv_path.exists():
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow(["时间", "类别", "数量", "最高置信度", "模型"])

watch_text = "、".join(sorted(watch)) or "（未设置类别）"
nr.status(f"等待检测开始 · 监视 {watch_text}")
nr.log(f"监视 {watch_text}，置信度 ≥ {min_score:.2f}，至少 {min_count} 个，连续 {hold_frames} 帧，间隔 {cooldown:g} s")

streak = alerts = processed = 0
last_alert = 0.0
session = None
window_start, window_frames, latencies = time.time(), 0, []

for frame, meta in nr.frames(on_wait=lambda: nr.status(f"等待检测开始 · 监视 {watch_text}")):
    if meta.get("session") != session:
        session = meta.get("session")
        streak = 0
        nr.status(f"监视 {watch_text} · 模型 {meta.get('model', '—')}")
    names = meta.get("classNames", {})
    hits = [d for d in frame["det"] if d[4] >= min_score and names.get(str(d[5]), str(d[5])) in watch]
    processed += 1
    window_frames += 1
    latencies.append(frame["latency_ms"])

    streak = streak + 1 if len(hits) >= min_count else 0
    now = time.time()
    if streak >= hold_frames and now - last_alert >= cooldown:
        last_alert = now
        alerts += 1
        best = max(d[4] for d in hits)
        kinds = "、".join(sorted({names.get(str(d[5]), str(d[5])) for d in hits}))
        nr.notify(f"检测到 {kinds} × {len(hits)}（最高置信度 {best:.2f}）")
        nr.metric("alerts", alerts, label="告警次数")
        if beep:
            import winsound
            winsound.MessageBeep()
        if record:
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow([datetime.datetime.now().isoformat(timespec="seconds"), kinds, len(hits),
                                        f"{best:.3f}", meta.get("model", "")])

    # 指标每秒报告一次，避免逐帧输出
    if now - window_start >= 1:
        latencies.sort()
        nr.metric("fps", round(window_frames / (now - window_start), 1), label="处理帧率", unit="帧/秒")
        nr.metric("latency", round(latencies[len(latencies) // 2], 3), label="送达延迟中位数", unit="ms")
        nr.metric("processed", processed, label="已处理帧")
        nr.metric("streak", streak, label="当前连续帧")
        window_start, window_frames, latencies = now, 0, []

nr.log("收到停止请求，退出")
