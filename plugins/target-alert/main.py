"""示例插件：指定类别的目标连续出现一段时间后发出提示（扩展机制方案 5.5、7.8）。

实时消费者：只取最新帧，处理慢时跳过旧帧。宿主启用控制通道时设置热更新、提供操作按钮；
在阶段 3 的宿主上（没有控制通道）修改设置会重启插件，其余行为相同。
"""
import csv
import datetime
import threading
import time

import neural_reflex as nr


class Config:
    def __init__(self, cfg):
        self.watch = {c.strip() for c in str(cfg.get("classes", "person")).split(",") if c.strip()}
        self.min_score = float(cfg.get("minScore", 0.5))
        self.min_count = int(cfg.get("minCount", 1))
        self.hold_frames = int(cfg.get("holdFrames", 30))
        self.cooldown = float(cfg.get("cooldown", 10))
        self.beep = cfg.get("sound") == "beep"
        self.record = bool(cfg.get("record", True))
        self.watch_text = "、".join(sorted(self.watch)) or "（未设置类别）"

    def describe(self):
        return (f"监视 {self.watch_text}，置信度 ≥ {self.min_score:.2f}，至少 {self.min_count} 个，"
                f"连续 {self.hold_frames} 帧，间隔 {self.cooldown:g} s")


conf = Config(nr.settings())  # 热更新时整体替换，主循环每帧读取一次
alerts = 0
model = "—"
csv_path = nr.data_dir() / "alerts.csv"
csv_lock = threading.Lock()


def beep():
    import winsound
    winsound.MessageBeep()


def write_csv(row):
    """追加一行告警记录；失败时报告健康状态而不是退出。"""
    try:
        with csv_lock:
            new = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if new:
                    writer.writerow(["时间", "类别", "数量", "最高置信度", "模型"])
                writer.writerow(row)
        nr.health("ok")
    except OSError as e:
        nr.health("degraded", f"无法写入 alerts.csv：{e.strerror or e}")


@nr.on_settings
def changed(values):
    global conf
    conf = Config(values)
    nr.log("设置已更新：" + conf.describe())


@nr.action("test")
def test_alert():
    c = conf
    if c.beep:
        beep()
    nr.notify(f"测试提示：监视 {c.watch_text}")
    return "已发出测试提示"


@nr.action("reset")
def reset():
    global alerts
    alerts = 0
    nr.metric("alerts", alerts, label="告警次数")
    return "告警次数已清零"


@nr.on_diagnose
def diagnose():
    return {"alerts": alerts, "config": conf.describe(), "csv": str(csv_path)}


nr.log(conf.describe())
nr.metric("alerts", alerts, label="告警次数")
streak = 0
last_alert = 0.0
session = None
last_metric = 0.0

for frame, meta in nr.frames(on_wait=lambda: nr.status(f"等待检测开始 · 监视 {conf.watch_text}")):
    c = conf
    if meta.get("session") != session:
        session = meta.get("session")
        model = meta.get("model", "—")
        streak = 0
    names = meta.get("classNames", {})
    hits = [d for d in frame["det"] if d[4] >= c.min_score and names.get(str(d[5]), str(d[5])) in c.watch]
    streak = streak + 1 if len(hits) >= c.min_count else 0
    now = time.time()
    if streak >= c.hold_frames and now - last_alert >= c.cooldown:
        last_alert = now
        alerts += 1
        best = max(d[4] for d in hits)
        kinds = "、".join(sorted({names.get(str(d[5]), str(d[5])) for d in hits}))
        nr.notify(f"检测到 {kinds} × {len(hits)}（最高置信度 {best:.2f}）")
        nr.metric("alerts", alerts, label="告警次数")
        if c.beep:
            beep()
        if c.record:
            write_csv([datetime.datetime.now().isoformat(timespec="seconds"), kinds, len(hits), f"{best:.3f}", model])

    # 自己的指标每秒报告一次；消费帧率、延迟等由 SDK 自动上报
    if now - last_metric >= 1:
        last_metric = now
        nr.status(f"监视 {c.watch_text} · 模型 {model}")
        nr.metric("streak", streak, label="当前连续帧")

nr.log("收到停止请求，退出")
