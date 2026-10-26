"""示例插件：按顺序记录每帧检测结果（扩展机制方案 7.5、7.8）。

记录消费者：以 all 模式读取结果共享的环形缓冲。读取慢于检测、旧帧被覆盖时，SDK 跳到仍在缓冲中的帧，
本插件把缺口作为一条记录写明，读取记录的程序因此知道缺了哪些帧，而不是误以为连续。

文件格式（JSON Lines，每个会话一个或多个文件，位于数据目录 recordings 下）：
  {"type": "header", "session": ..., "part": 1, "meta": {...}, "recorder": "result-recorder 0.1.0"}
  {"type": "frame", "frame": 帧号, "acquire_qpc": ..., "result_qpc": ..., "region": [x, y, w, h],
   "det": [[x1, y1, x2, y2, score, cls], ...], "kpt": [[x, y, score, ...], ...]}
  {"type": "gap", "after": 帧号, "count": 缺少的帧数}
  {"type": "end", "reason": "session" | "rotate" | "size" | "stopped", "frames": 本文件帧数}
"""
import collections
import json
import os
import threading
import time

import neural_reflex as nr

VERSION = "0.1.0"
root = nr.data_dir() / "recordings"
root.mkdir(parents=True, exist_ok=True)


class Config:
    def __init__(self, cfg):
        self.classes = {c.strip() for c in str(cfg.get("classes", "")).split(",") if c.strip()}
        self.min_score = float(cfg.get("minScore", 0))
        self.keypoints = bool(cfg.get("keypoints", True))
        self.skip_empty = bool(cfg.get("skipEmpty", False))
        self.max_bytes = int(cfg.get("maxFileMB", 500)) * 1024 * 1024


class Recorder:
    """当前文件与统计。主循环写入，控制线程（操作）可能同时调用，用锁保护。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.file = None
        self.path = None
        self.session = None
        self.meta = {}
        self.part = 0
        self.file_frames = 0
        self.file_bytes = 0
        self.recorded = 0
        self.lost = 0
        self.recent_losses = collections.deque()  # (时刻, 帧数)
        self.write_error = None
        self.last_flush = self.last_frame = time.monotonic()

    def _write(self, record):
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            self.file.write(line)
            self.write_error = None
        except OSError as e:
            self.write_error = e.strerror or str(e)
            return
        self.file_bytes += len(line.encode("utf-8"))

    def _open(self, session, meta):
        self.part = self.part + 1 if session == self.session else 1
        self.session, self.meta = session, meta
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime((session or 0) / 1000 or time.time()))
        self.path = root / f"{stamp}-{session}-{self.part:03d}.jsonl"
        self.file = open(self.path, "a", encoding="utf-8", buffering=1024 * 1024)
        self.file_frames = self.file_bytes = 0
        self._write({"type": "header", "session": session, "part": self.part, "meta": meta,
                     "recorder": f"result-recorder {VERSION}"})
        nr.log(f"开始记录 {self.path.name}")

    def close(self, reason):
        with self.lock:
            self._close(reason)

    def _close(self, reason):
        if self.file is None:
            return
        self._write({"type": "end", "reason": reason, "frames": self.file_frames})
        try:
            self.file.close()
        except OSError:
            pass
        nr.log(f"结束 {self.path.name}：{self.file_frames} 帧，{self.file_bytes / 1048576:.1f} MB（{reason}）")
        self.file = None

    def rotate(self, reason):
        with self.lock:
            if self.file is None:
                return None
            session, meta = self.session, self.meta
            self._close(reason)
            self._open(session, meta)
            return self.path.name

    def frame(self, frame, meta, cfg):
        with self.lock:
            session = frame["session"]
            if self.file is None or session != self.session:
                self._close("session")
                self._open(session, meta)
            if frame["gap"]:
                # 环形缓冲被覆盖：写明缺口（all 模式下同一会话内的帧号缺口即丢失）
                self._write({"type": "gap", "after": frame["frame"] - frame["gap"] - 1, "count": frame["gap"]})
                self.lost += frame["gap"]
                self.recent_losses.append((time.monotonic(), frame["gap"]))
            names = meta.get("classNames", {})
            keep = [i for i, d in enumerate(frame["det"])
                    if d[4] >= cfg.min_score and (not cfg.classes or names.get(str(d[5]), str(d[5])) in cfg.classes)]
            if keep or not cfg.skip_empty:
                record = {"type": "frame", "frame": frame["frame"], "acquire_qpc": frame["acquire_qpc"],
                          "result_qpc": frame["result_qpc"], "region": frame["region"],
                          "det": [[round(v, 2) for v in frame["det"][i][:4]] + [round(frame["det"][i][4], 4), frame["det"][i][5]]
                                  for i in keep]}
                if cfg.keypoints and "kpt" in frame:
                    record["kpt"] = [[round(v, 2) for v in frame["kpt"][i]] for i in keep]
                self._write(record)
                self.file_frames += 1
                self.recorded += 1
            now = self.last_frame = time.monotonic()
            if now - self.last_flush >= 0.5:
                self.last_flush = now
                try:
                    self.file.flush()
                except OSError as e:
                    self.write_error = e.strerror or str(e)
            if self.file_bytes >= cfg.max_bytes:
                session, meta = self.session, self.meta
                self._close("size")
                self._open(session, meta)

    def flush(self):
        with self.lock:
            if self.file is not None:
                try:
                    self.file.flush()
                except OSError as e:
                    self.write_error = e.strerror or str(e)

    def idle_seconds(self):
        return time.monotonic() - self.last_frame

    def recent_lost(self, window=10.0):
        now = time.monotonic()
        while self.recent_losses and now - self.recent_losses[0][0] > window:
            self.recent_losses.popleft()
        return sum(n for _, n in self.recent_losses)


conf = Config(nr.settings())
rec = Recorder()


@nr.on_settings
def changed(values):
    global conf
    conf = Config(values)
    nr.log("设置已更新")


@nr.action("rotate")
def rotate():
    name = rec.rotate("rotate")
    return f"已新建 {name}" if name else "当前没有在记录"


@nr.action("open")
def open_dir():
    os.startfile(root)
    return None


@nr.action("clear")
def clear():
    current = rec.path
    removed = 0
    for p in root.glob("*.jsonl"):
        if p != current:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return f"已删除 {removed} 个文件"


@nr.on_diagnose
def diagnose():
    return {"file": str(rec.path) if rec.path else None, "recorded": rec.recorded, "lost": rec.lost,
            "writeError": rec.write_error}


def report():
    """每秒报告记录情况与健康状态（与读取循环分开，没有新帧时也更新）。"""
    last_recorded, last_time = 0, time.monotonic()
    while not nr.wait_stop(1.0):
        now = time.monotonic()
        rate = (rec.recorded - last_recorded) / (now - last_time)
        last_recorded, last_time = rec.recorded, now
        nr.metric("recorded", rec.recorded, label="已记录帧")
        nr.metric("rate", round(rate, 1), label="写入速率", unit="帧/秒")
        nr.metric("lost", rec.lost, label="缺口帧数")
        nr.metric("size", round(rec.file_bytes / 1048576, 1), label="当前文件", unit="MB")
        rec.flush()
        # 会话已结束且缓冲中的帧已读完：结束当前文件
        if rec.file is not None and nr.session().get("state") not in (None, "running") and rec.idle_seconds() > 1:
            rec.close("session")
        recent = rec.recent_lost()
        if rec.write_error:
            nr.health("error", f"写入失败：{rec.write_error}")
        elif recent:
            nr.health("degraded", f"最近 10 秒有 {recent} 帧来不及读取")
        else:
            nr.health("ok")
        nr.status(f"写入 {rec.path.name}" if rec.file else "等待检测开始")


threading.Thread(target=report, daemon=True).start()
for frame, meta in nr.frames(mode="all"):
    rec.frame(frame, meta, conf)
rec.close("stopped")
