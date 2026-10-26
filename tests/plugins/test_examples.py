"""示例插件的端到端测试（扩展机制方案 7.8）：在 nr_plugin_run 中托管 plugins/ 下的示例插件，
用 nr_publish 发布合成结果，检查插件的输出。用法同 test_protocol.py。"""
import json
import socket
import pathlib
import sys
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_protocol import BIN, ROOT, PluginRun, publish  # noqa: E402

PLUGINS = ROOT / "plugins"


def run_publisher(*args):
    out, _ = publish(*args).communicate(timeout=120)
    return json.loads(out.decode("utf-8").strip().splitlines()[-1])


class TargetAlertTests(unittest.TestCase):
    def test_alerts_and_live_settings(self):
        settings = {"classes": "person", "holdFrames": 10, "cooldown": 1, "record": True}
        with PluginRun(str(PLUGINS / "target-alert"), settings) as run:
            run.wait_state("running", timeout=15)
            time.sleep(0.5)
            pub = publish("--start-delay", 1.0, "--fps", 144, "--duration", 4)
            run.wait(lambda e: e.get("event") == "notify" and "person" in e["text"], timeout=10, what="告警提示")
            # 热更新：改为监视不存在的类别，不重启
            pid = run.wait_state("running")["pid"]
            mark = run.mark()
            run.send(cmd="settings", values={**settings, "classes": "giraffe"})
            run.wait_log("设置已更新：监视 giraffe", since=mark)
            pub.communicate(timeout=30)
            self.assertEqual(run.states().count("starting"), 1)
            self.assertGreater(run.metric("alerts"), 0)
            self.assertIsNotNone(run.metric("nr.fps"))
            # 操作：测试提示与清零（清零在界面上需要确认，宿主收到请求时直接执行）
            run.send(cmd="action", id="test")
            run.wait(lambda e: e.get("event") == "notify" and e["text"] == "已发出测试提示", since=mark)
            run.send(cmd="action", id="reset")
            run.wait(lambda e: e.get("event") == "notify" and e["text"] == "告警次数已清零", since=mark)
            run.wait_metric("alerts", lambda v: v == 0)
            self.assertEqual(run.wait_state("running")["pid"], pid)
            csv_path = pathlib.Path(run.data, "plugin-data", "target-alert", "alerts.csv")
        self.assertTrue(csv_path.exists())
        self.assertIn("person", csv_path.read_text(encoding="utf-8-sig"))


class ResultRecorderTests(unittest.TestCase):
    def read_records(self, run):
        files = sorted(pathlib.Path(run.data, "plugin-data", "result-recorder", "recordings").glob("*.jsonl"))
        return files, [[json.loads(line) for line in f.read_text(encoding="utf-8").splitlines()] for f in files]

    def test_records_every_frame_per_session(self):
        with PluginRun(str(PLUGINS / "result-recorder"), {"keypoints": True}) as run:
            run.wait_state("running", timeout=15)
            time.sleep(0.5)
            summary = run_publisher("--start-delay", 1.0, "--fps", 300, "--frames", 600, "--sessions", 2,
                                    "--gap", 0.5, "--keypoints", 2)
            run.wait_metric("recorded", lambda v: v >= 1200, timeout=15)
            time.sleep(2.5)  # 会话结束后约 1 s 关闭文件
        files, records = self.read_records(run)
        self.assertEqual(len(files), 2)
        for recs, s in zip(records, summary["sessions"]):
            self.assertEqual(recs[0]["type"], "header")
            self.assertEqual(recs[0]["session"], s["session"])
            frames = [r for r in recs if r["type"] == "frame"]
            self.assertEqual([r["frame"] for r in frames], list(range(600)))
            self.assertEqual(len(frames[0]["kpt"][0]), 6)  # 2 个关键点 × (x, y, score)
            self.assertFalse([r for r in recs if r["type"] == "gap"])
            self.assertEqual(recs[-1]["type"], "end")
            self.assertEqual(recs[-1]["frames"], 600)

    def test_gaps_are_recorded(self):
        # 3000 帧/秒、每帧 200 个目标：插件来不及按序读取并写入，环形缓冲被覆盖
        with PluginRun(str(PLUGINS / "result-recorder")) as run:
            run.wait_state("running", timeout=15)
            time.sleep(0.5)
            summary = run_publisher("--start-delay", 1.0, "--fps", 3000, "--frames", 9000, "--objects", 200)
            run.wait_metric("recorded", lambda v: v + (run.metric("lost") or 0) >= summary["published"], timeout=30)
            time.sleep(2.5)
            lost = run.metric("lost")
            health = [e["health"] for e in run.events if e.get("event") == "health" and e["health"]]
        self.assertGreater(lost, 0)
        self.assertIn("degraded", [h["state"] for h in health])  # 丢帧期间报告降级
        files, records = self.read_records(run)
        recs = [r for rs in records for r in rs]
        frames = [r["frame"] for r in recs if r["type"] == "frame"]
        gaps = sum(r["count"] for r in recs if r["type"] == "gap")
        self.assertEqual(len(frames) + gaps, summary["published"])
        self.assertEqual(gaps, lost)
        # 帧号严格递增，缺口记录恰好覆盖缺少的帧
        covered = set(frames)
        for r in recs:
            if r["type"] == "gap":
                covered.update(range(r["after"] + 1, r["after"] + 1 + r["count"]))
        self.assertEqual(covered, set(range(summary["published"])))


class UdpForwardTests(unittest.TestCase):
    """C++ SDK 写的插件：限速、热更新速率、数据报内容、操作。"""

    def setUp(self):
        if not (PLUGINS / "udp-forward" / "bin" / "udp-forward.exe").exists():
            self.skipTest("udp-forward 未构建")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.2)
        self.port = self.sock.getsockname()[1]

    def tearDown(self):
        self.sock.close()

    def drain(self):
        """取出已排队的全部数据报。"""
        out = []
        self.sock.setblocking(False)
        try:
            while True:
                out.append(json.loads(self.sock.recv(70000).decode("utf-8")))
        except BlockingIOError:
            pass
        self.sock.settimeout(0.2)
        return out

    def receive(self, seconds):
        """丢弃已排队的数据报，再接收 seconds 秒内到达的数据报。"""
        self.drain()
        out, end = [], time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                out.append(json.loads(self.sock.recv(70000).decode("utf-8")))
            except socket.timeout:
                pass
        return out

    def test_forwarding(self):
        settings = {"port": self.port, "maxFps": 50, "classes": "person,car", "minScore": 0}
        with PluginRun(str(PLUGINS / "udp-forward"), settings) as run:
            run.wait_state("running", timeout=10)
            pid = run.wait_state("running")["pid"]
            pub = publish("--start-delay", 0.5, "--fps", 300, "--duration", 6, "--objects", 3)
            time.sleep(1.5)
            metas = [p for p in self.drain() if p["type"] == "meta"]  # 会话开始时发送
            frames = [p for p in self.receive(2) if p["type"] == "frame"]
            self.assertTrue(metas)
            self.assertEqual(metas[0]["classNames"]["0"], "person")
            self.assertGreater(len(frames), 80)   # 50 帧/秒 × 2 s
            self.assertLess(len(frames), 115)
            # 合成目标的类别为 0、1、2（person、car、dog）：dog 被过滤
            self.assertEqual({d[5] for f in frames for d in f["det"]}, {0, 1})
            self.assertTrue(all(len(f["region"]) == 4 and abs(f["ts"] - time.time() * 1000) < 5000 for f in frames))
            # 热更新速率：不限速后随发布速率发送
            run.send(cmd="settings", values={**settings, "maxFps": 0})
            run.wait_log("设置已更新")
            time.sleep(0.5)
            fast = [p for p in self.receive(1.5) if p["type"] == "frame"]
            self.assertGreater(len(fast), 300)
            run.send(cmd="action", id="test")
            self.assertIn("test", [p["type"] for p in self.receive(1)])
            run.wait(lambda e: e.get("event") == "notify" and "测试数据报" in e["text"])
            self.assertEqual(run.wait_state("running")["pid"], pid)
            self.assertIsNotNone(run.metric("nr.fps"))
            pub.communicate(timeout=30)

    def test_bad_settings_fail_without_retry(self):
        with PluginRun(str(PLUGINS / "udp-forward"), {"host": ""}) as run:
            e = run.wait_state("failed", timeout=15)
            self.assertIn("目标地址为空", e["error"])
            time.sleep(2)
            self.assertEqual(run.states().count("starting"), 1)  # 插件报告 failed（不重试）：不重启


if __name__ == "__main__":
    for exe in ("nr_plugin_run.exe", "nr_publish.exe"):
        if not (BIN / exe).exists():
            sys.exit(f"找不到 {BIN / exe}，请先构建（cmake --build --preset release）或设置 NR_BIN")
    unittest.main(verbosity=2)
