"""插件协议测试（扩展机制方案 7.7）。只依赖 Python 标准库。

用宿主的同一个插件托管实现（nr_plugin_run）与同一个结果发布实现（nr_publish）验证协议 1.1：
特性协商、就绪与启动超时、failed、重启策略、停止超时与进程树、热更新、请求与超时、
日志限流、控制通道积压、诊断、环形缓冲按序读取与丢帧统计、旧 SDK 兼容。

用法：
  python tests/plugins/test_protocol.py            全部测试
  python tests/plugins/test_protocol.py -k ring    名称含 ring 的测试
可执行文件默认取 build/bin/Release，可用环境变量 NR_BIN 指定。测试使用独立的共享内存名，
不影响正在运行的 NeuralReflex。
"""
import ctypes
import json
import os
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
BIN = pathlib.Path(os.environ.get("NR_BIN", ROOT / "build" / "bin" / "Release"))
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
SDK = ROOT / "sdk" / "python"
LEGACY_SDK = pathlib.Path(__file__).resolve().parent / "legacy_sdk"
MAPPING = f"Local\\NeuralReflex.Results.test-{os.getpid()}"


def process_alive(pid):
    k32 = ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code))
    k32.CloseHandle(ctypes.c_void_p(h))
    return code.value == 259  # STILL_ACTIVE


class PluginRun:
    """在 nr_plugin_run 中托管一个测试插件，收集它输出的事件。"""

    def __init__(self, fixture, settings=None, duration=None):
        self.data = tempfile.mkdtemp(prefix="nrpt-")
        plugin = pathlib.Path(fixture) if pathlib.Path(fixture).is_absolute() else FIXTURES / fixture
        args = [str(BIN / "nr_plugin_run.exe"), str(plugin), "--data-dir", self.data,
                "--sdk", str(SDK), "--mapping", MAPPING, "--resources"]
        if settings:
            args += ["--settings", json.dumps(settings)]
        if duration:
            args += ["--duration", str(duration)]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.events = []
        self.cv = threading.Condition()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for raw in self.proc.stdout:
            try:
                event = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            with self.cv:
                self.events.append(event)
                self.cv.notify_all()

    def mark(self):
        """当前事件数；传给 wait 的 since，只看此后的事件。"""
        with self.cv:
            return len(self.events)

    def wait(self, predicate, timeout=10.0, what="事件", since=0):
        """等待满足条件的事件并返回它；超时则失败并列出最近的事件。"""
        deadline = time.monotonic() + timeout
        with self.cv:
            while True:
                for e in self.events[since:]:
                    if predicate(e):
                        return e
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    recent = "\n".join(json.dumps(e, ensure_ascii=False) for e in self.events[-25:])
                    raise AssertionError(f"等待{what}超时；最近的事件：\n{recent}")
                self.cv.wait(remaining)

    def wait_state(self, state, timeout=10.0, since=0):
        return self.wait(lambda e: e.get("event") == "state" and e.get("state") == state, timeout, f"状态 {state}",
                         since)

    def wait_log(self, text, timeout=10.0, since=0):
        return self.wait(lambda e: e.get("event") == "log" and text in e.get("text", ""), timeout, f"日志“{text}”",
                         since)

    def metric(self, key):
        with self.cv:
            values = [e["value"] for e in self.events if e.get("event") == "metric" and e.get("key") == key]
        return values[-1] if values else None

    def wait_metric(self, key, predicate, timeout=15.0):
        return self.wait(lambda e: e.get("event") == "metric" and e.get("key") == key and predicate(e["value"]),
                         timeout, f"指标 {key}")

    def states(self):
        with self.cv:
            return [e["state"] for e in self.events if e.get("event") == "state"]

    def send(self, **command):
        self.proc.stdin.write((json.dumps(command) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def close(self, timeout=45):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait(timeout)
        self.reader.join(5)
        return self.events[-1] if self.events else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            try:
                self.close()
            except subprocess.TimeoutExpired:
                self.proc.kill()


def publish(*args):
    """后台运行 nr_publish；返回进程，结束后标准输出最后一行为汇总 JSON。"""
    return subprocess.Popen([str(BIN / "nr_publish.exe"), "--mapping", MAPPING, "--quiet", *map(str, args)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def check(fixture):
    out = subprocess.run([str(BIN / "nr_plugin_run.exe"), str(FIXTURES / fixture), "--check"],
                         capture_output=True, timeout=20)
    return out.returncode, json.loads(out.stdout.decode("utf-8"))


class ManifestTests(unittest.TestCase):
    def test_valid(self):
        code, r = check("fx-control")
        self.assertEqual(code, 0)
        self.assertEqual(r["activeFeatures"], ["control"])

    def test_optional_features(self):
        code, r = check("fx-optional")
        self.assertEqual(code, 0)
        self.assertEqual(r["activeFeatures"], ["control"])  # future.thing 宿主不支持，不启用

    def test_incompatible_major(self):
        code, r = check("fx-incompatible")
        self.assertEqual(code, 1)
        self.assertIn("协议主版本 2", r["incompatible"])

    def test_missing_feature(self):
        code, r = check("fx-missing")
        self.assertEqual(code, 1)
        self.assertIn("overlay.draw", r["incompatible"])

    def test_invalid(self):
        code, r = check("fx-invalid")
        self.assertEqual(code, 1)
        self.assertIn("restart", r["invalid"])

    def test_setting_extensions_and_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp)
            manifest = {
                "schema": 1, "id": "test-groups", "api": 1,
                "entry": {"command": "python", "args": ["main.py"]},
                "settingGroups": [
                    {"id": "basic", "title": "基础组", "description": "基础描述"},
                    {"id": "adv", "title": "高级组", "collapsed": True}
                ],
                "settings": [
                    {"key": "enabled", "type": "boolean", "default": True, "group": "basic", "col": 1},
                    {"key": "rate", "type": "number", "default": 10, "unit": "ms", "group": "basic", "col": 1, "dependsOn": {"key": "enabled", "value": True}},
                    {"key": "mode", "type": "enum", "options": [{"value": "a", "label": "A"}], "default": "a", "group": "adv"}
                ]
            }
            (p / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
            out = subprocess.run([str(BIN / "nr_plugin_run.exe"), str(p), "--check"],
                                 capture_output=True, timeout=20)
            self.assertEqual(out.returncode, 0)
            res = json.loads(out.stdout.decode("utf-8"))
            self.assertEqual(len(res["settingGroups"]), 2)
            self.assertEqual(res["settingGroups"][0]["title"], "基础组")
            self.assertEqual(res["settingsSchema"][1]["unit"], "ms")
            self.assertEqual(res["settingsSchema"][1]["col"], 1)
            self.assertEqual(res["settingsSchema"][1]["dependsOn"]["key"], "enabled")

    def test_invalid_depends_on_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp)
            manifest = {
                "schema": 1, "id": "test-invalid-dep", "api": 1,
                "entry": {"command": "python", "args": ["main.py"]},
                "settings": [
                    {"key": "foo", "type": "number", "default": 1, "dependsOn": {"key": "nonexistent", "value": True}}
                ]
            }
            (p / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
            out = subprocess.run([str(BIN / "nr_plugin_run.exe"), str(p), "--check"],
                                 capture_output=True, timeout=20)
            self.assertEqual(out.returncode, 1)
            res = json.loads(out.stdout.decode("utf-8"))
            self.assertIn("未定义的 key", res["invalid"])

    def test_incompatible_not_started(self):
        with PluginRun("fx-missing") as run:
            run.wait_state("incompatible")
            last = run.close()
        self.assertEqual(last["event"], "exit")
        self.assertNotIn("running", run.states())


class LifecycleTests(unittest.TestCase):
    def test_legacy_plugin(self):
        with PluginRun("fx-legacy") as run:
            run.wait_state("running")
            self.assertNotIn("starting", run.states())  # 未启用 control：进程启动即运行
            env = run.wait_log("env")
            self.assertEqual(env["fields"]["api"], "1.1")
            self.assertEqual(env["fields"]["features"], "")
            run.close()
            run.wait_log("legacy stopping", timeout=1)
            run.wait_log("已停止，退出代码 0", timeout=1)

    def test_ready(self):
        with PluginRun("fx-control", {"delay": 800}) as run:
            run.wait_state("starting")
            run.wait_state("running", timeout=10)
            run.wait_log("已就绪")
            run.close()
            run.wait_log("stopping", timeout=1)  # 收到 stop 消息后正常退出

    def test_start_timeout(self):
        with PluginRun("fx-noready") as run:
            run.wait_state("starting")
            e = run.wait(lambda e: e.get("event") == "state" and "启动超时" in e.get("error", ""), timeout=10)
            self.assertEqual(e["state"], "backoff")

    def test_failed_without_retry(self):
        with PluginRun("fx-control", {"mode": "fail"}) as run:
            e = run.wait_state("failed")
            self.assertIn("bad config", e["error"])
            time.sleep(2.5)
            self.assertEqual(run.states().count("starting"), 1)  # 不重启
            run.send(cmd="settings", values={"mode": "normal"})  # 修改设置后重试
            run.wait_state("running", timeout=10)

    def test_failed_with_retry(self):
        with PluginRun("fx-control", {"mode": "fail-retry"}) as run:
            e = run.wait_state("backoff")
            self.assertIn("temporary problem", e["error"])

    def test_crash_reason_from_stderr(self):
        with PluginRun("fx-control", {"mode": "crash"}) as run:
            e = run.wait_state("backoff")
            self.assertIn("代码 3", e["error"])
            self.assertIn("boom-line", e["error"])

    def test_restart_on_failure_clean_exit(self):
        with PluginRun("fx-onfailure") as run:
            run.wait_state("exited")
            time.sleep(1.5)
            self.assertEqual(run.states().count("running"), 1)

    def test_restart_never(self):
        with PluginRun("fx-never") as run:
            e = run.wait_state("failed")
            self.assertIn("代码 5", e["error"])

    def test_stop_timeout_kills_process_tree(self):
        with PluginRun("fx-raw") as run:
            run.wait_state("running")
            run.wait(lambda e: e.get("event") == "status" and e.get("text", "").startswith("child"))
            child = int(pathlib.Path(run.data, "plugin-data", "fx-raw", "child.pid").read_text())
            self.assertTrue(process_alive(child))
            started, mark = time.monotonic(), run.mark()
            run.send(cmd="disable")
            run.wait_log("已结束进程树", timeout=5)
            run.wait_state("stopped", timeout=5, since=mark)
            self.assertLess(time.monotonic() - started, 4)
            time.sleep(0.3)
            self.assertFalse(process_alive(child))

    def test_resources(self):
        with PluginRun("fx-control") as run:
            run.wait_state("running")
            e = run.wait(lambda e: e.get("event") == "resources" and e["resources"].get("cpu") is not None, timeout=5)
            self.assertGreater(e["resources"]["memoryMB"], 1)
            self.assertGreaterEqual(e["resources"]["processes"], 1)


class ControlTests(unittest.TestCase):
    def test_live_settings_and_restart(self):
        with PluginRun("fx-control") as run:
            pid = run.wait_state("running")["pid"]
            run.send(cmd="settings", values={"level": 7, "label": "b"})
            run.wait_log("设置已热更新")
            e = run.wait(lambda e: e.get("event") == "log" and e["text"] == "settings")
            self.assertEqual(e["fields"]["level"], 7)
            self.assertEqual(e["fields"]["label"], "b")
            self.assertEqual(run.states().count("starting"), 1)
            # 改动非 live 设置项：重启插件
            run.send(cmd="settings", values={"level": 7, "label": "b", "restartKey": 3})
            e = run.wait(lambda e: e.get("event") == "state" and e["state"] == "running" and e["pid"] != pid, timeout=10)
            self.assertNotEqual(e["pid"], pid)

    def test_requests_to_host(self):
        with PluginRun("fx-control") as run:
            info = run.wait_log("host.info")
            self.assertEqual(info["fields"]["api"], "1.1")
            self.assertEqual(info["fields"]["features"], ["control"])
            self.assertEqual(run.wait_log("session.state")["fields"]["state"], "idle")
            self.assertEqual(run.wait_log("request error")["fields"]["code"], "unknown_method")

    def test_session_event(self):
        with PluginRun("fx-control") as run:
            run.wait_state("running")
            run.send(cmd="session", data={"state": "running", "session": 42, "publishing": True, "model": "m"})
            e = run.wait_log("session running")
            self.assertEqual(e["fields"]["session_id"], 42)

    def test_actions(self):
        with PluginRun("fx-control", {"label": "x"}) as run:
            run.wait_state("running")
            run.send(cmd="action", id="echo")
            e = run.wait(lambda e: e.get("event") == "notify")
            self.assertEqual((e["level"], e["text"]), ("info", "echo x"))
            run.send(cmd="action", id="boom")
            e = run.wait(lambda e: e.get("event") == "notify" and e["level"] == "error")
            self.assertIn("kaboom", e["text"])
            run.send(cmd="action", id="nope")
            run.wait(lambda e: e.get("event") == "error" and "nope" in e.get("text", ""))

    def test_action_timeout(self):
        with PluginRun("fx-control") as run:
            run.wait_state("running")
            started = time.monotonic()
            run.send(cmd="action", id="slow")
            e = run.wait(lambda e: e.get("event") == "notify" and e["level"] == "error", timeout=15)
            self.assertIn("时限", e["text"])
            self.assertGreater(time.monotonic() - started, 9)

    def test_diagnose(self):
        with PluginRun("fx-control", {"level": 5}) as run:
            run.wait_state("running")
            run.send(cmd="diagnose")
            e = run.wait(lambda e: e.get("event") == "diagnostics")
            d = e["plugins"][0]["diagnose"]
            self.assertEqual(d["sdk"], "python")
            self.assertEqual(d["plugin"], {"mode": "normal", "level": 5})

    def test_log_rate_limit(self):
        with PluginRun("fx-control", {"mode": "log-flood"}) as run:
            run.wait_log("已丢弃", timeout=10)
            run.wait_state("running")

    def test_control_backlog(self):
        with PluginRun("fx-raw") as run:
            run.wait_state("running")
            filler = "x" * 2048
            for i in range(400):
                run.send(cmd="session", data={"state": "running", "session": i, "filler": filler})
            run.wait_log("控制通道积压", timeout=10)
            e = run.wait_state("backoff", timeout=10)
            self.assertIn("控制通道积压", e["error"])


class ResultsTests(unittest.TestCase):
    """结果读取：Python SDK（fx-consumer）。ResultsTestsCpp 以相同的测试检查 C++ SDK。"""
    consumer = "fx-consumer"

    def consume(self, settings, publisher_args, done, timeout=15):
        """托管消费者插件，发布结果，等待 done(run, 发布帧数) 成立或超时。"""
        with PluginRun(self.consumer, settings) as run:
            run.wait_state("running")
            time.sleep(0.5)  # 插件每 0.2 s 尝试打开共享内存
            pub = publish("--start-delay", 1.0, *publisher_args)
            out, _ = pub.communicate(timeout=60)
            summary = json.loads(out.decode("utf-8").strip().splitlines()[-1])
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not done(run, summary["published"]):
                time.sleep(0.2)
            time.sleep(0.5)
            return run, summary

    def test_all_mode_receives_every_frame(self):
        run, summary = self.consume({"mode": "all"}, ["--fps", 500, "--frames", 1000],
                                    lambda r, total: (r.metric("count") or 0) >= total)
        self.assertEqual(run.metric("count"), summary["published"])
        self.assertEqual(run.metric("gaps"), 0)
        self.assertEqual(run.metric("nr.lost"), 0)
        self.assertEqual(run.metric("mismatch"), 0)

    def test_all_mode_starts_from_buffered_frames(self):
        # 插件在会话开始后才打开映射：本会话已发布、仍在环形缓冲中的帧也要收到
        pub = publish("--fps", 60, "--frames", 300)
        time.sleep(0.3)
        with PluginRun(self.consumer, {"mode": "all"}) as run:
            run.wait_state("running")
            out, _ = pub.communicate(timeout=30)
            total = json.loads(out.decode("utf-8").strip().splitlines()[-1])["published"]
            run.wait_metric("count", lambda v: v >= total, timeout=10)
            time.sleep(0.5)
            self.assertEqual(run.metric("count"), total)
            self.assertEqual(run.metric("gaps"), 0)
            self.assertEqual(run.metric("nr.lost"), 0)

    def test_all_mode_slow_consumer_counts_loss(self):
        run, summary = self.consume({"mode": "all", "sleepMs": 4}, ["--fps", 1000, "--frames", 3000],
                                    lambda r, total: (r.metric("count") or 0) + (r.metric("nr.lost") or 0) >= total)
        count, lost, gaps = run.metric("count"), run.metric("nr.lost"), run.metric("gaps")
        self.assertGreater(lost, 0)
        self.assertEqual(count + lost, summary["published"])  # 收到的帧加丢失的帧等于发布的帧
        self.assertEqual(gaps, lost)  # 同一会话内，缺口之和等于丢失帧数

    def test_latest_mode_skips(self):
        run, summary = self.consume({"mode": "latest", "sleepMs": 5}, ["--fps", 1000, "--frames", 2000],
                                    lambda r, total: False, timeout=1.5)
        self.assertGreater(run.metric("nr.skipped"), 0)
        self.assertEqual(run.metric("nr.lost"), 0)
        self.assertEqual(run.metric("gaps"), run.metric("nr.skipped"))
        self.assertLessEqual(run.metric("count") + run.metric("gaps"), summary["published"])

    def test_max_fps(self):
        with PluginRun(self.consumer, {"mode": "latest", "maxFps": 50}) as run:
            run.wait_state("running")
            time.sleep(0.5)
            pub = publish("--start-delay", 0.5, "--fps", 500, "--duration", 4)
            run.wait_metric("nr.fps", lambda v: v > 0, timeout=10)
            time.sleep(2)
            fps = run.metric("nr.fps")
            pub.communicate(timeout=30)
        self.assertGreater(fps, 40)
        self.assertLess(fps, 55)

    def test_sessions_in_all_mode(self):
        run, summary = self.consume({"mode": "all"}, ["--fps", 400, "--frames", 300, "--sessions", 2, "--gap", 0.5],
                                    lambda r, total: (r.metric("count") or 0) >= total)
        sessions = run.metric("sessions")
        self.assertEqual(sorted(sessions.values()), [300, 300])
        self.assertEqual(sorted(int(k) for k in sessions), sorted(s["session"] for s in summary["sessions"]))
        self.assertEqual(run.metric("mismatch"), 0)
        self.assertEqual(run.metric("gaps"), 0)



class ResultsTestsCpp(ResultsTests):
    consumer = "fx-consumer-cpp"

    def setUp(self):
        if not (FIXTURES / "fx-consumer-cpp" / "bin" / "fx-consumer-cpp.exe").exists():
            self.skipTest("fx-consumer-cpp 未构建")


class CompatibilityTests(unittest.TestCase):
    def test_stage3_sdk_reads_new_layout(self):
        script = (
            "import sys, json; sys.path.insert(0, sys.argv[1]); import neural_reflex as nr\n"
            "r = nr.open_results(timeout=10); n = 0; seen = r.published(); last = None; missed = 0\n"
            "print('attached', flush=True)\n"
            "import time; end = time.time() + 10\n"
            "while time.time() < end and n < 500:\n"
            "    if not r.wait_next(seen, 200): continue\n"
            "    seen, f = r.read_frame(); n += 1\n"
            "    if last is not None: missed += f['frame'] - last - 1\n"
            "    last = f['frame']\n"
            "print(json.dumps({'n': n, 'missed': missed, 'det': len(f['det'])}))\n")
        env = dict(os.environ, NR_RESULTS_MAPPING=MAPPING)
        env.pop("NR_PLUGIN_ID", None)
        reader = subprocess.Popen([sys.executable, "-c", script, str(LEGACY_SDK)], stdout=subprocess.PIPE, env=env)
        # 发布方先写入元数据并等待 3 s；读取方报告已接入后才开始发布第一帧
        pub = publish("--start-delay", 3.0, "--fps", 200, "--frames", 500)
        self.assertEqual(reader.stdout.readline().strip(), b"attached")
        pub.communicate(timeout=30)
        out, _ = reader.communicate(timeout=30)
        r = json.loads(out.decode("utf-8"))
        self.assertEqual(r["n"] + r["missed"], 500)
        self.assertGreater(r["n"], 200)
        self.assertEqual(r["det"], 3)


if __name__ == "__main__":
    for exe in ("nr_plugin_run.exe", "nr_publish.exe"):
        if not (BIN / exe).exists():
            sys.exit(f"找不到 {BIN / exe}，请先构建（cmake --build --preset release）或设置 NR_BIN")
    unittest.main(verbosity=2)
