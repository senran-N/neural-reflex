"""测试插件：按 mode 设置表现不同的生命周期行为。"""
import sys
import time

import neural_reflex as nr

cfg = nr.settings()
mode = cfg.get("mode", "normal")
nr.log("env", api=nr.host_api(), features=nr.features(), host=nr.host_version())


@nr.on_settings
def changed(values):
    nr.log("settings", fields=values)


@nr.on_event("session")
def session(data):
    nr.log("session " + str(data.get("state")), session_id=data.get("session"))


@nr.action("echo")
def echo():
    return "echo " + str(nr.settings().get("label"))


@nr.action("slow")
def slow():
    time.sleep(15)
    return "late"


@nr.action("boom")
def boom():
    raise RuntimeError("kaboom")


@nr.on_diagnose
def diagnose():
    return {"mode": mode, "level": nr.settings().get("level")}


time.sleep(cfg.get("delay", 0) / 1000)
if mode == "fail":
    nr.fail("bad config")
if mode == "fail-retry":
    nr.fail("temporary problem", retry=True)
if mode == "crash":
    print("boom-line", file=sys.stderr, flush=True)
    time.sleep(0.2)
    sys.exit(3)
if mode == "log-flood":
    for i in range(3000):
        nr.log(f"line {i}")

nr.ready()
info = nr.request("host.info")
nr.log("host.info", api=info.get("api"), features=info.get("features"))
state = nr.request("session.state")
nr.log("session.state", state=state.get("state"))
try:
    nr.request("no.such.method")
except nr.NRError as e:
    nr.log("request error", code=e.code)
nr.health("ok", "fine")
nr.status("running " + mode)
while not nr.wait_stop(0.1):
    pass
nr.log("stopping")
