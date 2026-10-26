"""不使用 SDK 的插件：报告就绪后既不读取标准输入，也不响应停止；派生的子进程继承输出管道。"""
import json
import os
import pathlib
import subprocess
import sys
import time

print(json.dumps({"type": "ready"}), flush=True)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
pathlib.Path(os.environ["NR_PLUGIN_DATA"], "child.pid").write_text(str(child.pid))
print(json.dumps({"type": "status", "text": f"child {child.pid}"}), flush=True)
time.sleep(600)
