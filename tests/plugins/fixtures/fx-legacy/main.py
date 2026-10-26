"""阶段 3 风格的插件：不声明特性，宿主关闭标准输入表示停止。"""
import os
import time

import neural_reflex as nr

nr.log("env", api=os.environ.get("NR_API"), features=os.environ.get("NR_FEATURES"))
nr.status("legacy running")
while not nr.stop_requested():
    time.sleep(0.05)
nr.log("legacy stopping")
