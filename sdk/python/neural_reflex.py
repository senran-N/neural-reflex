"""NeuralReflex 外部进程插件 SDK（扩展机制方案 5.5、第 7 节）。只依赖 Python 标准库。

插件由宿主启动，通过环境变量取得设置，通过标准输出向宿主报告状态、指标、日志与提示，
通过共享内存读取每帧检测结果。启用了 control 特性的插件还通过标准输入接收事件、
设置热更新、操作请求与停止消息。典型用法：

    import neural_reflex as nr

    @nr.action("reset")
    def reset():
        return "已清零"

    for frame, meta in nr.frames():          # 首次调用时自动报告就绪
        cfg = nr.settings()                  # 热更新后返回新值
        names = meta.get("classNames", {})
        ...
        nr.metric("alerts", count, label="告警次数")

接收策略：frames(mode="latest") 只取最新帧（实时消费者，处理慢时跳过中间的帧）；
frames(mode="all") 按序读取环形缓冲（记录消费者，被覆盖时 frame["gap"] 给出缺口大小）。
frames() 每秒自动上报保留指标 nr.fps、nr.latency、nr.process、nr.skipped、nr.lost。

宿主请求停止（stop 消息或关闭标准输入）时 frames() 随即结束。
"""
import ctypes
import json
import os
import pathlib
import struct
import sys
import threading
import time
from ctypes import wintypes

PROTOCOL = 1   # 插件协议主版本（plugin.json 的 api 字段）
API = "1.1"    # 本 SDK 实现的协议版本

MAPPING = os.environ.get("NR_RESULTS_MAPPING", "Local\\NeuralReflex.Results.v1")
EVENTS = (MAPPING + ".Event0", MAPPING + ".Event1")
MAGIC, LAYOUT = 0x5352524E, 1
MAX_DETECTIONS, MAX_KEYPOINTS = 300, 32


class NRError(Exception):
    """宿主或插件返回的结构化错误（code、message）。"""

    def __init__(self, code, message=""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code, self.message = code, message


# ---------- 宿主传入的信息 ----------


def plugin_id():
    return os.environ.get("NR_PLUGIN_ID", "")


def plugin_dir():
    return pathlib.Path(os.environ.get("NR_PLUGIN_DIR", pathlib.Path.cwd()))


def data_dir():
    """插件可写的数据目录，不存在时创建。"""
    path = pathlib.Path(os.environ.get("NR_PLUGIN_DATA", pathlib.Path.cwd() / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def host_api():
    """宿主协议版本，如 "1.1"；阶段 3 的宿主没有这个变量，返回 "1.0"。"""
    return os.environ.get("NR_API", "1.0")


def host_version():
    return os.environ.get("NR_HOST_VERSION", "")


def features():
    """宿主对本插件实际启用的特性（7.2）。以此判断能力，不以版本号推断。"""
    return [f for f in os.environ.get("NR_FEATURES", "").split(",") if f]


def has_feature(name):
    return name in features()


def _load_settings():
    try:
        value = json.loads(os.environ.get("NR_PLUGIN_SETTINGS", "{}"))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


_settings = _load_settings()


def settings():
    """当前设置（宿主已按 plugin.json 的声明校验并补全缺省值）。live 设置项热更新后返回新值。"""
    return dict(_settings)


# ---------- 向宿主报告：标准输出每行一个 JSON ----------

_out_lock = threading.Lock()


def _emit(message):
    line = json.dumps(message, ensure_ascii=False)
    with _out_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def status(text):
    _emit({"type": "status", "text": str(text)})


def metric(key, value, label=None, unit=None):
    message = {"type": "metric", "key": key, "value": value}
    if label is not None:
        message["label"] = label
    if unit is not None:
        message["unit"] = unit
    _emit(message)


def log(text, level="info", fields=None, **kw):
    """日志；fields 字典与其余关键字参数作为结构化字段一并发送。宿主每秒至多接收 100 行。"""
    message = {"type": "log", "level": level, "text": str(text)}
    merged = {**(fields or {}), **kw}
    if merged:
        message["fields"] = merged
    _emit(message)


def notify(text):
    _emit({"type": "notify", "text": str(text)})


def health(state, text=""):
    """插件自评的健康状态：ok、degraded 或 error（7.3）。"""
    _emit({"type": "health", "state": state, "text": str(text)})


_ready_sent = False


def ready():
    """报告插件可以工作（7.3）。frames() 首次调用时自动发出；初始化较慢的插件应在初始化完成后调用。"""
    global _ready_sent
    if not _ready_sent:
        _ready_sent = True
        _emit({"type": "ready"})


def fail(text, retry=False, exit_code=1):
    """报告无法恢复的错误并退出（7.3）。retry 为 False 时宿主不再重启，直到用户修改设置或手动重启。"""
    _emit({"type": "failed", "text": str(text), "retry": bool(retry)})
    sys.stdout.flush()
    os._exit(exit_code)


def require_modules(*names):
    """检查依赖的模块，缺少时以 fail() 报告并退出。"""
    import importlib.util
    missing = [n for n in names if importlib.util.find_spec(n) is None]
    if missing:
        fail("缺少 Python 模块：" + "、".join(missing) + f"（请用 {sys.executable} -m pip install 安装）")


# ---------- 控制通道（7.4） ----------

_stop = threading.Event()
_control_started = False
_control_lock = threading.Lock()
_settings_handlers, _event_handlers, _actions = [], {}, {}
_diagnose_handler = None
_session = {}
_pending, _pending_lock, _next_request = {}, threading.Lock(), [0]


def on_settings(func):
    """装饰器：live 设置项热更新时调用 func(values)。在控制线程中调用，不要阻塞。"""
    _settings_handlers.append(func)
    return func


def on_event(name):
    """装饰器：收到宿主事件时调用 func(data)，如 on_event("session")。在控制线程中调用，不要阻塞。"""
    def register(func):
        _event_handlers.setdefault(name, []).append(func)
        return func
    return register


def action(action_id):
    """装饰器：处理 plugin.json 中声明的操作。func() 返回字符串（显示为提示）、字典（原样作为结果）或 None。
    在独立线程中调用；抛出异常时宿主显示失败原因。"""
    def register(func):
        _actions[action_id] = func
        return func
    return register


def on_diagnose(func):
    """装饰器：导出诊断时调用 func()，返回的字典写入诊断文件。"""
    global _diagnose_handler
    _diagnose_handler = func
    return func


def session():
    """最近一次 session 事件的数据（state、session、publishing、model）；未收到时为空字典。"""
    return dict(_session)


def request(method, params=None, timeout=5.0):
    """向宿主发请求并等待回答（需要 control 特性）。失败时抛出 NRError。"""
    if not has_feature("control"):
        raise NRError("unsupported", "宿主未对本插件启用 control 特性")
    _start_control()
    with _pending_lock:
        _next_request[0] += 1
        rid = _next_request[0]
        slot = {"event": threading.Event(), "message": None}
        _pending[rid] = slot
    _emit({"type": "request", "id": rid, "method": method, "params": params or {}})
    if not slot["event"].wait(timeout):
        with _pending_lock:
            _pending.pop(rid, None)
        raise NRError("timeout", f"宿主未在 {timeout} 秒内回答 {method}")
    message = slot["message"]
    if "error" in message:
        error = message["error"] if isinstance(message["error"], dict) else {"code": "failed", "message": str(message["error"])}
        raise NRError(error.get("code", "failed"), error.get("message", ""))
    return message.get("result")


def _call(func, *args):
    try:
        func(*args)
    except Exception as e:  # 回调中的异常不能结束控制线程
        log(f"回调 {getattr(func, '__name__', func)} 出错：{e!r}", level="error")


def _answer(rid, method, params):
    try:
        if method == "action":
            handler = _actions.get((params or {}).get("id"))
            if handler is None:
                raise NRError("invalid_params", f"没有操作 {(params or {}).get('id')}")
            result = handler()
            result = {"text": result} if isinstance(result, str) else (result if isinstance(result, dict) else {})
        elif method == "diagnose":
            result = {"sdk": "python", "api": API, "consumer": dict(_consumer)}
            if _diagnose_handler is not None:
                result["plugin"] = _diagnose_handler()
        else:
            raise NRError("unknown_method", f"插件没有方法 {method}")
        _emit({"type": "response", "id": rid, "result": result})
    except NRError as e:
        _emit({"type": "response", "id": rid, "error": {"code": e.code, "message": e.message}})
    except Exception as e:
        _emit({"type": "response", "id": rid, "error": {"code": "failed", "message": str(e) or repr(e)}})


def _dispatch(message):
    global _settings
    kind = message.get("type")
    if kind == "stop":
        _stop.set()
    elif kind == "settings" and isinstance(message.get("values"), dict):
        _settings = dict(message["values"])
        for func in list(_settings_handlers):
            _call(func, settings())
    elif kind == "event":
        name, data = message.get("name"), message.get("data")
        if name == "session" and isinstance(data, dict):
            _session.clear()
            _session.update(data)
        for func in list(_event_handlers.get(name, [])):
            _call(func, data)
    elif kind == "request":
        threading.Thread(target=_answer, args=(message.get("id"), message.get("method"), message.get("params")),
                         daemon=True).start()
    elif kind == "response":
        with _pending_lock:
            slot = _pending.pop(message.get("id"), None)
        if slot is not None:
            slot["message"] = message
            slot["event"].set()
    # cancel 与未知类型：忽略（7.2 兼容规则）


def _control_loop():
    # 直接读文件描述符 0：守护线程阻塞在 sys.stdin 的缓冲读取上时，解释器退出会因取不到锁而崩溃
    control = has_feature("control")
    pending = b""
    try:
        while True:
            chunk = os.read(0, 65536)
            if not chunk:
                break
            if not control:
                continue  # 阶段 3 的语义：宿主关闭标准输入表示请求停止
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for raw in lines:
                try:
                    message = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(message, dict):
                    _dispatch(message)
    except OSError:
        pass
    _stop.set()


def _start_control():
    global _control_started
    with _control_lock:
        if _control_started or not plugin_id():
            return
        _control_started = True
    threading.Thread(target=_control_loop, name="nr-control", daemon=True).start()


def stop_requested():
    """宿主是否已请求停止。仅在宿主托管时（存在 NR_PLUGIN_ID）监视标准输入。"""
    _start_control()
    return _stop.is_set()


def wait_stop(timeout=None):
    """等待宿主请求停止；返回是否已请求停止。"""
    _start_control()
    return _stop.wait(timeout)


# 宿主托管时立即开始处理控制通道，以免错过启动时的事件与设置
_start_control()


# ---------- 结果共享内存（布局见扩展机制方案 5.2、7.5） ----------

FILE_MAP_READ, SYNCHRONIZE, WAIT_OBJECT_0 = 0x0004, 0x00100000, 0
HEADER = struct.Struct("<IIIIqqIIIIq")      # 56 字节
FRAME_HEAD = struct.Struct("<Qqqiiiiiiii")  # 帧号、两个时刻、检测区域、目标数、关键点数、候选数、溢出数
DETECTION = struct.Struct("<fffffi")        # x1, y1, x2, y2, score, cls
KEYPOINT = struct.Struct("<fff")            # x, y, score
RING_HEADER = struct.Struct("<III")         # 槽数、槽跨度、槽内帧记录偏移
SLOT_HEAD = struct.Struct("<qq")            # 槽序号、会话号

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.OpenFileMappingW.restype = wintypes.HANDLE
_k32.OpenFileMappingW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_k32.MapViewOfFile.restype = ctypes.c_void_p
_k32.MapViewOfFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_size_t]
_k32.OpenEventW.restype = wintypes.HANDLE
_k32.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]


def qpc():
    v = ctypes.c_int64()
    _k32.QueryPerformanceCounter(ctypes.byref(v))
    return v.value


class Results:
    """直接读取共享内存。一般使用 frames()；需要自行控制等待时使用本类。"""

    def __init__(self):
        mapping = _k32.OpenFileMappingW(FILE_MAP_READ, False, MAPPING)
        if not mapping:
            raise FileNotFoundError("没有找到结果共享内存：宿主未开启结果共享或尚未开始检测")
        self.base = _k32.MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0)
        self.events = [_k32.OpenEventW(SYNCHRONIZE, False, name) for name in EVENTS]
        if not self.base or not all(self.events):
            raise OSError("打开结果共享内存或通知事件失败")
        h = HEADER.unpack(ctypes.string_at(self.base, HEADER.size))
        magic, version, self.total, self.frame_offset = h[:4]
        self.meta_offset, self.meta_capacity, self.qpc_frequency = h[6], h[7], h[10]
        if magic != MAGIC or version != LAYOUT:
            raise ValueError(f"布局不兼容：魔数 {magic:#x}，版本 {version}（SDK 支持版本 {LAYOUT}）")
        # 环形区（7.5）：偏移 56 在帧记录之前才属于头部；阶段 2 的宿主在这里为 0
        self.ring_offset = ctypes.c_uint32.from_address(self.base + 56).value if self.frame_offset >= 64 else 0
        self.ring_slots = self.ring_stride = self.ring_record = 0
        if self.ring_offset:
            self.ring_slots, self.ring_stride, self.ring_record = RING_HEADER.unpack(
                ctypes.string_at(self.base + self.ring_offset, RING_HEADER.size))

    @property
    def has_ring(self):
        return self.ring_slots > 0

    def _seq(self, offset):
        return ctypes.c_int64.from_address(self.base + offset).value

    def published(self):
        """已发布帧数（帧序号锁 / 2）。"""
        return self._seq(16) // 2

    def metadata_version(self):
        return self._seq(24)

    def metadata(self):
        while True:
            s1 = self._seq(24)
            if s1 & 1:
                continue
            length = ctypes.c_uint32.from_address(self.base + 40).value
            data = ctypes.string_at(self.base + self.meta_offset, length)
            if self._seq(24) == s1:
                return json.loads(data.decode("utf-8")) if data else {}

    def _record(self, address):
        """复制一条帧记录（不检查一致性，由调用方按序号锁核对）。"""
        head = FRAME_HEAD.unpack(ctypes.string_at(address, FRAME_HEAD.size))
        count, num_kpt = max(0, min(head[7], MAX_DETECTIONS)), max(0, min(head[8], MAX_KEYPOINTS))
        dets = ctypes.string_at(address + FRAME_HEAD.size, count * DETECTION.size)
        kpts = ctypes.string_at(address + FRAME_HEAD.size + MAX_DETECTIONS * DETECTION.size,
                                count * num_kpt * KEYPOINT.size)
        return head, count, num_kpt, dets, kpts

    @staticmethod
    def _frame(head, count, num_kpt, dets, kpts):
        frame = {
            "frame": head[0], "acquire_qpc": head[1], "result_qpc": head[2],
            "region": list(head[3:7]), "candidates": head[9], "overflow": head[10],
            "det": [list(DETECTION.unpack_from(dets, i * DETECTION.size)) for i in range(count)],
        }
        if num_kpt:
            pts = [KEYPOINT.unpack_from(kpts, i * KEYPOINT.size) for i in range(count * num_kpt)]
            frame["kpt"] = [[v for p in pts[i * num_kpt:(i + 1) * num_kpt] for v in p] for i in range(count)]
        return frame

    def read_frame(self):
        """按序号锁读取最新的一帧；返回 (已发布帧数, 帧字典)。"""
        while True:
            s1 = self._seq(16)
            if s1 & 1:
                continue
            raw = self._record(self.base + self.frame_offset)
            if self._seq(16) != s1:
                continue  # 读取期间宿主写入了新的一帧，重读
            return s1 // 2, self._frame(*raw)

    def read_seq(self, k):
        """从环形缓冲读取第 k 次发布的帧（7.5）。返回 ("ok", 会话号, 帧)、("overrun", None, None)
        （已被覆盖）或 ("pending", None, None)（尚未写完）。"""
        slot = self.base + self.ring_offset + 64 + ((k - 1) % self.ring_slots) * self.ring_stride
        s1, session = SLOT_HEAD.unpack(ctypes.string_at(slot, SLOT_HEAD.size))
        if s1 < 2 * k:
            return "pending", None, None
        if s1 > 2 * k:
            return "overrun", None, None
        raw = self._record(slot + self.ring_record)
        if self._seq_at(slot) != s1:
            return "overrun", None, None
        return "ok", session, self._frame(*raw)

    def _seq_at(self, address):
        return ctypes.c_int64.from_address(address).value

    def first_buffered(self, session):
        """环形缓冲中仍保留的、属于 session 的最早一帧的发布序号；没有时返回已发布帧数 + 1。
        留出 1/8 槽的余量，以免刚开始读就被覆盖。"""
        published = self.published()
        k = max(1, published - self.ring_slots + 1 + self.ring_slots // 8)
        while k <= published:
            slot = self.base + self.ring_offset + 64 + ((k - 1) % self.ring_slots) * self.ring_stride
            s1, slot_session = SLOT_HEAD.unpack(ctypes.string_at(slot, SLOT_HEAD.size))
            if s1 == 2 * k and slot_session == session:
                return k
            k += 1
        return published + 1

    def wait_next(self, seen, timeout_ms):
        """等待已发布帧数超过 seen。宿主发布第 k 帧时设置 Event[k % 2]，并预先重置 Event[(k + 1) % 2]。"""
        if self.published() > seen:
            return True
        return _k32.WaitForSingleObject(self.events[(seen + 1) % 2], timeout_ms) == WAIT_OBJECT_0

    def latency_ms(self, frame):
        """结果可用到现在的时间。"""
        return (qpc() - frame["result_qpc"]) * 1000.0 / self.qpc_frequency


def open_results(on_wait=None, timeout=None):
    """打开结果共享内存，宿主尚未发布时每 0.2 s 重试；on_wait 在第一次等待时调用。
    宿主请求停止或超过 timeout 秒时返回 None。"""
    waited, deadline = False, None if timeout is None else time.time() + timeout
    while not stop_requested() and (deadline is None or time.time() < deadline):
        try:
            return Results()
        except FileNotFoundError:
            if not waited and on_wait:
                on_wait()
            waited = True
            time.sleep(0.2)
    return None


# 消费统计（保留指标与 diagnose 回答共用）
_consumer = {"mode": None, "frames": 0, "skipped": 0, "lost": 0}


class _Reporter:
    """每秒上报保留指标（7.6）：处理帧率、送达延迟中位数、每帧处理耗时 P95、累计跳过与丢失帧数。"""

    def __init__(self, enabled):
        self.enabled = enabled and bool(plugin_id())
        self.start, self.frames, self.latency, self.process = time.monotonic(), 0, [], []

    def frame(self, latency_ms):
        self.frames += 1
        self.latency.append(latency_ms)
        self.tick()

    def tick(self):
        """没有新帧时也每秒上报一次，指标随之归零或保持最新。"""
        now = time.monotonic()
        if now - self.start >= 1:
            self.flush(now)

    def processed(self, process_ms):
        self.process.append(process_ms)

    def flush(self, now):
        if self.enabled:
            metric("nr.fps", round(self.frames / (now - self.start), 1))
            if self.latency:
                self.latency.sort()
                metric("nr.latency", round(self.latency[len(self.latency) // 2], 3))
            if self.process:
                self.process.sort()
                metric("nr.process", round(self.process[min(len(self.process) - 1, int(len(self.process) * 0.95))], 3))
            metric("nr.skipped", _consumer["skipped"])
            metric("nr.lost", _consumer["lost"])
        self.start, self.frames, self.latency, self.process = now, 0, [], []


def frames(mode="latest", max_fps=None, on_wait=None, timeout_ms=200, report=True):
    """逐帧产出 (frame, meta)。meta 为帧所属会话的元数据（会话切换时自动更新）。

    mode="latest"：只取最新帧，处理慢于检测时跳过中间的帧；max_fps 限制本插件的消费速率。
    mode="all"：按序读取环形缓冲，从当前会话仍在缓冲中的最早一帧开始（会话已停止时从下一帧开始）；
    读取慢于检测导致覆盖时跳到仍在缓冲中的帧。宿主不支持环形缓冲时
    退回 latest 并记录警告。
    frame["gap"] 为与上一次产出的帧之间缺少的帧数（同一会话内；latest 模式下为跳过，all 模式下为丢失）。
    report=True 时每秒上报保留指标。宿主请求停止时结束。
    """
    ready()
    results = open_results(on_wait)
    if results is None:
        return
    if mode == "all" and not results.has_ring:
        log("宿主的结果共享没有环形缓冲，按 latest 模式读取", level="warning")
        mode = "latest"
    _consumer["mode"] = mode
    reporter = _Reporter(report)
    min_interval = 1.0 / max_fps if (max_fps and mode == "latest") else 0.0
    meta_version, meta = results.metadata_version(), results.metadata()
    metas = {meta.get("session"): meta}  # 会话号 → 元数据；all 模式下可能还在读上一个会话的帧
    last_index, last_session = None, None
    last_yield = next_due = 0.0
    seen = results.published()
    next_k = seen + 1
    if mode == "all" and meta.get("state") == "running":
        # 打开映射前宿主可能已发布了本会话的若干帧，它们仍在环形缓冲中
        next_k = results.first_buffered(meta.get("session"))

    while not stop_requested():
        if results.metadata_version() != meta_version:
            meta_version, meta = results.metadata_version(), results.metadata()
            metas[meta.get("session")] = meta
            while len(metas) > 4:
                metas.pop(next(iter(metas)))

        if mode == "all":
            if results.published() < next_k:
                if not results.wait_next(next_k - 1, timeout_ms):
                    reporter.tick()
                continue
            state, session_id, frame = results.read_seq(next_k)
            if state == "pending":
                time.sleep(0)
                continue
            if state == "overrun":
                # 跳到仍在缓冲中的帧，并留出余量以免立即再次被覆盖
                oldest = results.published() - results.ring_slots + 1
                target = max(next_k + 1, oldest + results.ring_slots // 8)
                _consumer["lost"] += target - next_k
                next_k = target
                continue
            next_k += 1
            frame_meta = metas.get(session_id, meta)
        else:
            if min_interval:
                # 未到交付时刻：等下一帧到达再判断，以帧到达为节拍
                now = time.monotonic()
                if now < next_due:
                    results.wait_next(results.published(), max(1, min(timeout_ms, int((next_due - now) * 1000) + 1)))
                    reporter.tick()
                    continue
            if not results.wait_next(seen, timeout_ms):
                reporter.tick()
                continue
            seen, frame = results.read_frame()
            session_id, frame_meta = meta.get("session"), meta
            if min_interval:
                # 按计划时刻推进，长期平均速率等于上限；落后超过一个间隔时从现在重新计时
                now = time.monotonic()
                next_due = now + min_interval if now - next_due > min_interval else next_due + min_interval

        # 缺口：同一会话内帧号不连续的部分
        gap = 0
        if session_id == last_session and last_index is not None and frame["frame"] > last_index + 1:
            gap = frame["frame"] - last_index - 1
            if mode == "latest":
                _consumer["skipped"] += gap
        last_index, last_session = frame["frame"], session_id
        frame["gap"] = gap
        frame["session"] = session_id
        frame["latency_ms"] = results.latency_ms(frame)
        _consumer["frames"] += 1
        reporter.frame(frame["latency_ms"])
        last_yield = time.monotonic()
        yield frame, frame_meta
        reporter.processed((time.monotonic() - last_yield) * 1000.0)  # 插件处理这一帧的时间
