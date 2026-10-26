#pragma once

// NeuralReflex 外部进程插件 C++ SDK（扩展机制方案 5.5、第 7 节）。只有头文件，依赖 nlohmann/json
// 与 Windows SDK，与 sdk/python/neural_reflex.py 功能对应。典型用法：
//
//   nr::sdk::Plugin plugin;
//   plugin.OnAction("test", [&] { return nlohmann::json("已完成"); });
//   nr::sdk::ForEachFrame(plugin, {.mode = "latest", .maxFps = 60}, [&](const nr::sdk::Frame& f, const nlohmann::json& meta) {
//     ...
//     return true;  // false 结束读取
//   });
//
// Plugin 对象应在整个进程期间存在：控制线程与请求处理线程引用它。

#include <nlohmann/json.hpp>

#include <windows.h>

#include <fcntl.h>
#include <io.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nr::sdk {

using json = nlohmann::json;

inline constexpr int kProtocol = 1;     // 插件协议主版本（plugin.json 的 api 字段）
inline constexpr char kApi[] = "1.1";   // 本 SDK 实现的协议版本
inline constexpr int kMaxDetections = 300, kMaxKeypoints = 32;

// 宿主或插件返回的结构化错误
struct Error : std::runtime_error {
  std::string code;
  Error(std::string c, const std::string& message) : std::runtime_error(c + ": " + message), code(std::move(c)) {}
};

namespace detail {

inline std::string Env(const char* name, const std::string& fallback = "") {
  char* value = nullptr;
  size_t size = 0;
  if (_dupenv_s(&value, &size, name) != 0 || !value) return fallback;
  std::string s(value);
  std::free(value);
  return s;
}

inline std::wstring Wide(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return w;
}

inline int64_t Qpc() {
  LARGE_INTEGER v;
  QueryPerformanceCounter(&v);
  return v.QuadPart;
}

}  // namespace detail

class Plugin {
 public:
  Plugin() {
    _setmode(_fileno(stdout), _O_BINARY);
    id_ = detail::Env("NR_PLUGIN_ID");
    for (const std::string& f : Split(detail::Env("NR_FEATURES"))) features_.push_back(f);
    try {
      settings_ = json::parse(detail::Env("NR_PLUGIN_SETTINGS", "{}"));
    } catch (const json::exception&) {
      settings_ = json::object();
    }
    if (!settings_.is_object()) settings_ = json::object();
    // 宿主托管时立即开始处理控制通道，以免错过启动时的事件与设置
    if (!id_.empty()) std::thread(&Plugin::ControlLoop, this).detach();
  }
  Plugin(const Plugin&) = delete;
  Plugin& operator=(const Plugin&) = delete;

  // ---------- 宿主传入的信息 ----------
  const std::string& Id() const { return id_; }
  std::filesystem::path Dir() const { return detail::Wide(detail::Env("NR_PLUGIN_DIR", ".")); }
  std::filesystem::path DataDir() const {
    std::filesystem::path p = detail::Wide(detail::Env("NR_PLUGIN_DATA", "data"));
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return p;
  }
  std::string HostApi() const { return detail::Env("NR_API", "1.0"); }
  std::string HostVersion() const { return detail::Env("NR_HOST_VERSION"); }
  std::string ResultsMapping() const { return detail::Env("NR_RESULTS_MAPPING", "Local\\NeuralReflex.Results.v1"); }
  const std::vector<std::string>& Features() const { return features_; }
  bool HasFeature(const std::string& f) const { return std::find(features_.begin(), features_.end(), f) != features_.end(); }
  // 当前设置；live 设置项热更新后返回新值
  json Settings() const {
    std::lock_guard lock(mutex_);
    return settings_;
  }
  // 最近一次 session 事件的数据
  json Session() const {
    std::lock_guard lock(mutex_);
    return session_;
  }

  // ---------- 向宿主报告 ----------
  void Status(const std::string& text) { Emit({{"type", "status"}, {"text", text}}); }
  void Metric(const std::string& key, const json& value, const std::string& label = "", const std::string& unit = "") {
    json m = {{"type", "metric"}, {"key", key}, {"value", value}};
    if (!label.empty()) m["label"] = label;
    if (!unit.empty()) m["unit"] = unit;
    Emit(m);
  }
  void Log(const std::string& text, const std::string& level = "info", const json& fields = nullptr) {
    json m = {{"type", "log"}, {"level", level}, {"text", text}};
    if (fields.is_object() && !fields.empty()) m["fields"] = fields;
    Emit(m);
  }
  void Notify(const std::string& text) { Emit({{"type", "notify"}, {"text", text}}); }
  void Health(const std::string& state, const std::string& text = "") {
    Emit({{"type", "health"}, {"state", state}, {"text", text}});
  }
  // 报告可以工作（7.3）；ForEachFrame 首次调用时自动发出
  void Ready() {
    if (!ready_.exchange(true)) Emit({{"type", "ready"}});
  }
  // 报告无法恢复的错误并退出；retry 为 false 时宿主不再重启，直到用户修改设置或手动重启
  [[noreturn]] void Fail(const std::string& text, bool retry = false, int exitCode = 1) {
    Emit({{"type", "failed"}, {"text", text}, {"retry", retry}});
    std::fflush(stdout);
    ExitProcess(static_cast<UINT>(exitCode));
  }

  // ---------- 控制通道（7.4） ----------
  // 处理函数在控制线程（设置、事件）或独立线程（操作、诊断）中调用
  void OnSettings(std::function<void(const json&)> f) {
    std::lock_guard lock(mutex_);
    settingsHandlers_.push_back(std::move(f));
  }
  void OnEvent(const std::string& name, std::function<void(const json&)> f) {
    std::lock_guard lock(mutex_);
    eventHandlers_[name].push_back(std::move(f));
  }
  // 返回字符串（显示为提示）或对象（原样作为结果）；抛出异常时宿主显示失败原因
  void OnAction(const std::string& id, std::function<json()> f) {
    std::lock_guard lock(mutex_);
    actions_[id] = std::move(f);
  }
  void OnDiagnose(std::function<json()> f) {
    std::lock_guard lock(mutex_);
    diagnose_ = std::move(f);
  }

  // 向宿主发请求并等待回答（需要 control 特性）；失败时抛出 Error
  json Request(const std::string& method, const json& params = json::object(), int timeoutMs = 5000) {
    if (!HasFeature("control")) throw Error("unsupported", "宿主未对本插件启用 control 特性");
    std::future<json> answer;
    int64_t id;
    {
      std::lock_guard lock(mutex_);
      id = ++nextRequest_;
      answer = pending_[id].get_future();
    }
    Emit({{"type", "request"}, {"id", id}, {"method", method}, {"params", params}});
    if (answer.wait_for(std::chrono::milliseconds(timeoutMs)) != std::future_status::ready) {
      std::lock_guard lock(mutex_);
      pending_.erase(id);
      throw Error("timeout", "宿主未在时限内回答 " + method);
    }
    const json m = answer.get();
    if (m.contains("error")) {
      const json& e = m["error"];
      throw Error(e.is_object() ? e.value("code", "failed") : "failed",
                  e.is_object() ? e.value("message", "") : e.dump());
    }
    return m.value("result", json());
  }

  bool StopRequested() const { return stop_; }
  // 等待宿主请求停止；返回是否已请求停止
  bool WaitStop(int timeoutMs) {
    std::unique_lock lock(mutex_);
    return stopCv_.wait_for(lock, std::chrono::milliseconds(timeoutMs), [&] { return stop_.load(); });
  }

  // 消费统计（保留指标与诊断共用）
  std::atomic<uint64_t> consumedFrames{0}, skippedFrames{0}, lostFrames{0};

 private:
  static std::vector<std::string> Split(const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
      const size_t comma = s.find(',', start);
      const std::string part = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
      if (!part.empty()) out.push_back(part);
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    return out;
  }

  void Emit(const json& m) {
    const std::string line = m.dump(-1, ' ', false, json::error_handler_t::replace) + "\n";
    std::lock_guard lock(outMutex_);
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fflush(stdout);
  }

  void SetStop() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    stopCv_.notify_all();
  }

  void Answer(json id, std::string method, json params) {
    json response = {{"type", "response"}, {"id", id}};
    try {
      std::function<json()> handler;
      {
        std::lock_guard lock(mutex_);
        if (method == "action") {
          const std::string action = params.is_object() ? params.value("id", "") : "";
          auto it = actions_.find(action);
          if (it == actions_.end()) throw Error("invalid_params", "没有操作 " + action);
          handler = it->second;
        } else if (method == "diagnose") {
          handler = diagnose_;
        } else {
          throw Error("unknown_method", "插件没有方法 " + method);
        }
      }
      if (method == "diagnose") {
        json result = {{"sdk", "cpp"},
                       {"api", kApi},
                       {"consumer", {{"frames", consumedFrames.load()}, {"skipped", skippedFrames.load()}, {"lost", lostFrames.load()}}}};
        if (handler) result["plugin"] = handler();
        response["result"] = result;
      } else {
        const json r = handler();
        response["result"] = r.is_string() ? json{{"text", r}} : (r.is_object() ? r : json::object());
      }
    } catch (const Error& e) {
      response["error"] = {{"code", e.code}, {"message", e.what()}};
    } catch (const std::exception& e) {
      response["error"] = {{"code", "failed"}, {"message", e.what()}};
    }
    Emit(response);
  }

  void Dispatch(const json& m) {
    const std::string type = m.value("type", "");
    if (type == "stop") {
      SetStop();
    } else if (type == "settings" && m.contains("values") && m["values"].is_object()) {
      std::vector<std::function<void(const json&)>> handlers;
      {
        std::lock_guard lock(mutex_);
        settings_ = m["values"];
        handlers = settingsHandlers_;
      }
      for (auto& f : handlers) Safe([&] { f(m["values"]); });
    } else if (type == "event") {
      const std::string name = m.value("name", "");
      const json data = m.value("data", json());
      std::vector<std::function<void(const json&)>> handlers;
      {
        std::lock_guard lock(mutex_);
        if (name == "session" && data.is_object()) session_ = data;
        if (auto it = eventHandlers_.find(name); it != eventHandlers_.end()) handlers = it->second;
      }
      for (auto& f : handlers) Safe([&] { f(data); });
    } else if (type == "request") {
      std::thread(&Plugin::Answer, this, m.value("id", json()), m.value("method", ""), m.value("params", json()))
          .detach();
    } else if (type == "response" && m.contains("id") && m["id"].is_number_integer()) {
      std::lock_guard lock(mutex_);
      if (auto it = pending_.find(m["id"].get<int64_t>()); it != pending_.end()) {
        it->second.set_value(m);
        pending_.erase(it);
      }
    }
    // cancel 与未知类型：忽略（7.2 兼容规则）
  }

  template <class F>
  void Safe(F&& f) {
    try {
      f();
    } catch (const std::exception& e) {
      Log(std::string("回调出错：") + e.what(), "error");
    }
  }

  void ControlLoop() {
    const bool control = HasFeature("control");
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    std::string pending;
    char buf[65536];
    DWORD got = 0;
    while (ReadFile(in, buf, sizeof(buf), &got, nullptr) && got > 0) {
      if (!control) continue;  // 阶段 3 的语义：宿主关闭标准输入表示请求停止
      pending.append(buf, got);
      size_t start = 0;
      for (size_t pos; (pos = pending.find('\n', start)) != std::string::npos; start = pos + 1) {
        try {
          const json m = json::parse(pending.begin() + static_cast<std::ptrdiff_t>(start),
                                     pending.begin() + static_cast<std::ptrdiff_t>(pos));
          if (m.is_object()) Dispatch(m);
        } catch (const json::exception&) {
        }
      }
      pending.erase(0, start);
    }
    SetStop();
  }

  std::string id_;
  std::vector<std::string> features_;
  mutable std::mutex mutex_, outMutex_;
  std::condition_variable stopCv_;
  std::atomic<bool> stop_{false}, ready_{false};
  json settings_, session_ = json::object();
  std::vector<std::function<void(const json&)>> settingsHandlers_;
  std::map<std::string, std::vector<std::function<void(const json&)>>> eventHandlers_;
  std::map<std::string, std::function<json()>> actions_;
  std::function<json()> diagnose_;
  int64_t nextRequest_ = 0;
  std::map<int64_t, std::promise<json>> pending_;
};

// ---------- 结果共享内存（5.2、7.5） ----------

struct Detection {
  float x1, y1, x2, y2, score;
  int32_t cls;
};
static_assert(sizeof(Detection) == 24);

struct Frame {
  int64_t session = 0;
  uint64_t index = 0;  // 会话内帧号
  int64_t acquireQpc = 0, resultQpc = 0;
  int32_t region[4] = {};  // x, y, w, h，屏幕物理像素
  int32_t candidates = 0, overflow = 0;
  int numKeypoints = 0;
  std::vector<Detection> det;
  std::vector<float> kpt;  // 第 i 个目标的第 j 个点：kpt[(i * numKeypoints + j) * 3 + 0..2] = x, y, score
  uint64_t gap = 0;        // 与上一次产出的帧之间缺少的帧数（同一会话内）
  double latencyMs = 0;    // 结果可用到读取完成
};

class Results {
 public:
  // 共享内存不存在时返回空指针
  static std::unique_ptr<Results> Open(const std::string& mapping) {
    const std::wstring name = detail::Wide(mapping);
    HANDLE m = OpenFileMappingW(FILE_MAP_READ, FALSE, name.c_str());
    if (!m) return nullptr;
    auto r = std::unique_ptr<Results>(new Results());
    r->mapping_ = m;
    r->base_ = static_cast<const uint8_t*>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0));
    for (int i = 0; i < 2; ++i) r->events_[i] = OpenEventW(SYNCHRONIZE, FALSE, (name + (i ? L".Event1" : L".Event0")).c_str());
    if (!r->base_ || !r->events_[0] || !r->events_[1]) throw std::runtime_error("打开结果共享内存或通知事件失败");
    if (r->U32(0) != 0x5352524E || r->U32(4) != 1) throw std::runtime_error("结果共享内存的布局不兼容");
    r->frameOffset_ = r->U32(12);
    r->metaOffset_ = r->U32(32);
    r->qpcFrequency_ = r->I64(48);
    r->ringOffset_ = r->frameOffset_ >= 64 ? r->U32(56) : 0;
    if (r->ringOffset_) {
      r->ringSlots_ = r->U32(r->ringOffset_);
      r->ringStride_ = r->U32(r->ringOffset_ + 4);
      r->ringRecord_ = r->U32(r->ringOffset_ + 8);
    }
    return r;
  }
  ~Results() {
    if (base_) UnmapViewOfFile(base_);
    for (HANDLE e : events_) {
      if (e) CloseHandle(e);
    }
    if (mapping_) CloseHandle(mapping_);
  }

  bool HasRing() const { return ringSlots_ > 0; }
  uint32_t RingSlots() const { return ringSlots_; }
  int64_t Published() const { return Seq(16) / 2; }
  int64_t MetadataVersion() const { return Seq(24); }

  json Metadata() const {
    for (;;) {
      const int64_t s1 = Seq(24);
      if (s1 & 1) continue;
      const uint32_t length = U32(40);
      std::string data(reinterpret_cast<const char*>(base_ + metaOffset_), length);
      if (Seq(24) == s1) return data.empty() ? json::object() : json::parse(data, nullptr, false);
    }
  }

  // 最新的一帧；返回已发布帧数
  int64_t ReadLatest(Frame& f) const {
    for (;;) {
      const int64_t s1 = Seq(16);
      if (s1 & 1) continue;
      ReadRecord(base_ + frameOffset_, f);
      if (Seq(16) == s1) return s1 / 2;
    }
  }

  enum class Slot { kOk, kOverrun, kPending };
  // 环形缓冲中第 k 次发布的帧（7.5）
  Slot ReadSeq(int64_t k, Frame& f) const {
    const uint8_t* slot = base_ + ringOffset_ + 64 + static_cast<size_t>((k - 1) % ringSlots_) * ringStride_;
    const int64_t s1 = AtomicI64(slot);
    if (s1 < 2 * k) return Slot::kPending;
    if (s1 > 2 * k) return Slot::kOverrun;
    f.session = *reinterpret_cast<const int64_t*>(slot + 8);
    ReadRecord(slot + ringRecord_, f);
    return AtomicI64(slot) == s1 ? Slot::kOk : Slot::kOverrun;
  }

  // 环形缓冲中仍保留的、属于 session 的最早一帧的发布序号；没有时返回已发布帧数 + 1。
  // 留出 1/8 槽的余量，以免刚开始读就被覆盖
  int64_t FirstBuffered(int64_t session) const {
    const int64_t published = Published();
    for (int64_t k = std::max<int64_t>(1, published - ringSlots_ + 1 + ringSlots_ / 8); k <= published; ++k) {
      const uint8_t* slot = base_ + ringOffset_ + 64 + static_cast<size_t>((k - 1) % ringSlots_) * ringStride_;
      if (AtomicI64(slot) == 2 * k && *reinterpret_cast<const int64_t*>(slot + 8) == session) return k;
    }
    return published + 1;
  }

  // 等待已发布帧数超过 seen
  bool WaitNext(int64_t seen, DWORD timeoutMs) const {
    if (Published() > seen) return true;
    return WaitForSingleObject(events_[(seen + 1) % 2], timeoutMs) == WAIT_OBJECT_0;
  }

  double LatencyMs(const Frame& f) const {
    return static_cast<double>(detail::Qpc() - f.resultQpc) * 1000.0 / static_cast<double>(qpcFrequency_);
  }

 private:
  Results() = default;
  uint32_t U32(size_t off) const { return *reinterpret_cast<const uint32_t*>(base_ + off); }
  int64_t I64(size_t off) const { return *reinterpret_cast<const int64_t*>(base_ + off); }
  // 序号读取：对齐的 8 字节读取在 x64 上是原子的。映射是只读的，不能用 Interlocked 系列（它们需要写权限）
  static int64_t AtomicI64(const uint8_t* p) {
    const int64_t v = *reinterpret_cast<const volatile int64_t*>(p);
    std::atomic_thread_fence(std::memory_order_acquire);
    return v;
  }
  int64_t Seq(size_t off) const { return AtomicI64(base_ + off); }

  static void ReadRecord(const uint8_t* r, Frame& f) {
    f.index = *reinterpret_cast<const uint64_t*>(r);
    f.acquireQpc = *reinterpret_cast<const int64_t*>(r + 8);
    f.resultQpc = *reinterpret_cast<const int64_t*>(r + 16);
    std::memcpy(f.region, r + 24, 16);
    const int count = std::clamp(*reinterpret_cast<const int32_t*>(r + 40), 0, kMaxDetections);
    f.numKeypoints = std::clamp(*reinterpret_cast<const int32_t*>(r + 44), 0, kMaxKeypoints);
    f.candidates = *reinterpret_cast<const int32_t*>(r + 48);
    f.overflow = *reinterpret_cast<const int32_t*>(r + 52);
    f.det.resize(static_cast<size_t>(count));
    std::memcpy(f.det.data(), r + 56, sizeof(Detection) * static_cast<size_t>(count));
    f.kpt.resize(static_cast<size_t>(count) * f.numKeypoints * 3);
    std::memcpy(f.kpt.data(), r + 56 + sizeof(Detection) * kMaxDetections, sizeof(float) * f.kpt.size());
  }

  HANDLE mapping_ = nullptr;
  HANDLE events_[2] = {};
  const uint8_t* base_ = nullptr;
  uint32_t frameOffset_ = 0, metaOffset_ = 0, ringOffset_ = 0, ringSlots_ = 0, ringStride_ = 0, ringRecord_ = 0;
  int64_t qpcFrequency_ = 1;
};

struct ReadOptions {
  std::string mode = "latest";  // latest：只取最新帧；all：按序读取环形缓冲
  double maxFps = 0;            // latest 模式下本插件的消费速率上限，0 表示不限
  bool report = true;           // 每秒上报保留指标 nr.*
  DWORD timeoutMs = 200;
};

// 逐帧调用 handler(frame, meta)，直到宿主请求停止或 handler 返回 false。行为与 Python SDK 的 frames() 相同。
inline void ForEachFrame(Plugin& plugin, ReadOptions o, const std::function<bool(const Frame&, const json&)>& handler) {
  using clock = std::chrono::steady_clock;
  plugin.Ready();
  std::unique_ptr<Results> results;
  while (!plugin.StopRequested() && !(results = Results::Open(plugin.ResultsMapping()))) plugin.WaitStop(200);
  if (!results) return;
  if (o.mode == "all" && !results->HasRing()) {
    plugin.Log("宿主的结果共享没有环形缓冲，按 latest 模式读取", "warning");
    o.mode = "latest";
  }
  const bool all = o.mode == "all";
  const auto minInterval = std::chrono::duration<double>(o.maxFps > 0 && !all ? 1.0 / o.maxFps : 0.0);

  int64_t metaVersion = results->MetadataVersion();
  json meta = results->Metadata();
  std::map<int64_t, json> metas = {{meta.value("session", int64_t{0}), meta}};
  int64_t seen = results->Published(), next = seen + 1;
  if (all && meta.value("state", "") == "running") {
    // 打开映射前宿主可能已发布了本会话的若干帧，它们仍在环形缓冲中
    next = results->FirstBuffered(meta.value("session", int64_t{0}));
  }
  int64_t lastSession = -1;
  uint64_t lastIndex = 0;
  bool haveLast = false;
  auto lastYield = clock::time_point{}, nextDue = clock::time_point{};
  auto windowStart = clock::now();
  uint64_t windowFrames = 0;
  std::vector<double> latency, process;
  Frame f;

  const auto tick = [&] {
    const auto now = clock::now();
    const double secs = std::chrono::duration<double>(now - windowStart).count();
    if (secs < 1) return;
    if (o.report && !plugin.Id().empty()) {
      plugin.Metric("nr.fps", std::round(windowFrames / secs * 10) / 10);
      if (!latency.empty()) {
        std::sort(latency.begin(), latency.end());
        plugin.Metric("nr.latency", std::round(latency[latency.size() / 2] * 1000) / 1000);
      }
      if (!process.empty()) {
        std::sort(process.begin(), process.end());
        plugin.Metric("nr.process", std::round(process[std::min(process.size() - 1, process.size() * 95 / 100)] * 1000) / 1000);
      }
      plugin.Metric("nr.skipped", plugin.skippedFrames.load());
      plugin.Metric("nr.lost", plugin.lostFrames.load());
    }
    windowStart = now;
    windowFrames = 0;
    latency.clear();
    process.clear();
  };

  while (!plugin.StopRequested()) {
    if (results->MetadataVersion() != metaVersion) {
      metaVersion = results->MetadataVersion();
      meta = results->Metadata();
      metas[meta.value("session", int64_t{0})] = meta;
      while (metas.size() > 4) metas.erase(metas.begin());
    }
    const json* frameMeta = &meta;
    if (all) {
      if (results->Published() < next) {
        if (!results->WaitNext(next - 1, o.timeoutMs)) tick();
        continue;
      }
      const auto state = results->ReadSeq(next, f);
      if (state == Results::Slot::kPending) {
        YieldProcessor();
        continue;
      }
      if (state == Results::Slot::kOverrun) {
        // 跳到仍在缓冲中的帧，并留出余量以免立即再次被覆盖
        const int64_t oldest = results->Published() - results->RingSlots() + 1;
        const int64_t target = std::max(next + 1, oldest + static_cast<int64_t>(results->RingSlots() / 8));
        plugin.lostFrames += static_cast<uint64_t>(target - next);
        next = target;
        continue;
      }
      ++next;
      if (auto it = metas.find(f.session); it != metas.end()) frameMeta = &it->second;
    } else {
      if (minInterval.count() > 0) {
        // 未到交付时刻：等下一帧到达再判断。以帧到达为节拍，不受系统计时器 15.6 ms 粒度影响
        const auto now = clock::now();
        if (now < nextDue) {
          const auto ms = std::chrono::ceil<std::chrono::milliseconds>(nextDue - now).count();
          results->WaitNext(results->Published(), std::min<DWORD>(o.timeoutMs, static_cast<DWORD>(ms)));
          tick();
          continue;
        }
      }
      if (!results->WaitNext(seen, o.timeoutMs)) {
        tick();
        continue;
      }
      seen = results->ReadLatest(f);
      f.session = meta.value("session", int64_t{0});
      if (minInterval.count() > 0) {
        // 按计划时刻推进，长期平均速率等于上限；落后超过一个间隔时从现在重新计时
        const auto now = clock::now();
        const auto interval = std::chrono::duration_cast<clock::duration>(minInterval);
        nextDue = now - nextDue > interval ? now + interval : nextDue + interval;
      }
    }

    f.gap = 0;
    if (haveLast && f.session == lastSession && f.index > lastIndex + 1) {
      f.gap = f.index - lastIndex - 1;
      if (!all) plugin.skippedFrames += f.gap;
    }
    haveLast = true;
    lastSession = f.session;
    lastIndex = f.index;
    f.latencyMs = results->LatencyMs(f);
    ++plugin.consumedFrames;
    ++windowFrames;
    latency.push_back(f.latencyMs);
    tick();
    lastYield = clock::now();
    if (!handler(f, *frameMeta)) return;
    process.push_back(std::chrono::duration<double, std::milli>(clock::now() - lastYield).count());
  }
}

}  // namespace nr::sdk
