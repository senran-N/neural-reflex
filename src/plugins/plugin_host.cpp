#include "plugins/plugin_host.h"

#include "common/win_util.h"
#include "results/result_publisher.h"

#include <psapi.h>
#include <tlhelp32.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <cwctype>
#include <deque>
#include <fstream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <utility>

namespace nr {

using json = nlohmann::json;

struct ProcessEntry {
  DWORD pid, ppid;
};

namespace {

constexpr size_t kMaxLogLines = 200;
constexpr size_t kSnapshotLogLines = 100;
constexpr uint64_t kNotifyIntervalMs = 2000;
constexpr uint64_t kCrashWindowMs = 60000;
constexpr size_t kCrashLimit = 5;
constexpr uint64_t kMaxBackoffMs = 30000;
constexpr size_t kMaxLineBytes = 1 << 20;     // 单行消息上限（7.4）
constexpr size_t kMaxQueuedMessages = 256;    // 控制通道队列（7.4）
constexpr size_t kMaxPendingRequests = 16;
constexpr uint64_t kActionTimeoutMs = 10000;
constexpr uint64_t kDiagnoseTimeoutMs = 3000;
constexpr double kLogRate = 100, kLogBurst = 200;  // 每秒行数与突发（7.6）
constexpr uint64_t kLogFileBytes = 1 << 20;
constexpr uint64_t kResourceIntervalMs = 1000;

uint64_t NowMs() { return GetTickCount64(); }

int64_t UnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string LocalTimeText(int64_t unixMs) {
  const std::time_t t = static_cast<std::time_t>(unixMs / 1000);
  std::tm tm{};
  localtime_s(&tm, &t);
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d.%03d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(unixMs % 1000));
  return buf;
}

// 按 MSVC 运行时的命令行解析规则加引号
std::wstring Quote(const std::wstring& a) {
  if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
  std::wstring r = L"\"";
  size_t slashes = 0;
  for (wchar_t c : a) {
    if (c == L'\\') {
      ++slashes;
    } else if (c == L'"') {
      r.append(slashes * 2 + 1, L'\\');
      r += L'"';
      slashes = 0;
    } else {
      r.append(slashes, L'\\');
      r += c;
      slashes = 0;
    }
  }
  r.append(slashes * 2, L'\\');
  return r + L"\"";
}

std::wstring Upper(std::wstring s) {
  for (wchar_t& c : s) c = static_cast<wchar_t>(std::towupper(c));
  return s;
}

std::string Seconds(uint64_t ms) {
  char buf[32];
  if (ms % 1000 == 0) std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(ms / 1000));
  else std::snprintf(buf, sizeof(buf), "%.1f", static_cast<double>(ms) / 1000.0);
  return buf;
}

std::string Join(const std::vector<std::string>& items, const char* sep) {
  std::string out;
  for (const auto& s : items) out += (out.empty() ? "" : sep) + s;
  return out;
}

// 插件输出可能不是 UTF-8（例如按系统代码页输出的程序）：按系统代码页转换，避免界面消息中出现无效 UTF-8
std::string SanitizeUtf8(const std::string& s) {
  if (s.empty() ||
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0) > 0) {
    return s;
  }
  const int n = MultiByteToWideChar(CP_ACP, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring w(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_ACP, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
  return ToUtf8(w);
}

// 继承当前环境并覆盖指定变量；按变量名不区分大小写排序
std::wstring BuildEnvironment(const std::map<std::wstring, std::wstring>& overrides) {
  std::map<std::wstring, std::pair<std::wstring, std::wstring>> vars;  // 大写名 → (名, 值)
  if (wchar_t* block = GetEnvironmentStringsW()) {
    for (const wchar_t* p = block; *p; p += wcslen(p) + 1) {
      const std::wstring entry = p;
      const size_t eq = entry.find(L'=', 1);  // 以 '=' 开头的是驱动器当前目录，名字含第一个 '='
      if (eq == std::wstring::npos) continue;
      vars[Upper(entry.substr(0, eq))] = {entry.substr(0, eq), entry.substr(eq + 1)};
    }
    FreeEnvironmentStringsW(block);
  }
  for (const auto& [name, value] : overrides) vars[Upper(name)] = {name, value};
  std::wstring out;
  for (const auto& [key, nv] : vars) out += nv.first + L"=" + nv.second + L'\0';
  return out + L'\0';
}

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("无法读取 " + ToUtf8(path.wstring()));
  std::string s((std::istreambuf_iterator<char>(f)), {});
  if (s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xEF) s.erase(0, 3);  // UTF-8 BOM
  return s;
}

std::string Dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// 插件消息的字段按类型宽松读取：类型不符时用缺省值
std::string Str(const json& m, const char* key, const std::string& fallback = "") {
  return m.contains(key) && m[key].is_string() ? m[key].get<std::string>() : fallback;
}

bool Bool(const json& m, const char* key, bool fallback) {
  return m.contains(key) && m[key].is_boolean() ? m[key].get<bool>() : fallback;
}

json Error(const std::string& code, const std::string& message) { return {{"code", code}, {"message", message}}; }

std::vector<std::string> StringList(const json& m, const char* key) {
  if (!m.contains(key)) return {};
  const json& v = m[key];
  if (!v.is_array() || !std::all_of(v.begin(), v.end(), [](const json& x) { return x.is_string(); })) {
    throw std::runtime_error(std::string(key) + " 须为字符串数组");
  }
  return v.get<std::vector<std::string>>();
}

uint64_t Milliseconds(const json& m, const char* key, uint64_t fallback, uint64_t lo, uint64_t hi) {
  if (!m.contains(key)) return fallback;
  if (!m[key].is_number()) throw std::runtime_error(std::string(key) + " 须为毫秒数");
  return static_cast<uint64_t>(std::clamp(m[key].get<double>(), static_cast<double>(lo), static_cast<double>(hi)));
}

// 设置项声明的校验（扩展机制方案第 6 节、7.4）
void CheckSettingsSchema(const json& schema) {
  if (!schema.is_array()) throw std::runtime_error("settings 须为数组");
  std::vector<std::string> keys;
  for (const json& s : schema) {
    if (!s.is_object()) throw std::runtime_error("设置项须为对象");
    const std::string key = s.value("key", "");
    const std::string type = s.value("type", "");
    if (key.empty()) throw std::runtime_error("设置项缺少 key");
    if (std::find(keys.begin(), keys.end(), key) != keys.end()) throw std::runtime_error("设置项 key 重复：" + key);
    keys.push_back(key);
    if (type != "boolean" && type != "number" && type != "string" && type != "enum") {
      throw std::runtime_error("设置项 " + key + " 的类型不受支持：" + type);
    }
    if (type == "enum" && (!s.contains("options") || !s["options"].is_array() || s["options"].empty())) {
      throw std::runtime_error("设置项 " + key + " 缺少 options");
    }
    if (s.contains("live") && !s["live"].is_boolean()) throw std::runtime_error("设置项 " + key + " 的 live 须为布尔值");
  }
}

void CheckActions(const json& actions) {
  if (!actions.is_array()) throw std::runtime_error("actions 须为数组");
  std::vector<std::string> ids;
  for (const json& a : actions) {
    if (!a.is_object() || !a.contains("id") || !a["id"].is_string() || a["id"].get<std::string>().empty()) {
      throw std::runtime_error("操作须有字符串 id");
    }
    if (!a.contains("label") || !a["label"].is_string()) throw std::runtime_error("操作 " + a["id"].get<std::string>() + " 缺少 label");
    const std::string id = a["id"];
    if (std::find(ids.begin(), ids.end(), id) != ids.end()) throw std::runtime_error("操作 id 重复：" + id);
    ids.push_back(id);
  }
}

json DefaultOf(const json& s) {
  const std::string type = s.value("type", "");
  if (s.contains("default")) return s["default"];
  if (type == "boolean") return false;
  if (type == "number") return s.value("min", 0.0);
  if (type == "enum") return s["options"][0].value("value", json());
  return "";
}

// ---------- 进程树（7.3） ----------
// 作业对象不足以覆盖整个进程树：Python 虚拟环境的启动器把解释器放进自己的、允许静默脱离的作业对象，
// 解释器再派生的进程就不属于插件的作业对象。因此另按父进程号查找后代。

std::vector<ProcessEntry> SnapshotProcesses() {
  std::vector<ProcessEntry> out;
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  if (snap == INVALID_HANDLE_VALUE) return out;
  PROCESSENTRY32W e{sizeof(e)};
  for (BOOL ok = Process32FirstW(snap, &e); ok; ok = Process32NextW(snap, &e)) {
    out.push_back({e.th32ProcessID, e.th32ParentProcessID});
  }
  CloseHandle(snap);
  return out;
}

uint64_t FileTimeValue(const FILETIME& t) {
  return (static_cast<uint64_t>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
}

uint64_t CreationTime(HANDLE process) {
  FILETIME created{}, exited{}, kernel{}, user{};
  return GetProcessTimes(process, &created, &exited, &kernel, &user) ? FileTimeValue(created) : 0;
}

// 根进程的后代（不含根）及其句柄。只接受创建时间不早于父进程的进程，排除进程号被复用的情况；
// 根进程已退出时仍可按它的进程号与创建时间查找。
std::vector<std::pair<DWORD, HANDLE>> OpenDescendants(const std::vector<ProcessEntry>& all, DWORD root,
                                                      uint64_t rootCreated, DWORD access) {
  std::vector<std::pair<DWORD, HANDLE>> out;
  std::vector<std::pair<DWORD, uint64_t>> frontier = {{root, rootCreated}};
  std::vector<DWORD> seen = {root};
  while (!frontier.empty()) {
    const auto [parent, parentCreated] = frontier.back();
    frontier.pop_back();
    for (const ProcessEntry& e : all) {
      if (e.ppid != parent || e.pid == 0 || std::find(seen.begin(), seen.end(), e.pid) != seen.end()) continue;
      HANDLE h = OpenProcess(access | PROCESS_QUERY_LIMITED_INFORMATION, FALSE, e.pid);
      if (!h) continue;
      const uint64_t created = CreationTime(h);
      if (created < parentCreated) {
        CloseHandle(h);
        continue;
      }
      seen.push_back(e.pid);
      out.emplace_back(e.pid, h);
      frontier.emplace_back(e.pid, created);
    }
  }
  return out;
}

// 结束整个进程树：作业对象内的进程，以及按父进程关系找到的后代
void KillTree(DWORD pid, uint64_t created, HANDLE job, HANDLE process, UINT code) {
  const auto descendants = OpenDescendants(SnapshotProcesses(), pid, created, PROCESS_TERMINATE);
  if (job) TerminateJobObject(job, code);
  if (process) TerminateProcess(process, code);
  for (const auto& [id, h] : descendants) {
    TerminateProcess(h, code);
    CloseHandle(h);
  }
}

const json* FindSetting(const json& schema, const std::string& key) {
  for (const json& s : schema) {
    if (s.value("key", "") == key) return &s;
  }
  return nullptr;
}

}  // namespace

const std::vector<std::string>& HostPluginFeatures() {
  static const std::vector<std::string> features = {"control", "results.ring"};
  return features;
}

std::vector<std::string> PluginManifest::ActiveFeatures() const {
  std::vector<std::string> out;
  const auto& host = HostPluginFeatures();
  for (const auto* list : {&features, &optionalFeatures}) {
    for (const auto& f : *list) {
      if (std::find(host.begin(), host.end(), f) != host.end() && std::find(out.begin(), out.end(), f) == out.end()) {
        out.push_back(f);
      }
    }
  }
  return out;
}

bool PluginManifest::Uses(const std::string& feature) const {
  const auto active = ActiveFeatures();
  return std::find(active.begin(), active.end(), feature) != active.end();
}

PluginManifest ReadPluginManifest(const std::filesystem::path& dir) {
  static const std::regex kIdPattern("^[a-z0-9][a-z0-9.-]{0,63}$");
  PluginManifest m;
  m.dir = dir;
  m.id = ToUtf8(dir.filename().wstring());
  try {
    const json j = json::parse(ReadText(dir / "plugin.json"));
    if (!j.is_object()) throw std::runtime_error("须为 JSON 对象");
    if (j.value("schema", 0) != 1) throw std::runtime_error("schema 须为 1");
    m.id = j.value("id", "");
    if (!std::regex_match(m.id, kIdPattern)) throw std::runtime_error("id 只能包含小写字母、数字、- 与 .");
    m.name = j.value("name", m.id);
    m.version = j.value("version", "");
    m.description = j.value("description", "");
    if (!j.contains("api") || !j["api"].is_number_integer()) throw std::runtime_error("api 须为整数（协议主版本）");
    m.api = j["api"].get<int>();
    const json& e = j.at("entry");
    m.command = e.at("command").get<std::string>();
    m.args = e.value("args", std::vector<std::string>{});
    m.settings = j.value("settings", json::array());
    CheckSettingsSchema(m.settings);
    m.features = StringList(j, "features");
    m.optionalFeatures = StringList(j, "optionalFeatures");
    m.startTimeoutMs = Milliseconds(j, "startTimeout", m.startTimeoutMs, 1000, 120000);
    m.stopTimeoutMs = Milliseconds(j, "stopTimeout", m.stopTimeoutMs, 500, 30000);
    m.restart = j.value("restart", "always");
    if (m.restart != "always" && m.restart != "on-failure" && m.restart != "never") {
      throw std::runtime_error("restart 须为 always、on-failure 或 never");
    }
    m.actions = j.value("actions", json::array());
    CheckActions(m.actions);
  } catch (const std::exception& ex) {
    m.invalid = std::string("plugin.json 无效：") + ex.what();
  }
  if (m.name.empty()) m.name = m.id;
  if (m.invalid.empty()) {
    if (m.api != kPluginApiMajor) {
      m.incompatible = "插件需要协议主版本 " + std::to_string(m.api) + "，宿主协议版本为 " + kPluginApiText;
    } else {
      std::vector<std::string> missing;
      const auto& host = HostPluginFeatures();
      for (const auto& f : m.features) {
        if (std::find(host.begin(), host.end(), f) == host.end()) missing.push_back(f);
      }
      if (!missing.empty()) {
        m.incompatible = "宿主不支持插件需要的特性：" + Join(missing, "、") + "（宿主协议版本 " + kPluginApiText + "）";
      }
    }
  }
  return m;
}

json NormalizePluginSettings(const json& schema, const json& values) {
  json out = json::object();
  for (const json& s : schema) {
    const std::string key = s.value("key", ""), type = s.value("type", "");
    const json fallback = DefaultOf(s);
    const json v = values.is_object() && values.contains(key) ? values[key] : fallback;
    if (type == "boolean") {
      out[key] = v.is_boolean() ? v : fallback;
    } else if (type == "number") {
      if (!v.is_number()) {
        out[key] = fallback;
        continue;
      }
      double x = v.get<double>();
      if (s.contains("min")) x = std::max(x, s["min"].get<double>());
      if (s.contains("max")) x = std::min(x, s["max"].get<double>());
      // 整数值按整数保存，插件读到的也是整数
      if (x == std::floor(x) && std::abs(x) < 9007199254740992.0) out[key] = static_cast<int64_t>(x);
      else out[key] = x;
    } else if (type == "string") {
      out[key] = v.is_string() ? v : fallback;
    } else {
      bool ok = false;
      for (const json& o : s["options"]) ok = ok || (o.contains("value") && o["value"] == v);
      out[key] = ok ? v : fallback;
    }
  }
  return out;
}

struct PluginHost::Plugin {
  PluginManifest m;
  std::string source;               // bundled | user | dev
  std::vector<std::string> active;  // 实际启用的特性
  bool control = false;

  // 用户选择
  bool enabled = false;
  json settings = json::object();

  // 运行状态：starting | running | stopping | stopped | exited | backoff | failed | incompatible | invalid
  std::string state = "stopped";
  std::string error;
  HANDLE process = nullptr, job = nullptr;
  HANDLE stdoutRead = nullptr, stderrRead = nullptr;
  DWORD pid = 0;
  std::thread outReader, errReader, writer;
  std::deque<std::string> outQueue;  // 待写入标准输入的行
  std::condition_variable writerCv;
  bool closeInput = false;  // 写完队列后关闭标准输入
  bool restartAfterStop = false;
  bool stopIsFailure = false;  // 宿主因故障而停止（启动超时、控制通道积压），退出按失败处理
  std::string failureReason;
  std::string failedText;  // 插件以 failed 消息报告的原因
  bool failedRetry = false;
  std::string lastStderr;
  uint64_t stopDeadline = 0, nextStart = 0, startedAt = 0, readyAt = 0;
  int64_t startedUnix = 0;
  std::deque<uint64_t> exits;
  int restarts = 0;
  uint64_t lastNotify = 0;

  struct Pending {
    std::string method;
    uint64_t deadline = 0;
    Callback callback;
  };
  int64_t nextRequestId = 0;
  std::map<int64_t, Pending> pending;

  // 插件报告
  std::string statusText;
  json health;                // null 或 {state, text}
  std::vector<json> metrics;  // {key, label, value, unit}，按首次出现排序
  std::deque<json> logs;      // {seq, t, level, text, fields?}

  // 日志限流与文件
  double logTokens = kLogBurst;
  uint64_t logRefill = 0, droppedLogs = 0, lastDropReport = 0;
  std::ofstream logFile;
  uint64_t logFileBytes = 0;
  bool logDirty = false;

  // 资源（整个进程树）
  uint64_t created = 0;                // 主进程创建时间（FILETIME），用于查找后代
  uint64_t lastSample = 0, lastSampleTime = 0;  // GetTickCount64 与 FILETIME
  std::map<DWORD, uint64_t> cpuByPid;  // 上次采样时各进程的 CPU 时间（100 ns）
  json resources;                      // null 或 {cpu, memoryMB, processes}

  bool Compatible() const { return m.invalid.empty() && m.incompatible.empty(); }
  bool Alive() const { return process != nullptr; }
};

PluginHost::PluginHost(PluginHostOptions options) : options_(std::move(options)) {
  std::error_code ec;
  if (!options_.userDir.empty()) std::filesystem::create_directories(options_.userDir, ec);
}

PluginHost::~PluginHost() {
  {
    std::lock_guard lock(mutex_);
    quit_ = true;
    for (auto& [id, p] : plugins_) RequestStop(*p);
  }
  wake_.notify_all();
  if (supervisor_.joinable()) supervisor_.join();
}

void PluginHost::Start(const json& saved) {
  {
    std::lock_guard lock(mutex_);
    saved_ = saved.is_object() ? saved : json::object();
    Scan();
  }
  supervisor_ = std::thread(&PluginHost::Supervise, this);
}

void PluginHost::Rescan() {
  {
    std::lock_guard lock(mutex_);
    Scan();
  }
  wake_.notify_all();
}

void PluginHost::Scan() {
  std::error_code ec;
  for (const auto& root : options_.searchDirs) {
    if (!std::filesystem::is_directory(root, ec)) continue;
    const char* source =
        !options_.userDir.empty() && std::filesystem::equivalent(root, options_.userDir, ec) ? "user" : "bundled";
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
      if (entry.is_directory() && std::filesystem::exists(entry.path() / "plugin.json")) {
        AddManifest(ReadPluginManifest(entry.path()), source);
      }
    }
  }
  for (const auto& dir : options_.pluginDirs) AddManifest(ReadPluginManifest(dir), "dev");
  ++version_;
}

void PluginHost::AddManifest(PluginManifest m, const char* source) {
  auto it = plugins_.find(m.id);
  if (it != plugins_.end()) {
    // 运行中的插件保持不变，停止后下次扫描再更新清单
    if (it->second->Alive()) return;
    it->second->m = std::move(m);
    it->second->source = source;
  } else {
    auto p = std::make_unique<Plugin>();
    p->m = std::move(m);
    p->source = source;
    const json s = saved_.value(p->m.id, json::object());
    p->enabled = s.value("enabled", false);
    p->settings = s.value("settings", json::object());
    it = plugins_.emplace(p->m.id, std::move(p)).first;
  }
  Plugin& p = *it->second;
  p.settings = NormalizePluginSettings(p.m.settings, p.settings);
  p.active = p.m.ActiveFeatures();
  p.control = std::find(p.active.begin(), p.active.end(), "control") != p.active.end();
  if (!p.m.invalid.empty()) {
    p.state = "invalid";
    p.error = p.m.invalid;
  } else if (!p.m.incompatible.empty()) {
    p.state = "incompatible";
    p.error = p.m.incompatible;
  } else if (p.state == "invalid" || p.state == "incompatible") {
    p.state = "stopped";
    p.error.clear();
  }
}

json PluginHost::Saved() const {
  std::lock_guard lock(mutex_);
  json out = saved_;  // 保留当前未找到的插件的设置
  for (const auto& [id, p] : plugins_) out[id] = {{"enabled", p->enabled}, {"settings", p->settings}};
  return out;
}

json PluginHost::PluginJson(const Plugin& p, size_t maxLogs) const {
  json logs = json::array();
  const size_t skip = p.logs.size() > maxLogs ? p.logs.size() - maxLogs : 0;
  for (size_t i = skip; i < p.logs.size(); ++i) logs.push_back(p.logs[i]);
  return {{"id", p.m.id},
          {"name", p.m.name},
          {"version", p.m.version},
          {"description", p.m.description},
          {"source", p.source},
          {"dir", ToUtf8(p.m.dir.wstring())},
          {"api", p.m.api},
          {"compatible", p.Compatible()},
          {"command", p.m.command},
          {"features", p.m.features},
          {"optionalFeatures", p.m.optionalFeatures},
          {"activeFeatures", p.active},
          {"control", p.control},
          {"restartPolicy", p.m.restart},
          {"actions", p.m.actions},
          {"enabled", p.enabled},
          {"state", p.state},
          {"error", p.error},
          {"pid", p.pid},
          {"restarts", p.restarts},
          {"startedAt", p.Alive() ? p.startedUnix : 0},
          {"readyMs", p.Alive() && p.readyAt ? json(p.readyAt - p.startedAt) : json()},
          {"status", p.statusText},
          {"health", p.health},
          {"metrics", p.metrics},
          {"resources", p.resources},
          {"settingsSchema", p.m.settings},
          {"settings", p.settings},
          {"logs", logs}};
}

json PluginHost::Snapshot() const {
  std::lock_guard lock(mutex_);
  json items = json::array();
  for (const auto& [id, p] : plugins_) items.push_back(PluginJson(*p, kSnapshotLogLines));
  return {{"type", "plugins"},
          {"items", items},
          {"api", kPluginApiText},
          {"features", HostPluginFeatures()},
          {"hostVersion", options_.hostVersion},
          {"userDir", ToUtf8(options_.userDir.wstring())},
          {"resultsMapping", ToUtf8(MappingName())}};
}

bool PluginHost::AnyEnabled() const {
  std::lock_guard lock(mutex_);
  return std::any_of(plugins_.begin(), plugins_.end(),
                     [](const auto& kv) { return kv.second->enabled && kv.second->Compatible(); });
}

std::vector<PluginHost::Notification> PluginHost::TakeNotifications() {
  std::lock_guard lock(mutex_);
  return std::exchange(notifications_, {});
}

PluginHost::Plugin& PluginHost::Find(const std::string& id) const {
  auto it = plugins_.find(id);
  if (it == plugins_.end()) throw std::runtime_error("没有插件 " + id);
  return *it->second;
}

void PluginHost::SetEnabled(const std::string& id, bool enabled) {
  {
    std::lock_guard lock(mutex_);
    Plugin& p = Find(id);
    if (enabled && !p.Compatible()) throw std::runtime_error(p.error);
    p.enabled = enabled;
    const bool idle = p.state == "failed" || p.state == "backoff" || p.state == "exited";
    if (enabled) {
      // 重新启用：清除失败记录，立即启动
      p.exits.clear();
      p.error.clear();
      if (idle) p.state = "stopped";
      p.nextStart = 0;
    } else {
      p.restartAfterStop = false;
      if (idle) p.state = "stopped";
      RequestStop(p);
    }
    ++version_;
  }
  wake_.notify_all();
}

void PluginHost::SetSettings(const std::string& id, const json& values) {
  {
    std::lock_guard lock(mutex_);
    Plugin& p = Find(id);
    const json old = p.settings;
    p.settings = NormalizePluginSettings(p.m.settings, values);
    std::vector<std::string> changed;
    bool allLive = true;
    for (const auto& [key, value] : p.settings.items()) {
      if (old.contains(key) && old[key] == value) continue;
      changed.push_back(key);
      const json* s = FindSetting(p.m.settings, key);
      allLive = allLive && s && s->value("live", false);
    }
    if (!changed.empty()) {
      if (p.Alive() && p.control && allLive && (p.state == "starting" || p.state == "running")) {
        // 只改动了可热更新的设置项：经控制通道下发，不重启（7.4）
        Send(p, {{"type", "settings"}, {"values", p.settings}});
        AddLog(p, "info", "设置已热更新：" + Join(changed, "、"));
      } else if (p.Alive()) {
        // 设置通过环境变量传入，重启后生效
        p.restartAfterStop = true;
        p.stopIsFailure = false;
        RequestStop(p);
      } else if (p.enabled && p.state == "failed") {
        // 插件因设置等原因报告无法继续：修改设置后重试
        p.exits.clear();
        p.error.clear();
        p.state = "stopped";
        p.nextStart = 0;
      }
    }
    ++version_;
  }
  wake_.notify_all();
}

void PluginHost::Restart(const std::string& id) {
  {
    std::lock_guard lock(mutex_);
    Plugin& p = Find(id);
    if (!p.Compatible()) throw std::runtime_error(p.error);
    if (p.Alive()) {
      p.restartAfterStop = true;
      p.stopIsFailure = false;
      RequestStop(p);
    } else if (p.enabled) {
      p.exits.clear();
      p.error.clear();
      p.state = "stopped";
      p.nextStart = 0;
    }
    ++version_;
  }
  wake_.notify_all();
}

void PluginHost::InvokeAction(const std::string& id, const std::string& action) {
  std::lock_guard lock(mutex_);
  Plugin& p = Find(id);
  const auto it = std::find_if(p.m.actions.begin(), p.m.actions.end(),
                               [&](const json& a) { return a.value("id", "") == action; });
  if (it == p.m.actions.end()) throw std::runtime_error("插件 " + p.m.name + " 没有操作 " + action);
  if (!p.control) throw std::runtime_error("插件未启用控制通道，不能执行操作");
  if (p.state != "running") throw std::runtime_error("插件未在运行");
  const std::string label = it->value("label", action);
  Plugin* pp = &p;  // 插件对象在宿主存续期间不会移除
  SendRequest(p, "action", {{"id", action}}, kActionTimeoutMs, [this, pp, label](const json& result, const json& error) {
    if (error.is_null()) {
      const std::string text = result.is_object() && result.contains("text") && result["text"].is_string()
                                   ? result["text"].get<std::string>()
                                   : label + "：已完成";
      notifications_.push_back({pp->m.name, "info", text});
      AddLog(*pp, "info", "操作“" + label + "”：" + text);
    } else {
      const std::string text = label + " 失败：" + Str(error, "message", Str(error, "code", "未知错误"));
      notifications_.push_back({pp->m.name, "error", text});
      AddLog(*pp, "warning", "操作" + text);
    }
  });
}

std::wstring PluginHost::MappingName() const {
  return options_.resultsMapping.empty() ? std::wstring(kResultsMappingName) : options_.resultsMapping;
}

std::filesystem::path PluginHost::PluginDir(const std::string& id) const {
  std::lock_guard lock(mutex_);
  return Find(id).m.dir;
}

void PluginHost::SetSession(const json& state) {
  std::lock_guard lock(mutex_);
  if (state == session_) return;
  session_ = state;
  for (auto& [id, p] : plugins_) {
    if (p->control && p->Alive()) Send(*p, {{"type", "event"}, {"name", "session"}, {"data", session_}});
  }
}

void PluginHost::CollectDiagnostics(std::function<void(json)> done) {
  struct Collection {
    json items = json::array();
    size_t outstanding = 0;
    std::function<void(json)> done;
  };
  auto c = std::make_shared<Collection>();
  c->done = std::move(done);
  {
    std::lock_guard lock(mutex_);
    std::vector<std::pair<Plugin*, size_t>> ask;
    for (auto& [id, p] : plugins_) {
      json item = PluginJson(*p, kMaxLogLines);
      item["startTimeout"] = p->m.startTimeoutMs;
      item["stopTimeout"] = p->m.stopTimeoutMs;
      if (p->control && p->state == "running") ask.emplace_back(p.get(), c->items.size());
      c->items.push_back(std::move(item));
    }
    c->outstanding = ask.size();
    if (ask.empty()) deferred_.push_back([c] { c->done(std::move(c->items)); });
    for (auto [p, index] : ask) {
      SendRequest(*p, "diagnose", json::object(), kDiagnoseTimeoutMs,
                  [this, c, index](const json& result, const json& error) {
                    c->items[index]["diagnose"] = error.is_null() ? result : json{{"error", error}};
                    if (--c->outstanding == 0) deferred_.push_back([c] { c->done(std::move(c->items)); });
                  });
    }
  }
  wake_.notify_all();
}

// ---------- 日志 ----------

void PluginHost::AddLog(Plugin& p, const std::string& level, const std::string& text, const json& fields) {
  json e = {{"seq", ++logSeq_}, {"t", UnixMs()}, {"level", level}, {"text", text}};
  if (fields.is_object() && !fields.empty()) e["fields"] = fields;
  WriteLogFile(p, e);
  p.logs.push_back(std::move(e));
  while (p.logs.size() > kMaxLogLines) p.logs.pop_front();
  ++version_;
}

// 来自插件的日志按令牌桶限流（7.6）；宿主自己的记录不限流
void PluginHost::PluginLog(Plugin& p, const std::string& level, const std::string& text, const json& fields) {
  const uint64_t now = NowMs();
  p.logTokens = std::min(kLogBurst, p.logTokens + static_cast<double>(now - p.logRefill) * kLogRate / 1000.0);
  p.logRefill = now;
  if (p.logTokens < 1) {
    ++p.droppedLogs;
    return;
  }
  p.logTokens -= 1;
  AddLog(p, level, text, fields);
}

void PluginHost::WriteLogFile(Plugin& p, const json& e) {
  if (options_.dataDir.empty()) return;
  const auto path = options_.dataDir / "logs" / "plugins" / FromUtf8(p.m.id + ".log");
  std::error_code ec;
  if (!p.logFile.is_open()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    p.logFileBytes = std::filesystem::exists(path, ec) ? std::filesystem::file_size(path, ec) : 0;
    p.logFile.open(path, std::ios::app | std::ios::binary);
    if (!p.logFile) return;
  }
  std::string line = LocalTimeText(e["t"].get<int64_t>()) + " " + e["level"].get<std::string>() + " " +
                     e["text"].get<std::string>();
  if (e.contains("fields")) line += " " + Dump(e["fields"]);
  line += "\n";
  p.logFile << line;
  p.logFileBytes += line.size();
  p.logDirty = true;
  if (p.logFileBytes > kLogFileBytes) {
    p.logFile.close();
    auto old = path;
    old += L".1";
    std::filesystem::remove(old, ec);
    std::filesystem::rename(path, old, ec);
    p.logFileBytes = 0;
    p.logDirty = false;
  }
}

// ---------- 插件输出（5.5、7.3、7.4） ----------

void PluginHost::OnLine(Plugin& p, std::string line, bool fromStderr) {
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
  if (line.empty()) return;
  line = SanitizeUtf8(line);
  std::lock_guard lock(mutex_);
  if (fromStderr) {
    p.lastStderr = line.size() > 500 ? line.substr(0, 500) + "…" : line;
    PluginLog(p, "error", line);
    return;
  }
  json m;
  try {
    m = json::parse(line);
  } catch (const std::exception&) {
    PluginLog(p, "info", line);
    return;
  }
  if (!m.is_object() || !m.contains("type") || !m["type"].is_string()) {
    PluginLog(p, "info", line);
    return;
  }
  // 字段类型不符等问题不能让读取线程退出（未捕获的异常会结束宿主进程）
  try {
    OnMessage(p, m, line);
  } catch (const std::exception& e) {
    PluginLog(p, "warning", std::string("无法处理的消息（") + e.what() + "）：" + line);
  }
}

void PluginHost::OnMessage(Plugin& p, const json& m, const std::string& line) {
  const std::string type = m["type"];
  const auto text = [&] { return m.contains("text") && m["text"].is_string() ? m["text"].get<std::string>() : ""; };
  if (type == "status") {
    p.statusText = text();
    ++version_;
  } else if (type == "metric" && m.contains("key") && m["key"].is_string()) {
    json metric = {{"key", m["key"]},
                   {"label", Str(m, "label", m["key"].get<std::string>())},
                   {"value", m.value("value", json())},
                   {"unit", Str(m, "unit")}};
    auto it = std::find_if(p.metrics.begin(), p.metrics.end(), [&](const json& x) { return x["key"] == m["key"]; });
    if (it != p.metrics.end()) {
      // 只更新值；标签与单位以首次报告为准，除非再次显式给出
      (*it)["value"] = metric["value"];
      if (m.contains("label")) (*it)["label"] = metric["label"];
      if (m.contains("unit")) (*it)["unit"] = metric["unit"];
    } else {
      p.metrics.push_back(metric);
    }
    ++version_;
  } else if (type == "log") {
    const std::string level = Str(m, "level", "info");
    PluginLog(p, level == "warning" || level == "error" ? level : "info", text(),
              m.contains("fields") ? m["fields"] : json());
  } else if (type == "notify") {
    const uint64_t now = NowMs();
    PluginLog(p, "info", "提示：" + text());
    if (now - p.lastNotify >= kNotifyIntervalMs) {
      p.lastNotify = now;
      notifications_.push_back({p.m.name, "info", text()});
    }
  } else if (type == "ready") {
    if (p.state == "starting") {
      p.state = "running";
      p.readyAt = NowMs();
      AddLog(p, "info", "已就绪，启动用时 " + std::to_string(p.readyAt - p.startedAt) + " ms");
    }
  } else if (type == "failed") {
    p.failedText = text().empty() ? "插件报告无法继续运行" : text();
    p.failedRetry = Bool(m, "retry", false);
    AddLog(p, "error", "插件报告无法继续：" + p.failedText);
  } else if (type == "health") {
    const std::string state = Str(m, "state");
    if (state == "ok" || state == "degraded" || state == "error") {
      const bool changed = p.health.is_null() || p.health["state"] != state;
      p.health = {{"state", state}, {"text", text()}};
      if (changed && state != "ok") AddLog(p, state == "error" ? "error" : "warning", "健康状态：" + text());
      ++version_;
    }
  } else if (type == "request") {
    OnPluginRequest(p, m);
  } else if (type == "response" && m.contains("id") && m["id"].is_number_integer()) {
    auto it = p.pending.find(m["id"].get<int64_t>());
    if (it == p.pending.end()) return;  // 已超时或已取消
    Callback cb = std::move(it->second.callback);
    p.pending.erase(it);
    json error;
    if (m.contains("error")) {
      error = m["error"].is_object() ? m["error"] : Error("failed", m["error"].is_string() ? m["error"].get<std::string>() : "插件返回错误");
    }
    cb(m.value("result", json()), error);
  } else {
    PluginLog(p, "info", line);
  }
}

// 插件 → 宿主的请求：在读取线程中立即回答，不做阻塞操作（7.4）
void PluginHost::OnPluginRequest(Plugin& p, const json& m) {
  if (!p.control) {
    AddLog(p, "warning", "插件未启用 control 特性，宿主不回答请求");
    return;
  }
  if (!m.contains("id")) return;
  const std::string method = Str(m, "method");
  json response = {{"type", "response"}, {"id", m["id"]}};
  if (method == "host.info") {
    response["result"] = {{"version", options_.hostVersion}, {"api", kPluginApiText}, {"features", p.active}};
  } else if (method == "session.state") {
    response["result"] = session_;
  } else {
    response["error"] = Error("unknown_method", "宿主没有方法 " + method);
  }
  Send(p, response);
}

// ---------- 控制通道（7.4） ----------

void PluginHost::Send(Plugin& p, const json& message) {
  if (!p.Alive() || p.closeInput) return;
  if (p.outQueue.size() >= kMaxQueuedMessages) {
    StopAsFailure(p, "控制通道积压：插件长时间未读取标准输入");
    return;
  }
  p.outQueue.push_back(Dump(message) + "\n");
  p.writerCv.notify_one();
}

void PluginHost::SendRequest(Plugin& p, const std::string& method, const json& params, uint64_t timeoutMs,
                             Callback callback) {
  if (!p.control || !p.Alive() || p.state != "running") {
    callback(json(), Error("failed", "插件未在运行"));
    return;
  }
  if (p.pending.size() >= kMaxPendingRequests) {
    callback(json(), Error("busy", "插件有太多未完成的请求"));
    return;
  }
  const int64_t id = ++p.nextRequestId;
  p.pending[id] = {method, NowMs() + timeoutMs, std::move(callback)};
  Send(p, {{"type", "request"}, {"id", id}, {"method", method}, {"params", params}});
}

void PluginHost::FailPending(Plugin& p, const std::string& message) {
  auto pending = std::exchange(p.pending, {});
  for (auto& [id, r] : pending) r.callback(json(), Error("failed", message));
}

void PluginHost::WriterLoop(Plugin& p, HANDLE pipe) {
  std::unique_lock lock(mutex_);
  for (;;) {
    p.writerCv.wait(lock, [&] { return !p.outQueue.empty() || p.closeInput; });
    if (p.outQueue.empty()) break;  // 已请求关闭且队列写完
    const std::string line = std::move(p.outQueue.front());
    p.outQueue.pop_front();
    lock.unlock();
    // 插件不读取标准输入时阻塞在这里，只影响本线程
    size_t off = 0;
    DWORD written = 0;
    bool ok = true;
    while (ok && off < line.size()) {
      ok = WriteFile(pipe, line.data() + off, static_cast<DWORD>(line.size() - off), &written, nullptr) && written > 0;
      off += written;
    }
    lock.lock();
    if (!ok) {
      p.outQueue.clear();  // 插件已关闭标准输入或已退出
      break;
    }
  }
  lock.unlock();
  CloseHandle(pipe);
}

// ---------- 进程托管（5.5、7.3） ----------

void PluginHost::Launch(Plugin& p) {
  p.error.clear();
  p.statusText.clear();
  p.metrics.clear();
  p.health = nullptr;
  p.resources = nullptr;
  p.cpuByPid.clear();
  p.lastSampleTime = 0;
  p.failedText.clear();
  p.failedRetry = false;
  p.stopIsFailure = false;
  p.failureReason.clear();
  p.lastStderr.clear();
  p.closeInput = false;
  p.outQueue.clear();
  p.logTokens = kLogBurst;
  p.logRefill = NowMs();

  // 入口：含路径或在插件目录中存在时相对插件目录，否则在 PATH 中查找
  std::filesystem::path app = FromUtf8(p.m.command);
  if (app.has_parent_path() || std::filesystem::exists(p.m.dir / app)) {
    app = p.m.dir / app;
    if (!std::filesystem::exists(app)) throw std::runtime_error("入口程序不存在：" + ToUtf8(app.wstring()));
  } else {
    wchar_t found[MAX_PATH];
    if (!SearchPathW(nullptr, app.c_str(), L".exe", MAX_PATH, found, nullptr)) {
      std::string hint;
      const std::string c = p.m.command;
      if (c == "python" || c == "python3" || c == "py" || c == "pythonw") hint = "，此插件需要 Python 3，请安装并加入 PATH";
      throw std::runtime_error("在 PATH 中找不到 " + c + hint);
    }
    app = found;
  }
  std::wstring cmdline = Quote(app.wstring());
  for (const std::string& a : p.m.args) cmdline += L" " + Quote(FromUtf8(a));

  const auto data = options_.dataDir / "plugin-data" / FromUtf8(p.m.id);
  std::filesystem::create_directories(data);
  wchar_t existing[32767];
  const DWORD n = GetEnvironmentVariableW(L"PYTHONPATH", existing, 32767);
  const std::wstring pythonPath = options_.sdkDir.wstring() + (n ? L";" + std::wstring(existing, n) : L"");
  std::wstring env = BuildEnvironment({{L"NR_PLUGIN_ID", FromUtf8(p.m.id)},
                                       {L"NR_PLUGIN_DIR", p.m.dir.wstring()},
                                       {L"NR_PLUGIN_DATA", data.wstring()},
                                       {L"NR_PLUGIN_SETTINGS", FromUtf8(Dump(p.settings))},
                                       {L"NR_RESULTS_MAPPING", MappingName()},
                                       {L"NR_API", FromUtf8(kPluginApiText)},
                                       {L"NR_FEATURES", FromUtf8(Join(p.active, ","))},
                                       {L"NR_HOST_VERSION", FromUtf8(options_.hostVersion)},
                                       {L"PYTHONPATH", pythonPath},
                                       {L"PYTHONUNBUFFERED", L"1"},
                                       {L"PYTHONIOENCODING", L"utf-8"}});

  // 三根管道；宿主一端不可继承，子进程只继承显式列出的句柄
  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  HANDLE inRead = nullptr, inWrite = nullptr, outRead = nullptr, outWrite = nullptr, errRead = nullptr,
         errWrite = nullptr;
  const auto closeAll = [&] {
    for (HANDLE h : {inRead, inWrite, outRead, outWrite, errRead, errWrite}) {
      if (h) CloseHandle(h);
    }
  };
  if (!CreatePipe(&inRead, &inWrite, &sa, 64 * 1024) || !CreatePipe(&outRead, &outWrite, &sa, 0) ||
      !CreatePipe(&errRead, &errWrite, &sa, 0)) {
    closeAll();
    throw std::runtime_error("创建管道失败");
  }
  SetHandleInformation(inWrite, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(outRead, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(errRead, HANDLE_FLAG_INHERIT, 0);

  // 每个插件一个作业对象：结束时连同它派生的进程一起结束，宿主退出（包括崩溃）时一并结束；
  // 作业对象同时给出整个进程树的 CPU 时间（7.3、7.6）
  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  if (job) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
  }

  HANDLE inherit[] = {inRead, outWrite, errWrite};
  SIZE_T attrSize = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attrSize);
  std::vector<char> attrBuf(attrSize);
  auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attrBuf.data());
  InitializeProcThreadAttributeList(attrs, 1, 0, &attrSize);
  UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit, sizeof(inherit), nullptr, nullptr);

  STARTUPINFOEXW si{};
  si.StartupInfo.cb = sizeof(si);
  si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  si.StartupInfo.hStdInput = inRead;
  si.StartupInfo.hStdOutput = outWrite;
  si.StartupInfo.hStdError = errWrite;
  si.lpAttributeList = attrs;
  PROCESS_INFORMATION pi{};
  const std::wstring dir = p.m.dir.wstring();
  const BOOL ok = CreateProcessW(app.c_str(), cmdline.data(), nullptr, nullptr, TRUE,
                                 CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT |
                                     EXTENDED_STARTUPINFO_PRESENT,
                                 env.data(), dir.c_str(), &si.StartupInfo, &pi);
  const DWORD err = GetLastError();
  DeleteProcThreadAttributeList(attrs);
  if (!ok) {
    closeAll();
    if (job) CloseHandle(job);
    throw std::runtime_error("启动失败（错误 " + std::to_string(err) + "）：" + ToUtf8(cmdline));
  }
  if (job && !AssignProcessToJobObject(job, pi.hProcess)) {
    CloseHandle(job);
    job = nullptr;
    AddLog(p, "warning", "无法放入作业对象（错误 " + std::to_string(GetLastError()) + "），停止时只能结束主进程");
  }
  ResumeThread(pi.hThread);
  CloseHandle(pi.hThread);
  CloseHandle(inRead);
  CloseHandle(outWrite);
  CloseHandle(errWrite);

  p.process = pi.hProcess;
  p.job = job;
  p.pid = pi.dwProcessId;
  p.created = CreationTime(pi.hProcess);
  p.stdoutRead = outRead;
  p.stderrRead = errRead;
  p.state = p.control ? "starting" : "running";
  p.startedAt = NowMs();
  p.startedUnix = UnixMs();
  p.readyAt = p.control ? 0 : p.startedAt;
  p.stopDeadline = 0;
  p.lastSample = 0;
  const auto reader = [this, &p](HANDLE pipe, bool fromStderr) {
    std::string pending;
    char buf[4096];
    DWORD got = 0;
    while (ReadFile(pipe, buf, sizeof(buf), &got, nullptr) && got > 0) {
      pending.append(buf, got);
      size_t start = 0;
      for (size_t pos; (pos = pending.find('\n', start)) != std::string::npos; start = pos + 1) {
        OnLine(p, pending.substr(start, pos - start), fromStderr);
      }
      pending.erase(0, start);
      if (pending.size() > kMaxLineBytes) {  // 没有换行的超长输出按块处理
        OnLine(p, pending, fromStderr);
        pending.clear();
      }
    }
    if (!pending.empty()) OnLine(p, pending, fromStderr);
  };
  p.outReader = std::thread(reader, outRead, false);
  p.errReader = std::thread(reader, errRead, true);
  p.writer = std::thread(&PluginHost::WriterLoop, this, std::ref(p), inWrite);
  AddLog(p, "info", "已启动，进程号 " + std::to_string(p.pid) + (p.control ? "，等待就绪" : ""));
  if (p.control) Send(p, {{"type", "event"}, {"name", "session"}, {"data", session_}});
}

// 请求停止：启用控制通道时先发 stop 消息，然后关闭标准输入；宽限期后结束进程树（由监督线程执行）
void PluginHost::RequestStop(Plugin& p) {
  if (!p.Alive() || p.state == "stopping") return;
  p.state = "stopping";
  p.stopDeadline = NowMs() + p.m.stopTimeoutMs;
  if (p.control && !p.closeInput && p.outQueue.size() < kMaxQueuedMessages) {
    p.outQueue.push_back(Dump({{"type", "stop"}}) + "\n");
  }
  p.closeInput = true;
  p.writerCv.notify_one();
  ++version_;
}

void PluginHost::StopAsFailure(Plugin& p, const std::string& reason) {
  if (!p.Alive()) return;
  if (!p.stopIsFailure) AddLog(p, "error", reason);
  p.stopIsFailure = true;
  p.failureReason = reason;
  p.restartAfterStop = false;
  RequestStop(p);
}

void PluginHost::Collect(Plugin& p) {
  // 插件的子进程可能继承了管道，读写线程迟迟等不到结束；取消阻塞的读写
  for (std::thread* t : {&p.outReader, &p.errReader, &p.writer}) {
    if (!t->joinable()) continue;
    for (int i = 0; i < 20 && WaitForSingleObject(t->native_handle(), 0) == WAIT_TIMEOUT; ++i) Sleep(10);
    if (WaitForSingleObject(t->native_handle(), 0) == WAIT_TIMEOUT) CancelSynchronousIo(t->native_handle());
    t->join();
  }
  for (HANDLE* h : {&p.stdoutRead, &p.stderrRead}) {
    if (*h) CloseHandle(*h);
    *h = nullptr;
  }
}

void PluginHost::OnExit(Plugin& p, uint64_t now) {
  DWORD code = 0;
  GetExitCodeProcess(p.process, &code);
  CloseHandle(p.process);
  p.process = nullptr;
  p.pid = 0;
  if (p.job) {
    CloseHandle(p.job);
    p.job = nullptr;
  }
  p.resources = nullptr;
  FailPending(p, "插件已退出");
  ++version_;

  if (p.state == "stopping" && !p.stopIsFailure) {
    AddLog(p, "info", "已停止，退出代码 " + std::to_string(code));
    p.state = "stopped";
    if (p.restartAfterStop) {
      p.restartAfterStop = false;
      p.exits.clear();
    }
    return;
  }

  // 未经请求的退出，或宿主因故障而停止
  std::string reason;
  if (p.stopIsFailure) {
    reason = p.failureReason;
  } else if (!p.failedText.empty()) {
    reason = p.failedText;
  } else {
    reason = "进程退出（代码 " + std::to_string(code) + "）";
    if (!p.lastStderr.empty()) reason += "：" + p.lastStderr;
  }
  if (!p.stopIsFailure) AddLog(p, code == 0 ? "info" : "error", "进程退出，代码 " + std::to_string(code));

  if (!p.stopIsFailure && !p.failedText.empty() && !p.failedRetry) {
    // 插件报告无法恢复的错误（设置错误、缺少依赖）：不重启，直到用户修改设置或手动重启
    p.state = "failed";
    p.error = p.failedText + "（修改设置或点“重启”后重试）";
    p.exits.clear();
    return;
  }
  if (!p.stopIsFailure && code == 0 && p.m.restart != "always") {
    p.state = "exited";
    p.error.clear();
    AddLog(p, "info", "插件已自行结束");
    return;
  }
  if (p.m.restart == "never") {
    p.state = "failed";
    p.error = reason;
    return;
  }
  // 插件应持续运行；按退避重启，短时间内反复退出则停止
  p.exits.push_back(now);
  while (!p.exits.empty() && now - p.exits.front() > kCrashWindowMs) p.exits.pop_front();
  if (p.exits.size() >= kCrashLimit) {
    p.state = "failed";
    p.error = "60 秒内退出 " + std::to_string(kCrashLimit) + " 次，已停止；重新启用以重试。最后一次：" + reason;
  } else {
    const uint64_t backoff = std::min<uint64_t>(kMaxBackoffMs, 1000ull << (p.exits.size() - 1));
    p.state = "backoff";
    p.nextStart = now + backoff;
    p.error = reason + "，" + std::to_string(backoff / 1000) + " 秒后重启";
  }
}

// 资源：整个进程树的 CPU 占用（占全部逻辑处理器的百分比）与私有内存
void PluginHost::SampleResources(Plugin& p, uint64_t now, const std::vector<ProcessEntry>& all) {
  const uint64_t elapsed = now - p.lastSample;
  p.lastSample = now;
  FILETIME nowFt{};
  GetSystemTimeAsFileTime(&nowFt);
  auto processes = OpenDescendants(all, p.pid, p.created, 0);
  HANDLE self = nullptr;
  DuplicateHandle(GetCurrentProcess(), p.process, GetCurrentProcess(), &self, 0, FALSE, DUPLICATE_SAME_ACCESS);
  if (self) processes.emplace_back(p.pid, self);

  uint64_t cpuDelta = 0, privateBytes = 0;
  std::map<DWORD, uint64_t> cpuByPid;
  for (const auto& [pid, h] : processes) {
    FILETIME created{}, exited{}, kernel{}, user{};
    if (GetProcessTimes(h, &created, &exited, &kernel, &user)) {
      const uint64_t cpu = FileTimeValue(kernel) + FileTimeValue(user);
      const auto it = p.cpuByPid.find(pid);
      // 上次采样之后才创建的进程，全部 CPU 时间都属于本次采样间隔
      if (it != p.cpuByPid.end()) cpuDelta += cpu > it->second ? cpu - it->second : 0;
      else if (p.lastSampleTime && FileTimeValue(created) >= p.lastSampleTime) cpuDelta += cpu;
      cpuByPid[pid] = cpu;
    }
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(h, reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
      privateBytes += pmc.PrivateUsage;
    }
    CloseHandle(h);
  }
  static const DWORD processors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  json cpuPercent;
  if (p.lastSampleTime && elapsed > 0) {
    const double pct = static_cast<double>(cpuDelta) / (static_cast<double>(elapsed) * 10000.0) /
                       std::max<DWORD>(1, processors) * 100.0;
    cpuPercent = std::round(pct * 10) / 10;
  }
  p.cpuByPid = std::move(cpuByPid);
  p.lastSampleTime = FileTimeValue(nowFt);
  p.resources = {{"cpu", cpuPercent},
                 {"memoryMB", std::round(static_cast<double>(privateBytes) / (1024.0 * 1024.0) * 10) / 10},
                 {"processes", processes.size()}};
  ++version_;
}

void PluginHost::Supervise() {
  std::unique_lock lock(mutex_);
  for (;;) {
    const uint64_t now = NowMs();
    std::vector<Plugin*> exited;
    std::optional<std::vector<ProcessEntry>> processes;  // 本轮需要采样时才取一次进程快照
    bool busy = false;
    for (auto& [id, ptr] : plugins_) {
      Plugin& p = *ptr;
      if (p.Alive()) {
        busy = true;
        if (WaitForSingleObject(p.process, 0) == WAIT_OBJECT_0) {
          exited.push_back(&p);
        } else if (p.state == "stopping" && now >= p.stopDeadline) {
          KillTree(p.pid, p.created, p.job, p.process, 1);
          AddLog(p, "warning", "未在 " + Seconds(p.m.stopTimeoutMs) + " 秒内退出，已结束进程树");
          p.stopDeadline = UINT64_MAX;
        } else if (p.state == "starting" && now - p.startedAt >= p.m.startTimeoutMs) {
          StopAsFailure(p, "启动超时：" + Seconds(p.m.startTimeoutMs) + " 秒内未报告就绪");
        }
        // 宿主请求超时：回答调用方并通知插件不再等待
        std::vector<int64_t> expired;
        for (const auto& [rid, r] : p.pending) {
          if (now >= r.deadline) expired.push_back(rid);
        }
        for (int64_t rid : expired) {
          Callback cb = std::move(p.pending[rid].callback);
          p.pending.erase(rid);
          cb(json(), Error("timeout", "插件未在时限内回答"));
          Send(p, {{"type", "cancel"}, {"id", rid}});
        }
        if (now - p.lastSample >= kResourceIntervalMs) {
          if (!processes) processes = SnapshotProcesses();
          SampleResources(p, now, *processes);
        }
      } else if (!quit_ && p.enabled && p.Compatible() &&
                 (p.state == "stopped" || (p.state == "backoff" && now >= p.nextStart))) {
        try {
          if (p.state == "backoff") ++p.restarts;
          Launch(p);
        } catch (const std::exception& e) {
          p.state = "failed";
          p.error = e.what();
          AddLog(p, "error", e.what());
        }
        busy = true;
      }
      if (p.droppedLogs && now - p.lastDropReport >= 1000) {
        p.lastDropReport = now;
        AddLog(p, "warning", "已丢弃 " + std::to_string(p.droppedLogs) + " 行日志（超过每秒 100 行）");
        p.droppedLogs = 0;
      }
      if (p.logDirty) {
        p.logFile.flush();
        p.logDirty = false;
      }
    }

    // 回收在锁外进行：读写线程处理输出时需要持锁
    if (!exited.empty()) {
      for (Plugin* p : exited) {
        p->closeInput = true;
        p->writerCv.notify_all();
        KillTree(p->pid, p->created, p->job, nullptr, 0);  // 主进程退出后结束它留下的进程
      }
      lock.unlock();
      for (Plugin* p : exited) Collect(*p);
      lock.lock();
      for (Plugin* p : exited) OnExit(*p, NowMs());
      continue;
    }

    if (!deferred_.empty()) {
      auto deferred = std::exchange(deferred_, {});
      lock.unlock();
      for (auto& f : deferred) f();
      lock.lock();
      continue;
    }

    if (quit_ && !busy) return;
    wake_.wait_for(lock, std::chrono::milliseconds(100));
  }
}

}  // namespace nr
