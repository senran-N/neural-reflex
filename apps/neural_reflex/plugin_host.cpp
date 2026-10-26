#include "plugin_host.h"

#include "pipeline/result_publisher.h"
#include "targets.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cwctype>
#include <deque>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <utility>

namespace nr::app {

using json = nlohmann::json;

namespace {

constexpr size_t kMaxLogLines = 200;
constexpr size_t kSnapshotLogLines = 100;
constexpr uint64_t kStopGraceMs = 2000;
constexpr uint64_t kNotifyIntervalMs = 2000;
constexpr uint64_t kCrashWindowMs = 60000;
constexpr size_t kCrashLimit = 5;
constexpr uint64_t kMaxBackoffMs = 30000;

uint64_t NowMs() { return GetTickCount64(); }

int64_t UnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
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

// 设置项声明的校验（扩展机制方案第 6 节）
void CheckSettingsSchema(const json& schema) {
  if (!schema.is_array()) throw std::runtime_error("settings 须为数组");
  std::vector<std::string> keys;
  for (const json& s : schema) {
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

// 按声明补全并校验设置值，不合法的值改用缺省值
json NormalizeSettings(const json& schema, const json& values) {
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

}  // namespace

struct PluginHost::Plugin {
  // 清单
  std::string id, name, version, description, source;
  int api = 0;
  std::filesystem::path dir;
  std::string command;
  std::vector<std::string> args;
  json settingsSchema = json::array();
  std::string invalid;  // 清单错误；非空时不可启用

  // 用户选择
  bool enabled = false;
  json settings = json::object();

  // 运行状态
  std::string state = "stopped";  // stopped | running | stopping | backoff | failed | incompatible | invalid
  std::string error;
  HANDLE process = nullptr;
  HANDLE stdinWrite = nullptr;
  HANDLE stdoutRead = nullptr, stderrRead = nullptr;
  DWORD pid = 0;
  std::thread outReader, errReader;
  bool restartAfterStop = false;
  uint64_t stopDeadline = 0, nextStart = 0, startedAt = 0;
  std::deque<uint64_t> exits;
  int restarts = 0;
  uint64_t lastNotify = 0;

  // 插件报告
  std::string statusText;
  std::vector<json> metrics;  // {key, label, value, unit}，按首次出现排序
  std::deque<json> logs;      // {t, level, text}

  bool Compatible() const { return invalid.empty() && api == kPluginApi; }
};

PluginHost::PluginHost(std::vector<std::filesystem::path> searchDirs, std::filesystem::path dataDir,
                       std::filesystem::path sdkDir)
    : searchDirs_(std::move(searchDirs)), dataDir_(std::move(dataDir)), sdkDir_(std::move(sdkDir)) {
  // 插件进程加入作业对象：宿主退出（包括崩溃）时一并结束
  job_ = CreateJobObjectW(nullptr, nullptr);
  if (job_) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
  }
  std::error_code ec;
  std::filesystem::create_directories(UserDir(), ec);
}

PluginHost::~PluginHost() {
  {
    std::lock_guard lock(mutex_);
    quit_ = true;
    for (auto& [id, p] : plugins_) RequestStop(*p);
  }
  wake_.notify_all();
  if (supervisor_.joinable()) supervisor_.join();
  if (job_) CloseHandle(job_);
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
  static const std::regex kIdPattern("^[a-z0-9][a-z0-9.-]{0,63}$");
  for (size_t d = 0; d < searchDirs_.size(); ++d) {
    std::error_code ec;
    if (!std::filesystem::is_directory(searchDirs_[d], ec)) continue;
    const char* source = d + 1 == searchDirs_.size() ? "user" : "bundled";
    for (const auto& entry : std::filesystem::directory_iterator(searchDirs_[d], ec)) {
      const auto manifestPath = entry.path() / "plugin.json";
      if (!entry.is_directory() || !std::filesystem::exists(manifestPath)) continue;

      auto p = std::make_unique<Plugin>();
      p->dir = entry.path();
      p->source = source;
      p->id = ToUtf8(entry.path().filename().wstring());
      try {
        const json m = json::parse(ReadText(manifestPath));
        if (m.value("schema", 0) != 1) throw std::runtime_error("schema 须为 1");
        p->id = m.value("id", "");
        if (!std::regex_match(p->id, kIdPattern)) throw std::runtime_error("id 只能包含小写字母、数字、- 与 .");
        p->name = m.value("name", p->id);
        p->version = m.value("version", "");
        p->description = m.value("description", "");
        p->api = m.value("api", 0);
        const json& e = m.at("entry");
        p->command = e.at("command").get<std::string>();
        p->args = e.value("args", std::vector<std::string>{});
        p->settingsSchema = m.value("settings", json::array());
        CheckSettingsSchema(p->settingsSchema);
      } catch (const std::exception& ex) {
        p->invalid = std::string("plugin.json 无效：") + ex.what();
      }
      if (p->name.empty()) p->name = p->id;

      auto it = plugins_.find(p->id);
      if (it != plugins_.end()) {
        Plugin& old = *it->second;
        // 运行中的插件保持不变，停止后下次扫描再更新清单
        if (old.process || old.state == "running" || old.state == "stopping") continue;
        p->enabled = old.enabled;
        p->settings = old.settings;
        if (old.state == "failed" || old.state == "backoff") {
          p->state = old.state;
          p->error = old.error;
          p->nextStart = old.nextStart;
        }
        p->exits = old.exits;
        p->restarts = old.restarts;
        p->logs = old.logs;
        it->second = std::move(p);
      } else {
        const json s = saved_.value(p->id, json::object());
        p->enabled = s.value("enabled", false);
        p->settings = s.value("settings", json::object());
        it = plugins_.emplace(p->id, std::move(p)).first;
      }
      Plugin& q = *it->second;
      q.settings = NormalizeSettings(q.settingsSchema, q.settings);
      if (!q.invalid.empty()) {
        q.state = "invalid";
        q.error = q.invalid;
      } else if (q.api != kPluginApi) {
        q.state = "incompatible";
        q.error = "插件需要协议版本 " + std::to_string(q.api) + "，宿主支持版本 " + std::to_string(kPluginApi);
      } else if (q.state == "invalid" || q.state == "incompatible") {
        q.state = "stopped";
        q.error.clear();
      }
    }
  }
  ++version_;
}

json PluginHost::Saved() const {
  std::lock_guard lock(mutex_);
  json out = saved_;  // 保留当前未找到的插件的设置
  for (const auto& [id, p] : plugins_) out[id] = {{"enabled", p->enabled}, {"settings", p->settings}};
  return out;
}

json PluginHost::Snapshot() const {
  std::lock_guard lock(mutex_);
  json items = json::array();
  for (const auto& [id, p] : plugins_) {
    json logs = json::array();
    const size_t skip = p->logs.size() > kSnapshotLogLines ? p->logs.size() - kSnapshotLogLines : 0;
    for (size_t i = skip; i < p->logs.size(); ++i) logs.push_back(p->logs[i]);
    items.push_back({{"id", p->id},
                     {"name", p->name},
                     {"version", p->version},
                     {"description", p->description},
                     {"source", p->source},
                     {"dir", ToUtf8(p->dir.wstring())},
                     {"api", p->api},
                     {"compatible", p->Compatible()},
                     {"command", p->command},
                     {"enabled", p->enabled},
                     {"state", p->state},
                     {"error", p->error},
                     {"pid", p->pid},
                     {"restarts", p->restarts},
                     {"status", p->statusText},
                     {"metrics", p->metrics},
                     {"settingsSchema", p->settingsSchema},
                     {"settings", p->settings},
                     {"logs", logs}});
  }
  return {{"type", "plugins"},
          {"items", items},
          {"api", kPluginApi},
          {"userDir", ToUtf8(UserDir().wstring())},
          {"resultsMapping", ToUtf8(kResultsMappingName)}};
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
    if (enabled) {
      // 重新启用：清除失败记录，立即启动
      p.exits.clear();
      p.error.clear();
      if (p.state == "failed" || p.state == "backoff") p.state = "stopped";
      p.nextStart = 0;
    } else {
      p.restartAfterStop = false;
      if (p.state == "backoff" || p.state == "failed") p.state = "stopped";
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
    p.settings = NormalizeSettings(p.settingsSchema, values);
    // 设置通过环境变量传入，运行中的插件重启后生效
    if (p.process) {
      p.restartAfterStop = true;
      RequestStop(p);
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
    if (p.process) {
      p.restartAfterStop = true;
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

std::filesystem::path PluginHost::PluginDir(const std::string& id) const {
  std::lock_guard lock(mutex_);
  return Find(id).dir;
}

void PluginHost::AddLog(Plugin& p, const std::string& level, const std::string& text) {
  p.logs.push_back({{"t", UnixMs()}, {"level", level}, {"text", text}});
  while (p.logs.size() > kMaxLogLines) p.logs.pop_front();
  ++version_;
}

// 输出协议（扩展机制方案 5.5）：每行一个 JSON 对象；其他内容按日志处理
void PluginHost::OnLine(Plugin& p, std::string line, bool fromStderr) {
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
  if (line.empty()) return;
  std::lock_guard lock(mutex_);
  if (fromStderr) {
    AddLog(p, "error", line);
    return;
  }
  json m;
  try {
    m = json::parse(line);
  } catch (const std::exception&) {
    AddLog(p, "info", line);
    return;
  }
  const std::string type = m.is_object() ? m.value("type", "") : "";
  const auto text = [&] { return m.contains("text") && m["text"].is_string() ? m["text"].get<std::string>() : ""; };
  if (type == "status") {
    p.statusText = text();
    ++version_;
  } else if (type == "metric" && m.contains("key") && m["key"].is_string()) {
    json metric = {{"key", m["key"]},
                   {"label", m.value("label", m["key"].get<std::string>())},
                   {"value", m.value("value", json())},
                   {"unit", m.value("unit", "")}};
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
    const std::string level = m.value("level", "info");
    AddLog(p, level == "warning" || level == "error" ? level : "info", text());
  } else if (type == "notify") {
    const uint64_t now = NowMs();
    AddLog(p, "info", "提示：" + text());
    if (now - p.lastNotify >= kNotifyIntervalMs) {
      p.lastNotify = now;
      notifications_.push_back({p.name, text()});
    }
  } else {
    AddLog(p, "info", line);
  }
}

void PluginHost::Launch(Plugin& p) {
  p.error.clear();
  p.statusText.clear();
  p.metrics.clear();

  // 入口：含路径或在插件目录中存在时相对插件目录，否则在 PATH 中查找
  std::filesystem::path app = FromUtf8(p.command);
  if (app.has_parent_path() || std::filesystem::exists(p.dir / app)) {
    app = p.dir / app;
  } else {
    wchar_t found[MAX_PATH];
    if (!SearchPathW(nullptr, app.c_str(), L".exe", MAX_PATH, found, nullptr)) {
      throw std::runtime_error("在 PATH 中找不到 " + p.command);
    }
    app = found;
  }
  std::wstring cmdline = Quote(app.wstring());
  for (const std::string& a : p.args) cmdline += L" " + Quote(FromUtf8(a));

  const auto data = dataDir_ / "plugin-data" / FromUtf8(p.id);
  std::filesystem::create_directories(data);
  wchar_t existing[32767];
  const DWORD n = GetEnvironmentVariableW(L"PYTHONPATH", existing, 32767);
  const std::wstring pythonPath = sdkDir_.wstring() + (n ? L";" + std::wstring(existing, n) : L"");
  std::wstring env = BuildEnvironment({{L"NR_PLUGIN_ID", FromUtf8(p.id)},
                                       {L"NR_PLUGIN_DIR", p.dir.wstring()},
                                       {L"NR_PLUGIN_DATA", data.wstring()},
                                       {L"NR_PLUGIN_SETTINGS", FromUtf8(p.settings.dump())},
                                       {L"NR_RESULTS_MAPPING", kResultsMappingName},
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
  if (!CreatePipe(&inRead, &inWrite, &sa, 0) || !CreatePipe(&outRead, &outWrite, &sa, 0) ||
      !CreatePipe(&errRead, &errWrite, &sa, 0)) {
    closeAll();
    throw std::runtime_error("创建管道失败");
  }
  SetHandleInformation(inWrite, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(outRead, HANDLE_FLAG_INHERIT, 0);
  SetHandleInformation(errRead, HANDLE_FLAG_INHERIT, 0);

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
  const std::wstring dir = p.dir.wstring();
  const BOOL ok = CreateProcessW(app.c_str(), cmdline.data(), nullptr, nullptr, TRUE,
                                 CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT |
                                     EXTENDED_STARTUPINFO_PRESENT,
                                 env.data(), dir.c_str(), &si.StartupInfo, &pi);
  const DWORD err = GetLastError();
  DeleteProcThreadAttributeList(attrs);
  if (!ok) {
    closeAll();
    throw std::runtime_error("启动失败（错误 " + std::to_string(err) + "）：" + ToUtf8(cmdline));
  }
  if (job_) AssignProcessToJobObject(job_, pi.hProcess);
  ResumeThread(pi.hThread);
  CloseHandle(pi.hThread);
  CloseHandle(inRead);
  CloseHandle(outWrite);
  CloseHandle(errWrite);

  p.process = pi.hProcess;
  p.pid = pi.dwProcessId;
  p.stdinWrite = inWrite;
  p.stdoutRead = outRead;
  p.stderrRead = errRead;
  p.state = "running";
  p.startedAt = NowMs();
  p.stopDeadline = 0;
  const auto reader = [this, &p](HANDLE pipe, bool fromStderr) {
    std::string pending;
    char buf[4096];
    DWORD got = 0;
    while (ReadFile(pipe, buf, sizeof(buf), &got, nullptr) && got > 0) {
      pending.append(buf, got);
      for (size_t pos; (pos = pending.find('\n')) != std::string::npos;) {
        OnLine(p, pending.substr(0, pos), fromStderr);
        pending.erase(0, pos + 1);
      }
      if (pending.size() > 65536) {  // 没有换行的超长输出按块处理
        OnLine(p, pending, fromStderr);
        pending.clear();
      }
    }
    if (!pending.empty()) OnLine(p, pending, fromStderr);
  };
  p.outReader = std::thread(reader, outRead, false);
  p.errReader = std::thread(reader, errRead, true);
  AddLog(p, "info", "已启动，进程号 " + std::to_string(p.pid));
}

// 请求停止：关闭标准输入，宽限期后结束进程（由监督线程执行）
void PluginHost::RequestStop(Plugin& p) {
  if (!p.process || p.state == "stopping") return;
  if (p.stdinWrite) {
    CloseHandle(p.stdinWrite);
    p.stdinWrite = nullptr;
  }
  p.state = "stopping";
  p.stopDeadline = NowMs() + kStopGraceMs;
  ++version_;
}

void PluginHost::Collect(Plugin& p) {
  // 插件的子进程可能继承了管道写端，读取线程迟迟读不到结束；取消阻塞的读取
  for (std::thread* t : {&p.outReader, &p.errReader}) {
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

void PluginHost::Supervise() {
  std::unique_lock lock(mutex_);
  for (;;) {
    const uint64_t now = NowMs();
    std::vector<Plugin*> exited;
    bool busy = false;
    for (auto& [id, ptr] : plugins_) {
      Plugin& p = *ptr;
      if (p.process) {
        busy = true;
        if (WaitForSingleObject(p.process, 0) == WAIT_OBJECT_0) {
          exited.push_back(&p);
        } else if (p.state == "stopping" && now >= p.stopDeadline) {
          TerminateProcess(p.process, 1);
          AddLog(p, "warning", "未在 2 秒内退出，已结束进程");
          p.stopDeadline = UINT64_MAX;
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
    }

    // 回收在锁外进行：读取线程处理输出时需要持锁
    if (!exited.empty()) {
      lock.unlock();
      for (Plugin* p : exited) Collect(*p);
      lock.lock();
      for (Plugin* pp : exited) {
        Plugin& p = *pp;
        DWORD code = 0;
        GetExitCodeProcess(p.process, &code);
        CloseHandle(p.process);
        p.process = nullptr;
        p.pid = 0;
        if (p.stdinWrite) {
          CloseHandle(p.stdinWrite);
          p.stdinWrite = nullptr;
        }
        const bool requested = p.state == "stopping";
        AddLog(p, requested ? "info" : "error", (requested ? "已停止，退出代码 " : "进程退出，代码 ") + std::to_string(code));
        if (requested) {
          p.state = "stopped";
          if (p.restartAfterStop) {
            p.restartAfterStop = false;
            p.exits.clear();
          }
        } else {
          // 插件应持续运行；未经请求的退出按策略重启
          p.exits.push_back(now);
          while (!p.exits.empty() && now - p.exits.front() > kCrashWindowMs) p.exits.pop_front();
          if (p.exits.size() >= kCrashLimit) {
            p.state = "failed";
            p.error = "60 秒内退出 " + std::to_string(kCrashLimit) + " 次，已停止；重新启用以重试";
          } else {
            const uint64_t backoff = std::min<uint64_t>(kMaxBackoffMs, 1000ull << (p.exits.size() - 1));
            p.state = "backoff";
            p.nextStart = now + backoff;
            p.error = "进程退出（代码 " + std::to_string(code) + "），" + std::to_string(backoff / 1000) + " 秒后重启";
          }
        }
        ++version_;
      }
      continue;
    }

    if (quit_ && !busy) return;
    wake_.wait_for(lock, std::chrono::milliseconds(100));
  }
}

}  // namespace nr::app
