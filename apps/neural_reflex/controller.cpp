#include "controller.h"

#include "common/d3d_util.h"
#include "resource.h"

#include <nlohmann/json.hpp>
#include <shellapi.h>
#include <shobjidl.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>

namespace nr::app {
namespace {

using json = nlohmann::json;

std::string HwndString(HWND hwnd) { return std::to_string(reinterpret_cast<uintptr_t>(hwnd)); }

HWND HwndFromString(const std::string& s) {
  return reinterpret_cast<HWND>(static_cast<uintptr_t>(std::stoull(s)));
}

json PercentilesJson(const Percentiles& p) {
  return {{"n", p.count}, {"p50", p.p50}, {"p95", p.p95}, {"p99", p.p99}, {"max", p.max}};
}

// 系统版本，写入诊断（7.6）
std::string WindowsVersion() {
  wchar_t display[64] = L"", build[32] = L"";
  DWORD ubr = 0, size = sizeof(display);
  const wchar_t* key = L"SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
  RegGetValueW(HKEY_LOCAL_MACHINE, key, L"DisplayVersion", RRF_RT_REG_SZ, nullptr, display, &size);
  size = sizeof(build);
  RegGetValueW(HKEY_LOCAL_MACHINE, key, L"CurrentBuild", RRF_RT_REG_SZ, nullptr, build, &size);
  size = sizeof(ubr);
  RegGetValueW(HKEY_LOCAL_MACHINE, key, L"UBR", RRF_RT_REG_DWORD, nullptr, &ubr, &size);
  return "Windows " + ToUtf8(display) + " " + ToUtf8(build) + "." + std::to_string(ubr);
}

std::string Lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

}  // namespace

Controller::Controller(Sender send, std::filesystem::path appDir, std::filesystem::path dataDir)
    : send_(std::move(send)), appDir_(std::move(appDir)), dataDir_(std::move(dataDir)) {
  std::filesystem::create_directories(dataDir_ / "engines");
  LoadSettings();
  if (settings_.model.empty() || !std::filesystem::exists(settings_.model)) {
    const auto models = FindModels();
    // 默认选 FP16 参考模型
    auto it = std::find_if(models.begin(), models.end(),
                           [](const auto& p) { return Lower(p.filename().string()).find("fp16") != std::string::npos; });
    settings_.model = it != models.end() ? *it : (models.empty() ? std::filesystem::path() : models.front());
  }
  if (!settings_.model.empty()) model_ = ReadModel(settings_.model);

  // 外部进程插件（扩展机制方案 5.5）：程序目录、源码树（开发时）、用户目录，后者覆盖前者
  {
    const auto repo = appDir_.parent_path().parent_path().parent_path();
    std::vector<std::filesystem::path> dirs = {appDir_ / "plugins"};
    std::error_code ec;
    if (std::filesystem::is_directory(repo / "plugins", ec) &&
        !std::filesystem::equivalent(repo / "plugins", appDir_ / "plugins", ec)) {
      dirs.push_back(repo / "plugins");
    }
    dirs.push_back(dataDir_ / "plugins");
    PluginHostOptions options;
    options.searchDirs = std::move(dirs);
    options.userDir = dataDir_ / "plugins";
    options.dataDir = dataDir_;
    options.sdkDir = std::filesystem::exists(appDir_ / "sdk" / "python") ? appDir_ / "sdk" / "python"
                                                                         : repo / "sdk" / "python";
    options.hostVersion = NR_VERSION_TEXT;
    plugins_ = std::make_unique<PluginHost>(std::move(options));
    SendSessionToPlugins();
    plugins_->Start(settings_.plugins);
  }

  // 上次的目标程序仍在运行时自动选中
  if (!settings_.lastTargetExe.empty()) {
    for (const auto& w : ListWindows(GetCurrentProcessId())) {
      if (_wcsicmp(w.exePath.c_str(), settings_.lastTargetExe.c_str()) == 0) {
        targetKind_ = "window";
        window_ = w;
        break;
      }
    }
  }

  try {
    AdapterInfo a = FindAdapterForMonitor(MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY));
    gpuName_ = ToUtf8(a.name);
  } catch (const std::exception&) {
  }
}

Controller::~Controller() {
  session_.Stop();
  SaveSettings();
  plugins_.reset();  // 请求插件退出，宽限期后结束进程
}

void Controller::OnMessage(const std::string& text, HWND owner) {
  json msg;
  try {
    msg = json::parse(text);
  } catch (const json::exception&) {
    return;
  }
  const std::string cmd = msg.value("cmd", "");
  try {
    if (cmd == "hello") {
      SendState();
      SendTargets();
      SendModels();
      SendPlugins();
    } else if (cmd == "setPluginEnabled") {
      const bool enabled = msg.at("enabled").get<bool>();
      plugins_->SetEnabled(msg.at("id").get<std::string>(), enabled);
      SaveSettings();
      // 插件依赖结果共享；运行中的会话未发布时重新启动以开启
      if (enabled && session_.Active() && !session_.Snapshot(0).publishing) {
        Start();
        Toast("info", "已重新启动检测，以便插件读取结果");
      }
      SendPlugins();
    } else if (cmd == "setPluginSettings") {
      plugins_->SetSettings(msg.at("id").get<std::string>(), msg.at("settings"));
      SaveSettings();
      SendPlugins();
    } else if (cmd == "pluginAction") {
      plugins_->InvokeAction(msg.at("id").get<std::string>(), msg.at("action").get<std::string>());
    } else if (cmd == "exportDiagnostics") {
      ExportDiagnostics();
    } else if (cmd == "restartPlugin") {
      plugins_->Restart(msg.at("id").get<std::string>());
      SendPlugins();
    } else if (cmd == "rescanPlugins") {
      plugins_->Rescan();
      SendPlugins();
    } else if (cmd == "openPluginDir") {
      const std::string id = msg.value("id", "");
      const auto dir = id.empty() ? plugins_->UserDir() : plugins_->PluginDir(id);
      ShellExecuteW(owner, L"open", dir.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    } else if (cmd == "refreshTargets") {
      SendTargets();
    } else if (cmd == "selectWindow") {
      HWND hwnd = HwndFromString(msg.at("hwnd").get<std::string>());
      for (const auto& w : ListWindows(GetCurrentProcessId())) {
        if (w.hwnd == hwnd) {
          targetKind_ = "window";
          window_ = w;
          settings_.lastTargetExe = w.exePath;
        }
      }
      SaveSettings();
      SendState();
    } else if (cmd == "selectMonitor") {
      const int index = msg.at("index").get<int>();
      for (const auto& m : ListMonitors()) {
        if (m.index == index) {
          targetKind_ = "monitor";
          monitor_ = m;
        }
      }
      SendState();
    } else if (cmd == "selectModel") {
      SelectModel(FromUtf8(msg.at("path").get<std::string>()));
    } else if (cmd == "browseModel") {
      BrowseModel(owner);
    } else if (cmd == "setSettings") {
      const json& s = msg.at("settings");
      const bool restart = session_.Active() && (s.value("mode", settings_.mode) != settings_.mode ||
                                                 s.value("overlay", settings_.overlay) != settings_.overlay ||
                                                 s.value("cudaGraph", settings_.cudaGraph) != settings_.cudaGraph ||
                                                 s.value("fp16", settings_.fp16) != settings_.fp16 ||
                                                 s.value("roiMode", settings_.roiMode) != settings_.roiMode ||
                                                 s.value("roiWidth", settings_.roiWidth) != settings_.roiWidth ||
                                                 s.value("roiHeight", settings_.roiHeight) != settings_.roiHeight ||
                                                 s.value("roiX", settings_.roiX) != settings_.roiX ||
                                                 s.value("roiY", settings_.roiY) != settings_.roiY ||
                                                 s.value("gpuRealtime", settings_.gpuRealtime) != settings_.gpuRealtime ||
                                                 s.value("publishResults", settings_.publishResults) != settings_.publishResults);
      settings_.conf = std::clamp(s.value("conf", settings_.conf), 0.01f, 0.99f);
      settings_.iou = std::clamp(s.value("iou", settings_.iou), 0.05f, 0.95f);
      settings_.mode = s.value("mode", settings_.mode) == "dd" ? "dd" : "wgc";
      settings_.overlay = s.value("overlay", settings_.overlay);
      settings_.cudaGraph = s.value("cudaGraph", settings_.cudaGraph);
      settings_.fp16 = s.value("fp16", settings_.fp16);
      settings_.roiMode = s.value("roiMode", settings_.roiMode);
      settings_.roiWidth = s.value("roiWidth", settings_.roiWidth);
      settings_.roiHeight = s.value("roiHeight", settings_.roiHeight);
      settings_.roiX = s.value("roiX", settings_.roiX);
      settings_.roiY = s.value("roiY", settings_.roiY);
      settings_.gpuRealtime = s.value("gpuRealtime", settings_.gpuRealtime);
      settings_.publishResults = s.value("publishResults", settings_.publishResults);
      session_.SetThresholds(settings_.conf, settings_.iou);
      SaveSettings();
      if (restart) Start();
      SendState();
    } else if (cmd == "start") {
      Start();
    } else if (cmd == "stop") {
      Stop();
    }
  } catch (const std::exception& e) {
    Toast("error", e.what());
  }
}

void Controller::OnTimer() {
  const SessionSnapshot s = session_.Snapshot(0);
  if (s.phase != lastPhase_) {
    lastPhase_ = s.phase;
    // 引擎在准备阶段编译并写入缓存，进入运行后刷新状态
    const auto stem = model_.path.stem().string();
    model_.engineCached = std::filesystem::exists(dataDir_ / "engines" / (stem + (settings_.fp16 ? ".fp16.engine" : ".engine"))) ||
                          std::filesystem::exists(dataDir_ / "engines" / (stem + ".engine"));
    SendState();
    if (s.phase == SessionPhase::kFailed) Toast("error", s.error);
    if (s.phase == SessionPhase::kRunning && !s.publishError.empty()) Toast("warning", "结果共享未开启：" + s.publishError);
    SendSessionToPlugins();
  }
  if (s.phase == SessionPhase::kRunning || s.phase == SessionPhase::kPreparing) SendStats();
  if (plugins_->Version() != sentPluginsVersion_) SendPlugins();
  for (const auto& n : plugins_->TakeNotifications()) Toast(n.level, n.plugin + "：" + n.text);

  std::vector<std::pair<std::string, std::string>> toasts;
  std::filesystem::path reveal;
  {
    std::lock_guard lock(asyncMutex_);
    toasts = std::exchange(asyncToasts_, {});
    reveal = std::exchange(asyncReveal_, {});
  }
  for (const auto& [level, text] : toasts) Toast(level, text);
  if (!reveal.empty()) {
    const std::wstring args = L"/select,\"" + reveal.wstring() + L"\"";
    ShellExecuteW(nullptr, L"open", L"explorer.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
  }
}

void Controller::SendPlugins() {
  sentPluginsVersion_ = plugins_->Version();
  send_(plugins_->Snapshot().dump(-1, ' ', false, json::error_handler_t::replace));
}

// 检测会话状态通知插件（扩展机制方案 7.4 的 session 事件）
void Controller::SendSessionToPlugins() {
  const SessionSnapshot s = session_.Snapshot(0);
  plugins_->SetSession({{"state", PhaseName(s.phase)},
                        {"session", s.sessionId},
                        {"publishing", s.publishing},
                        {"model", model_.name}});
}

// 诊断导出（7.6）：宿主信息与各插件的状态、日志与 diagnose 回答写入一个 JSON 文件
void Controller::ExportDiagnostics() {
  const SessionSnapshot s = session_.Snapshot(30);
  json seconds = json::array();
  for (const SessionSecond& sec : s.seconds) {
    seconds.push_back({{"t", sec.t}, {"dps", sec.detectionsPerSec}, {"p50", sec.latency.p50}, {"p95", sec.latency.p95}});
  }
  json host = {{"version", NR_VERSION_TEXT},
               {"os", WindowsVersion()},
               {"gpu", gpuName_},
               {"pluginApi", kPluginApiText},
               {"pluginFeatures", HostPluginFeatures()},
               {"session",
                {{"phase", PhaseName(s.phase)},
                 {"error", s.error},
                 {"adapter", s.adapter},
                 {"decoder", s.decoder},
                 {"inputSize", s.inputSize},
                 {"publishing", s.publishing},
                 {"publishError", s.publishError},
                 {"detections", s.detections},
                 {"latency", PercentilesJson(s.latency)},
                 {"seconds", seconds}}},
               {"model", model_.name},
               {"settings",
                {{"conf", settings_.conf},
                 {"iou", settings_.iou},
                 {"mode", settings_.mode},
                 {"overlay", settings_.overlay},
                 {"cudaGraph", settings_.cudaGraph},
                 {"fp16", settings_.fp16},
                 {"roiMode", settings_.roiMode},
                 {"roiWidth", settings_.roiWidth},
                 {"roiHeight", settings_.roiHeight},
                 {"roiX", settings_.roiX},
                 {"roiY", settings_.roiY},
                 {"gpuRealtime", settings_.gpuRealtime},
                 {"publishResults", settings_.publishResults}}}};
  const std::filesystem::path dir = dataDir_ / "diagnostics";
  plugins_->CollectDiagnostics([this, host = std::move(host), dir](json plugins) {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    localtime_s(&tm, &now);
    char name[64];
    std::strftime(name, sizeof(name), "diag-%Y%m%d-%H%M%S.json", &tm);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / name;
    const json out = {{"generatedAt", std::chrono::duration_cast<std::chrono::milliseconds>(
                                          std::chrono::system_clock::now().time_since_epoch())
                                          .count()},
                      {"host", host},
                      {"plugins", std::move(plugins)}};
    std::ofstream f(path, std::ios::binary);
    f << out.dump(2, ' ', false, json::error_handler_t::replace);
    std::lock_guard lock(asyncMutex_);
    if (f) {
      asyncToasts_.push_back({"info", "已导出诊断：" + ToUtf8(path.wstring())});
      asyncReveal_ = path;
    } else {
      asyncToasts_.push_back({"error", "导出诊断失败：无法写入 " + ToUtf8(path.wstring())});
    }
  });
}

void Controller::Start() {
  if (model_.path.empty()) throw std::runtime_error("请先加载模型");
  SessionConfig config;
  if (targetKind_ == "window") {
    if (!IsWindow(window_.hwnd)) throw std::runtime_error("目标窗口已关闭，请重新选择");
    config.window = window_.hwnd;
  } else if (targetKind_ == "monitor") {
    config.monitor = monitor_.handle;
  } else {
    throw std::runtime_error("请先选择采集目标");
  }
  config.mode = settings_.mode == "dd" ? CaptureMode::kDuplication : CaptureMode::kWgc;
  config.detector.engine.onnx = model_.path;
  config.detector.engine.cacheDir = dataDir_ / "engines";
  config.detector.engine.cudaGraph = settings_.cudaGraph;
  config.detector.confThreshold = settings_.conf;
  config.detector.iouThreshold = settings_.iou;
  config.overlay = settings_.overlay;
  config.gpuPriority = settings_.gpuRealtime ? GpuPriority::kRealtime : GpuPriority::kNormal;
  config.publishResults = settings_.publishResults || plugins_->AnyEnabled();
  if (settings_.roiMode == "center") {
    config.roi.mode = RoiMode::kCenter;
  } else if (settings_.roiMode == "custom") {
    config.roi.mode = RoiMode::kCustom;
  } else {
    config.roi.mode = RoiMode::kFull;
  }
  config.roi.width = settings_.roiWidth;
  config.roi.height = settings_.roiHeight;
  config.roi.x = settings_.roiX;
  config.roi.y = settings_.roiY;
  lastSentSecond_ = -1;
  session_.Start(config);
  lastPhase_ = SessionPhase::kPreparing;
  SendState();
  SendSessionToPlugins();  // OnTimer 只在阶段变化时通知，这里已同步更新 lastPhase_
  json reset = {{"type", "statsReset"}};
  send_(reset.dump());
}

void Controller::Stop() {
  SendStats();
  session_.Stop();
  lastPhase_ = session_.Snapshot(0).phase;
  const auto stem = model_.path.stem().string();
  model_.engineCached = std::filesystem::exists(dataDir_ / "engines" / (stem + (settings_.fp16 ? ".fp16.engine" : ".engine"))) ||
                        std::filesystem::exists(dataDir_ / "engines" / (stem + ".engine"));
  SendState();
  SendSessionToPlugins();
}

void Controller::SelectModel(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) throw std::runtime_error("模型文件不存在");
  if (ReadModel(path).head.find("end2end") != std::string::npos) {
    throw std::runtime_error("暂不支持端到端检测头，请使用一对多检测头（需要 NMS）的模型");
  }
  const bool wasActive = session_.Active();
  model_ = ReadModel(path);
  settings_.model = path;
  SaveSettings();
  if (wasActive) Start();
  SendState();
  SendModels();
}

void Controller::BrowseModel(HWND owner) {
  Microsoft::WRL::ComPtr<IFileOpenDialog> dialog;
  if (FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog)))) return;
  const COMDLG_FILTERSPEC filter[] = {{L"ONNX 模型", L"*.onnx"}};
  dialog->SetFileTypes(1, filter);
  dialog->SetTitle(L"加载检测模型");
  if (FAILED(dialog->Show(owner))) return;  // 用户取消
  Microsoft::WRL::ComPtr<IShellItem> item;
  if (FAILED(dialog->GetResult(&item))) return;
  PWSTR path = nullptr;
  if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
    SelectModel(path);
    CoTaskMemFree(path);
  }
}

ModelInfo Controller::ReadModel(const std::filesystem::path& path) const {
  ModelInfo m;
  m.path = path;
  m.name = ToUtf8(path.stem().wstring());
  std::error_code ec;
  m.sizeBytes = std::filesystem::file_size(path, ec);
  const std::string lower = Lower(m.name);
  m.precision = lower.find("fp16") != std::string::npos   ? "FP16"
                : lower.find("int8") != std::string::npos ? "INT8"
                : lower.find("fp8") != std::string::npos  ? "FP8"
                : lower.find("fp32") != std::string::npos ? "FP32"
                                                          : "";
  const auto stem = path.stem().string();
  m.engineCached = std::filesystem::exists(dataDir_ / "engines" / (stem + (settings_.fp16 ? ".fp16.engine" : ".engine"))) ||
                   std::filesystem::exists(dataDir_ / "engines" / (stem + ".engine"));

  // 模型清单（扩展机制方案 4.1）；与检测会话使用同一套查找与解码器推断规则
  try {
    const ModelManifest manifest = LoadModelManifest(path);
    m.inputSize = manifest.imgsz;
    m.head = manifest.head;
    m.classNames = manifest.classNames;
    m.numKeypoints = manifest.numKeypoints;
    CheckSupportedPreprocess(manifest);
    if (!manifest.decoder.empty() || !manifest.outputShape.empty()) {
      m.decoder = ResolveDecoderName(manifest, manifest.outputShape);
    }
  } catch (const std::exception& e) {
    m.unsupported = e.what();
  }
  return m;
}

std::vector<std::filesystem::path> Controller::FindModels() const {
  std::vector<std::filesystem::path> found;
  for (const auto& root : {appDir_ / "models", appDir_.parent_path().parent_path().parent_path() / "models"}) {
    std::error_code ec;
    if (!std::filesystem::is_directory(root, ec)) continue;
    for (const auto& e : std::filesystem::recursive_directory_iterator(root, ec)) {
      if (e.is_regular_file() && Lower(e.path().extension().string()) == ".onnx") found.push_back(e.path());
    }
    if (!found.empty()) break;
  }
  std::sort(found.begin(), found.end());
  return found;
}

void Controller::SendState() {
  const SessionSnapshot s = session_.Snapshot(0);
  json target = {{"kind", targetKind_}};
  if (targetKind_ == "window") {
    target["hwnd"] = HwndString(window_.hwnd);
    target["title"] = ToUtf8(window_.title);
    target["exe"] = ToUtf8(std::filesystem::path(window_.exePath).filename().wstring());
    target["pid"] = window_.pid;
    target["width"] = window_.clientWidth;
    target["height"] = window_.clientHeight;
    target["icon"] = window_.icon;
    target["alive"] = IsWindow(window_.hwnd) != FALSE;
  } else if (targetKind_ == "monitor") {
    target["index"] = monitor_.index;
    target["width"] = monitor_.rect.right - monitor_.rect.left;
    target["height"] = monitor_.rect.bottom - monitor_.rect.top;
    target["refreshHz"] = monitor_.refreshHz;
    target["primary"] = monitor_.primary;
  }
  json classes = json::object();
  for (const auto& [k, v] : model_.classNames) classes[std::to_string(k)] = v;
  json model = nullptr;
  if (!model_.path.empty()) {
    model = {{"path", ToUtf8(model_.path.wstring())}, {"name", model_.name},     {"sizeBytes", model_.sizeBytes},
             {"inputSize", model_.inputSize},         {"precision", model_.precision}, {"head", model_.head},
             {"decoder", model_.decoder},             {"keypoints", model_.numKeypoints},
             {"classes", classes},                    {"engineCached", model_.engineCached}};
  }
  const json state = {
      {"type", "state"},
      {"phase", PhaseName(s.phase)},
      {"error", s.error},
      {"target", target},
      {"model", model},
      {"settings",
       {{"conf", settings_.conf},
        {"iou", settings_.iou},
        {"mode", settings_.mode},
        {"overlay", settings_.overlay},
        {"cudaGraph", settings_.cudaGraph},
        {"fp16", settings_.fp16},
        {"roiMode", settings_.roiMode},
        {"roiWidth", settings_.roiWidth},
        {"roiHeight", settings_.roiHeight},
        {"roiX", settings_.roiX},
        {"roiY", settings_.roiY},
        {"gpuRealtime", settings_.gpuRealtime},
        {"publishResults", settings_.publishResults}}},
      {"gpu", gpuName_},
  };
  send_(state.dump());
}

void Controller::SendTargets() {
  json windows = json::array();
  for (const auto& w : ListWindows(GetCurrentProcessId())) {
    windows.push_back({{"hwnd", HwndString(w.hwnd)},
                       {"title", ToUtf8(w.title)},
                       {"exe", ToUtf8(std::filesystem::path(w.exePath).filename().wstring())},
                       {"pid", w.pid},
                       {"width", w.clientWidth},
                       {"height", w.clientHeight},
                       {"minimized", w.minimized},
                       {"icon", w.icon}});
  }
  json monitors = json::array();
  for (const auto& m : ListMonitors()) {
    monitors.push_back({{"index", m.index},
                        {"width", m.rect.right - m.rect.left},
                        {"height", m.rect.bottom - m.rect.top},
                        {"refreshHz", m.refreshHz},
                        {"primary", m.primary}});
  }
  send_(json{{"type", "targets"}, {"windows", windows}, {"monitors", monitors}}.dump());
}

void Controller::SendModels() {
  json items = json::array();
  auto models = FindModels();
  if (!model_.path.empty() && std::find(models.begin(), models.end(), model_.path) == models.end()) {
    models.push_back(model_.path);
  }
  for (const auto& p : models) {
    const ModelInfo m = ReadModel(p);
    items.push_back({{"path", ToUtf8(p.wstring())},
                     {"name", m.name},
                     {"sizeBytes", m.sizeBytes},
                     {"precision", m.precision},
                     {"inputSize", m.inputSize},
                     {"classCount", m.classNames.size()},
                     {"engineCached", m.engineCached},
                     {"decoder", m.decoder},
                     {"keypoints", m.numKeypoints},
                     {"supported", m.unsupported.empty()},
                     {"unsupported", m.unsupported}});
  }
  send_(json{{"type", "models"}, {"items", items}}.dump());
}

void Controller::SendStats() {
  const SessionSnapshot s = session_.Snapshot(120);
  json seconds = json::array();
  for (const SessionSecond& sec : s.seconds) {
    if (sec.t <= lastSentSecond_) continue;
    lastSentSecond_ = sec.t;
    seconds.push_back({{"t", sec.t},
                       {"dps", sec.detectionsPerSec},
                       {"ops", sec.overlayPerSec},
                       {"p50", sec.latency.p50},
                       {"p95", sec.latency.p95},
                       {"p99", sec.latency.p99},
                       {"pre", sec.preMs},
                       {"infer", sec.inferMs},
                       {"post", sec.postMs},
                       {"boxes", sec.boxes},
                       {"overflow", sec.overflowFrames}});
  }
  json classes = json::array();
  for (const auto& [cls, count] : s.lastClasses) {
    auto it = model_.classNames.find(cls);
    classes.push_back({{"id", cls}, {"name", it != model_.classNames.end() ? it->second : std::to_string(cls)},
                       {"count", count}});
  }
  const json stats = {
      {"type", "stats"},
      {"phase", PhaseName(s.phase)},
      {"elapsed", s.elapsed},
      {"prepareMs", s.prepareMs},
      {"adapter", s.adapter},
      {"inputSize", s.inputSize},
      {"numClasses", s.numClasses},
      {"detections", s.detections},
      {"overflowFrames", s.overflowFrames},
      {"latency", PercentilesJson(s.latency)},
      {"pre", PercentilesJson(s.pre)},
      {"infer", PercentilesJson(s.infer)},
      {"post", PercentilesJson(s.post)},
      {"vramUsed", s.vramUsedBytes},
      {"vramBudget", s.vramBudgetBytes},
      {"seconds", seconds},
      {"classes", classes},
  };
  send_(stats.dump());
}

void Controller::Toast(const std::string& level, const std::string& text) {
  send_(json{{"type", "toast"}, {"level", level}, {"text", text}}.dump());
}

void Controller::LoadSettings() {
  std::ifstream f(dataDir_ / "settings.json");
  if (!f) return;
  try {
    const json j = json::parse(f);
    settings_.model = FromUtf8(j.value("model", std::string()));
    settings_.conf = j.value("conf", settings_.conf);
    settings_.iou = j.value("iou", settings_.iou);
    settings_.mode = j.value("mode", settings_.mode);
    settings_.overlay = j.value("overlay", settings_.overlay);
    settings_.cudaGraph = j.value("cudaGraph", settings_.cudaGraph);
    settings_.fp16 = j.value("fp16", settings_.fp16);
    settings_.roiMode = j.value("roiMode", settings_.roiMode);
    settings_.roiWidth = j.value("roiWidth", settings_.roiWidth);
    settings_.roiHeight = j.value("roiHeight", settings_.roiHeight);
    settings_.roiX = j.value("roiX", settings_.roiX);
    settings_.roiY = j.value("roiY", settings_.roiY);
    settings_.gpuRealtime = j.value("gpuRealtime", settings_.gpuRealtime);
    settings_.publishResults = j.value("publishResults", settings_.publishResults);
    settings_.plugins = j.value("plugins", json::object());
    settings_.lastTargetExe = FromUtf8(j.value("lastTargetExe", std::string()));
  } catch (const std::exception&) {
  }
}

void Controller::SaveSettings() const {
  const json j = {{"model", ToUtf8(settings_.model.wstring())},
                  {"conf", settings_.conf},
                  {"iou", settings_.iou},
                  {"mode", settings_.mode},
                  {"overlay", settings_.overlay},
                  {"cudaGraph", settings_.cudaGraph},
                  {"fp16", settings_.fp16},
                  {"roiMode", settings_.roiMode},
                  {"roiWidth", settings_.roiWidth},
                  {"roiHeight", settings_.roiHeight},
                  {"roiX", settings_.roiX},
                  {"roiY", settings_.roiY},
                  {"gpuRealtime", settings_.gpuRealtime},
                  {"publishResults", settings_.publishResults},
                  {"plugins", plugins_ ? plugins_->Saved() : settings_.plugins},
                  {"lastTargetExe", ToUtf8(settings_.lastTargetExe)}};
  std::ofstream f(dataDir_ / "settings.json");
  f << j.dump(2);
}

}  // namespace nr::app
