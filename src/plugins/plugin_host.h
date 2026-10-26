#pragma once

// 外部进程插件的宿主（扩展机制方案 5.5、第 7 节）：发现插件包、托管插件进程、解析输出协议、
// 控制通道（事件、热更新、请求）、校验设置、测量资源。
// 所有公开方法可在任意线程调用；状态变化使 Version() 递增，界面据此决定是否刷新。

#include <nlohmann/json.hpp>

#include <windows.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nr {

// 插件协议版本（7.2）：主版本不同则不兼容；次版本只做增量
inline constexpr int kPluginApiMajor = 1;
inline constexpr char kPluginApiText[] = "1.1";
// 宿主支持的协议特性（7.2）
const std::vector<std::string>& HostPluginFeatures();

// 插件清单（plugin.json）。invalid 非空表示清单无效，incompatible 非空表示与宿主不兼容。
struct PluginManifest {
  std::filesystem::path dir;
  std::string id, name, version, description;
  int api = 0;
  std::string command;
  std::vector<std::string> args;
  nlohmann::json settings = nlohmann::json::array();
  std::vector<std::string> features, optionalFeatures;
  uint64_t startTimeoutMs = 15000, stopTimeoutMs = 2000;
  std::string restart = "always";  // always | on-failure | never
  nlohmann::json actions = nlohmann::json::array();
  std::string invalid, incompatible;

  // 实际启用的特性：features 与 optionalFeatures 中宿主支持的部分
  std::vector<std::string> ActiveFeatures() const;
  bool Uses(const std::string& feature) const;
};

// 读取并校验插件目录中的 plugin.json；不抛出异常，问题写在 invalid / incompatible 中
PluginManifest ReadPluginManifest(const std::filesystem::path& dir);

// 按设置项声明补全并校验设置值，不合法的值改用缺省值（第 6 节）
nlohmann::json NormalizePluginSettings(const nlohmann::json& schema, const nlohmann::json& values);

struct ProcessEntry;  // 进程快照中的一项（插件进程树，7.3）

struct PluginHostOptions {
  std::vector<std::filesystem::path> searchDirs;  // 其中每个子目录是一个插件包；按优先级从低到高，同 id 时后者覆盖前者
  std::vector<std::filesystem::path> pluginDirs;  // 单独指定的插件包目录，优先级最高
  std::filesystem::path userDir;                  // 用户安装目录（也在 searchDirs 中），可为空
  std::filesystem::path dataDir;                  // plugin-data 与 logs 的上级目录
  std::filesystem::path sdkDir;                   // Python SDK，前置到插件的 PYTHONPATH
  std::string hostVersion;
  std::wstring resultsMapping;  // 结果共享内存名，为空时用 kResultsMappingName
};

class PluginHost {
 public:
  struct Notification {
    std::string plugin;  // 插件名称
    std::string level;   // info | warning | error
    std::string text;
  };

  explicit PluginHost(PluginHostOptions options);
  ~PluginHost();
  PluginHost(const PluginHost&) = delete;
  PluginHost& operator=(const PluginHost&) = delete;

  // saved 为 Saved() 的结果（id → {enabled, settings}）。扫描插件目录并启动已启用的插件。
  void Start(const nlohmann::json& saved);
  // 重新扫描：新增插件加入列表；未运行插件的清单重新读取
  void Rescan();

  nlohmann::json Saved() const;
  nlohmann::json Snapshot() const;
  uint64_t Version() const { return version_; }
  bool AnyEnabled() const;
  std::vector<Notification> TakeNotifications();

  // 没有该插件或插件不可用时抛出异常
  void SetEnabled(const std::string& id, bool enabled);
  void SetSettings(const std::string& id, const nlohmann::json& values);
  void Restart(const std::string& id);
  // 清单 actions 中的操作（7.4）；结果以提示返回（TakeNotifications）
  void InvokeAction(const std::string& id, const std::string& action);
  std::filesystem::path PluginDir(const std::string& id) const;
  const std::filesystem::path& UserDir() const { return options_.userDir; }

  // 检测会话状态（7.4 session 事件），state 为事件的 data
  void SetSession(const nlohmann::json& state);

  // 诊断（7.6）：向运行中且启用控制通道的插件发 diagnose 请求，全部回答或超时后
  // 在宿主线程调用 done（不持锁），参数为每个插件的诊断信息数组
  void CollectDiagnostics(std::function<void(nlohmann::json)> done);

 private:
  struct Plugin;
  using Callback = std::function<void(const nlohmann::json& result, const nlohmann::json& error)>;

  void Scan();
  void AddManifest(PluginManifest m, const char* source);
  void Supervise();
  void Launch(Plugin& p);
  void RequestStop(Plugin& p);
  void StopAsFailure(Plugin& p, const std::string& reason);
  void OnExit(Plugin& p, uint64_t now);
  void OnLine(Plugin& p, std::string line, bool fromStderr);
  void OnMessage(Plugin& p, const nlohmann::json& m, const std::string& line);
  void OnPluginRequest(Plugin& p, const nlohmann::json& m);
  void Send(Plugin& p, const nlohmann::json& message);
  void SendRequest(Plugin& p, const std::string& method, const nlohmann::json& params, uint64_t timeoutMs,
                   Callback callback);
  void FailPending(Plugin& p, const std::string& message);
  void AddLog(Plugin& p, const std::string& level, const std::string& text,
              const nlohmann::json& fields = nullptr);
  void PluginLog(Plugin& p, const std::string& level, const std::string& text,
                 const nlohmann::json& fields = nullptr);
  void WriteLogFile(Plugin& p, const nlohmann::json& entry);
  void SampleResources(Plugin& p, uint64_t now, const std::vector<ProcessEntry>& processes);
  void Collect(Plugin& p);  // 不持锁调用：等待读写线程结束并关闭句柄
  void WriterLoop(Plugin& p, HANDLE pipe);
  nlohmann::json PluginJson(const Plugin& p, size_t maxLogs) const;
  Plugin& Find(const std::string& id) const;
  std::wstring MappingName() const;

  PluginHostOptions options_;

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::map<std::string, std::unique_ptr<Plugin>> plugins_;
  nlohmann::json saved_ = nlohmann::json::object();
  nlohmann::json session_ = {{"state", "idle"}, {"session", 0}, {"publishing", false}, {"model", ""}};
  std::vector<Notification> notifications_;
  std::vector<std::function<void()>> deferred_;  // 在监督线程中不持锁执行
  std::atomic<uint64_t> version_{1};
  uint64_t logSeq_ = 0;
  bool quit_ = false;
  std::thread supervisor_;
};

}  // namespace nr
