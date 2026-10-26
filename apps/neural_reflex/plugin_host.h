#pragma once

// 外部进程插件的宿主（扩展机制方案 5.5）：发现插件包、托管插件进程、解析输出协议、校验设置。
// 所有公开方法可在任意线程调用；状态变化使 Version() 递增，界面据此决定是否刷新。

#include <nlohmann/json.hpp>

#include <windows.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nr::app {

inline constexpr int kPluginApi = 1;  // 宿主支持的插件协议版本

class PluginHost {
 public:
  struct Notification {
    std::string plugin;  // 插件名称
    std::string text;
  };

  // searchDirs 按优先级从低到高；同 id 时后者覆盖前者。最后一个视为用户安装目录。
  PluginHost(std::vector<std::filesystem::path> searchDirs, std::filesystem::path dataDir,
             std::filesystem::path sdkDir);
  ~PluginHost();
  PluginHost(const PluginHost&) = delete;
  PluginHost& operator=(const PluginHost&) = delete;

  // saved 为 Saved() 的结果（id → {enabled, settings}）。扫描插件目录并启动已启用的插件。
  void Start(const nlohmann::json& saved);
  // 重新扫描：新增插件加入列表；已停止插件的清单重新读取
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
  std::filesystem::path PluginDir(const std::string& id) const;
  const std::filesystem::path& UserDir() const { return searchDirs_.back(); }

 private:
  struct Plugin;

  void Scan();
  void Supervise();
  void Launch(Plugin& p);
  void RequestStop(Plugin& p);
  void OnLine(Plugin& p, std::string line, bool fromStderr);
  void AddLog(Plugin& p, const std::string& level, const std::string& text);
  void Collect(Plugin& p);  // 不持锁调用：等待读取线程结束并关闭句柄
  Plugin& Find(const std::string& id) const;

  std::vector<std::filesystem::path> searchDirs_;
  std::filesystem::path dataDir_, sdkDir_;
  HANDLE job_ = nullptr;

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::map<std::string, std::unique_ptr<Plugin>> plugins_;
  nlohmann::json saved_ = nlohmann::json::object();
  std::vector<Notification> notifications_;
  std::atomic<uint64_t> version_{1};
  bool quit_ = false;
  std::thread supervisor_;
};

}  // namespace nr::app
