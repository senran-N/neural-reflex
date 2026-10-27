#pragma once

// 界面与检测会话之间的桥：解析界面发来的命令，向界面推送状态与统计（JSON）。
// 只在界面线程调用；实时处理在 Session 的线程中进行。

#include "pipeline/session.h"
#include "plugins/plugin_host.h"
#include "targets.h"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace nr::app {

struct ModelInfo {
  std::filesystem::path path;
  std::string name;
  uint64_t sizeBytes = 0;
  int inputSize = 0;
  std::string precision;  // FP16 / FP32 / INT8，未知为空
  std::string head;
  std::string decoder;      // 解码器名称；清单未声明且没有记录输出形状时为空，启动时再确定
  std::string unsupported;  // 非空时为不能使用的原因
  int numKeypoints = 0;
  std::map<int, std::string> classNames;
  bool engineCached = false;
};

struct AppSettings {
  std::filesystem::path model;
  float conf = 0.25f;
  float iou = 0.7f;
  std::string mode = "wgc";  // wgc | dd
  bool overlay = true;
  bool cudaGraph = true;
  bool fp16 = true;             // 硬件支持时启用 FP16 半精度加速
  std::string roiMode = "full"; // 局部检测区域模式：full | center | custom
  int roiWidth = 0;             // 0 为自动使用模型输入边长
  int roiHeight = 0;            // 0 为自动使用模型输入边长
  int roiX = 0;                 // custom 模式起点 X
  int roiY = 0;                 // custom 模式起点 Y
  bool gpuRealtime = false;  // 进程 GPU 调度优先级设为实时级（设计方案 6.4）
  bool publishResults = false;  // 结果写入共享内存（扩展机制方案第 5 节）
  std::wstring lastTargetExe;
  nlohmann::json plugins = nlohmann::json::object();  // 插件启用状态与设置（插件宿主创建前的读取结果）
};

class Controller {
 public:
  using Sender = std::function<void(const std::string& json)>;

  Controller(Sender send, std::filesystem::path appDir, std::filesystem::path dataDir);
  ~Controller();

  void OnMessage(const std::string& json, HWND owner);
  void OnTimer();  // 约 4 Hz

 private:
  void SendState();
  void SendTargets();
  void SendModels();
  void SendStats();
  void SendPlugins();
  void SendSessionToPlugins();
  void ExportDiagnostics();
  void Toast(const std::string& level, const std::string& text);

  void SelectModel(const std::filesystem::path& path);
  void BrowseModel(HWND owner);
  void Start();
  void Stop();
  void LoadSettings();
  void SaveSettings() const;
  ModelInfo ReadModel(const std::filesystem::path& path) const;
  std::vector<std::filesystem::path> FindModels() const;

  Sender send_;
  std::filesystem::path appDir_, dataDir_;
  AppSettings settings_;
  ModelInfo model_;

  // 当前采集目标
  std::string targetKind_ = "none";  // none | window | monitor
  WindowTarget window_;
  MonitorTarget monitor_;

  Session session_;
  SessionPhase lastPhase_ = SessionPhase::kIdle;
  double lastSentSecond_ = -1;
  std::string gpuName_;

  std::unique_ptr<PluginHost> plugins_;
  uint64_t sentPluginsVersion_ = 0;

  // 其他线程完成的工作（诊断导出）留给界面线程提示
  std::mutex asyncMutex_;
  std::vector<std::pair<std::string, std::string>> asyncToasts_;  // (级别, 文字)
  std::filesystem::path asyncReveal_;                             // 在资源管理器中选中的文件
};

}  // namespace nr::app
