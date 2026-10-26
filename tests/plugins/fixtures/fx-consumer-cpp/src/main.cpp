// 测试插件：与 fx-consumer（Python）相同的行为，用于检查 C++ SDK 与 Python SDK 的一致性。
// 以指标报告收到的帧数、缺口、各会话帧数，以及帧与元数据的会话是否一致。

#include "neural_reflex.hpp"

#include <map>
#include <mutex>
#include <string>
#include <thread>

int main() {
  nr::sdk::Plugin plugin;
  const nlohmann::json cfg = plugin.Settings();
  std::mutex mutex;
  uint64_t count = 0, gaps = 0, mismatch = 0;
  std::map<std::string, uint64_t> sessions;

  std::thread([&] {
    while (!plugin.WaitStop(200)) {
      std::lock_guard lock(mutex);
      plugin.Metric("count", count);
      plugin.Metric("gaps", gaps);
      plugin.Metric("mismatch", mismatch);
      plugin.Metric("sessions", sessions);
    }
  }).detach();

  nr::sdk::ReadOptions options;
  options.mode = cfg.value("mode", "latest");
  options.maxFps = cfg.value("maxFps", 0.0);
  const int sleepMs = cfg.value("sleepMs", 0);
  nr::sdk::ForEachFrame(plugin, options, [&](const nr::sdk::Frame& f, const nlohmann::json& meta) {
    {
      std::lock_guard lock(mutex);
      ++count;
      gaps += f.gap;
      ++sessions[std::to_string(f.session)];
      if (meta.value("session", int64_t{0}) != f.session) ++mismatch;
    }
    if (sleepMs) std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    return true;
  });
  return 0;
}
