// nr_plugin_run：以命令行运行宿主的同一个插件托管实现，托管单个插件（扩展机制方案 7.7）。
//
// 输出：每行一个 JSON 事件，"event" 为 state、status、health、metric、log、notify、resources、diagnostics、exit。
// 输入：标准输入每行一个 JSON 命令：
//   {"cmd":"action","id":...}  {"cmd":"settings","values":{...}}  {"cmd":"restart"}  {"cmd":"enable"}
//   {"cmd":"disable"}  {"cmd":"session","data":{...}}  {"cmd":"diagnose"}  {"cmd":"quit"}
// 标准输入结束、收到 quit 或到达 --duration 时停止插件并退出。

#include "common/win_util.h"
#include "plugins/plugin_host.h"

#include <nlohmann/json.hpp>

#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace {

using json = nlohmann::json;

std::mutex g_outMutex;

void Emit(const json& j) {
  const std::string line = j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n";
  std::lock_guard lock(g_outMutex);
  std::fwrite(line.data(), 1, line.size(), stdout);
  std::fflush(stdout);
}

std::filesystem::path ExeDir() {
  wchar_t path[MAX_PATH * 2];
  const DWORD n = GetModuleFileNameW(nullptr, path, static_cast<DWORD>(std::size(path)));
  return std::filesystem::path(std::wstring(path, n)).parent_path();
}

void Usage() {
  std::fwprintf(stderr,
                L"用法：nr_plugin_run <插件目录> [--settings JSON] [--data-dir 目录] [--sdk 目录] [--duration 秒]\n"
                L"                    [--mapping 共享内存名] [--resources] [--check]\n");
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  SetConsoleOutputCP(CP_UTF8);
  std::filesystem::path dir, dataDir, sdkDir;
  std::wstring mapping;
  json settings = json::object();
  double duration = 0;
  bool check = false, resources = false;
  for (int i = 1; i < argc; ++i) {
    const std::wstring a = argv[i];
    const auto next = [&]() -> std::wstring { return i + 1 < argc ? argv[++i] : L""; };
    try {
      if (a == L"--settings") settings = json::parse(nr::ToUtf8(next()));
      else if (a == L"--data-dir") dataDir = next();
      else if (a == L"--sdk") sdkDir = next();
      else if (a == L"--mapping") mapping = next();
      else if (a == L"--duration") duration = std::stod(next());
      else if (a == L"--check") check = true;
      else if (a == L"--resources") resources = true;
      else if (dir.empty() && a.rfind(L"--", 0) != 0) dir = a;
      else throw std::runtime_error("未知参数");
    } catch (const std::exception&) {
      Usage();
      return 2;
    }
  }
  if (dir.empty()) {
    Usage();
    return 2;
  }
  dir = std::filesystem::absolute(dir);

  const nr::PluginManifest m = nr::ReadPluginManifest(dir);
  if (check) {
    Emit({{"event", "check"},
          {"ok", m.invalid.empty() && m.incompatible.empty()},
          {"id", m.id},
          {"name", m.name},
          {"version", m.version},
          {"invalid", m.invalid},
          {"incompatible", m.incompatible},
          {"activeFeatures", m.ActiveFeatures()}});
    return m.invalid.empty() && m.incompatible.empty() ? 0 : 1;
  }

  if (dataDir.empty()) {
    wchar_t temp[MAX_PATH];
    GetTempPathW(MAX_PATH, temp);
    dataDir = std::filesystem::path(temp) / L"nr_plugin_run";
  }
  if (sdkDir.empty()) {
    const auto exe = ExeDir();
    sdkDir = std::filesystem::exists(exe / "sdk" / "python") ? exe / "sdk" / "python"
                                                             : exe.parent_path().parent_path().parent_path() / "sdk" / "python";
  }

  nr::PluginHostOptions options;
  options.pluginDirs = {dir};
  options.dataDir = dataDir;
  options.sdkDir = sdkDir;
  options.hostVersion = "dev";
  options.resultsMapping = mapping;
  {
  nr::PluginHost host(std::move(options));
  host.Start({{m.id, {{"enabled", true}, {"settings", settings}}}});

  // 命令从读取线程交给主循环
  std::mutex cmdMutex;
  std::deque<json> commands;
  std::atomic<bool> inputClosed{false};
  std::thread input([&] {
    for (std::string line; std::getline(std::cin, line);) {
      if (line.empty()) continue;
      try {
        std::lock_guard lock(cmdMutex);
        commands.push_back(json::parse(line));
      } catch (const std::exception& e) {
        Emit({{"event", "error"}, {"text", std::string("无法解析命令：") + e.what()}});
      }
    }
    inputClosed = true;
  });
  input.detach();  // 标准输入可能一直不结束；进程退出时一并结束

  const auto started = std::chrono::steady_clock::now();
  uint64_t seenVersion = 0, seenLog = 0;
  json last = json::object();
  bool quitting = false;
  auto quitDeadline = std::chrono::steady_clock::time_point::max();

  for (;;) {
    std::deque<json> pending;
    {
      std::lock_guard lock(cmdMutex);
      pending.swap(commands);
    }
    for (const json& c : pending) {
      const std::string cmd = c.value("cmd", "");
      try {
        if (cmd == "action") host.InvokeAction(m.id, c.at("id").get<std::string>());
        else if (cmd == "settings") host.SetSettings(m.id, c.at("values"));
        else if (cmd == "restart") host.Restart(m.id);
        else if (cmd == "enable") host.SetEnabled(m.id, true);
        else if (cmd == "disable") host.SetEnabled(m.id, false);
        else if (cmd == "session") host.SetSession(c.at("data"));
        else if (cmd == "diagnose") host.CollectDiagnostics([](json plugins) { Emit({{"event", "diagnostics"}, {"plugins", plugins}}); });
        else if (cmd == "quit") inputClosed = true;
        else Emit({{"event", "error"}, {"text", "未知命令 " + cmd}});
      } catch (const std::exception& e) {
        Emit({{"event", "error"}, {"text", e.what()}});
      }
    }

    for (const auto& n : host.TakeNotifications()) Emit({{"event", "notify"}, {"level", n.level}, {"text", n.text}});

    if (host.Version() != seenVersion) {
      seenVersion = host.Version();
      const json snapshot = host.Snapshot();
      for (const json& p : snapshot["items"]) {
        if (p["id"] != m.id) continue;
        for (const json& l : p["logs"]) {
          if (l["seq"].get<uint64_t>() <= seenLog) continue;
          seenLog = l["seq"].get<uint64_t>();
          json e = {{"event", "log"}, {"level", l["level"]}, {"text", l["text"]}};
          if (l.contains("fields")) e["fields"] = l["fields"];
          Emit(e);
        }
        if (p["state"] != last.value("state", json()) || p["error"] != last.value("error", json())) {
          Emit({{"event", "state"}, {"state", p["state"]}, {"error", p["error"]}, {"pid", p["pid"]}, {"restarts", p["restarts"]}});
        }
        if (p["status"] != last.value("status", json())) Emit({{"event", "status"}, {"text", p["status"]}});
        if (p["health"] != last.value("health", json())) Emit({{"event", "health"}, {"health", p["health"]}});
        for (const json& metric : p["metrics"]) {
          bool changed = true;
          for (const json& old : last.value("metrics", json::array())) {
            if (old["key"] == metric["key"]) changed = old["value"] != metric["value"];
          }
          if (changed) Emit({{"event", "metric"}, {"key", metric["key"]}, {"value", metric["value"]}});
        }
        if (resources && p["resources"] != last.value("resources", json()) && !p["resources"].is_null()) {
          Emit({{"event", "resources"}, {"resources", p["resources"]}});
        }
        last = p;
      }
    }

    const auto now = std::chrono::steady_clock::now();
    const bool timeUp = duration > 0 && now - started >= std::chrono::duration<double>(duration);
    if (!quitting && (inputClosed || timeUp)) {
      quitting = true;
      quitDeadline = now + std::chrono::seconds(40);
      try {
        host.SetEnabled(m.id, false);
      } catch (const std::exception&) {
      }
    }
    if (quitting) {
      const std::string state = last.value("state", "");
      const bool idle = last.value("pid", 0) == 0 && state != "stopping" && state != "starting" && state != "running";
      if (idle || now >= quitDeadline) break;
    }
    Sleep(20);
  }
  Emit({{"event", "exit"}, {"state", last.value("state", "")}});
  }
  // 读取线程可能仍阻塞在标准输入上：不经过静态析构直接退出
  ExitProcess(0);
}
