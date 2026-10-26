// 示例插件：把每帧检测结果以 JSON 数据报发送到指定的 UDP 地址（扩展机制方案 7.8）。
// 实时消费者：只取最新帧，按设置限速；发送不阻塞，网络出错时计数并报告健康状态，而不是等待。
//
// 数据报（UTF-8 JSON，每个不超过 64 KB）：
//   {"type":"meta","session":...,"model":...,"classNames":{...},"keypoints":{...},"coordinates":"screen-physical"}
//     会话开始时与之后每 5 s 发送一次
//   {"type":"frame","session":...,"frame":帧号,"ts":取得画面的 Unix 毫秒数,"latency_ms":...,
//    "region":[x,y,w,h],"det":[[x1,y1,x2,y2,score,cls],...],"kpt":[[x,y,score,...],...]}
//     超过大小时去掉关键点、再截断目标，并带 "truncated": true

#include <winsock2.h>
#include <ws2tcpip.h>

#include "neural_reflex.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <set>
#include <string>

namespace {

using nlohmann::json;
using nr::sdk::Frame;
using nr::sdk::Plugin;

constexpr size_t kMaxDatagram = 65000;

struct Config {
  double maxFps = 60;
  std::set<std::string> classes;
  double minScore = 0.25;
  bool keypoints = false;

  explicit Config(const json& s) {
    maxFps = s.value("maxFps", 60.0);
    minScore = s.value("minScore", 0.25);
    keypoints = s.value("keypoints", false);
    const std::string list = s.value("classes", "");
    size_t start = 0;
    while (start <= list.size()) {
      const size_t comma = list.find(',', start);
      std::string c = list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
      c.erase(0, c.find_first_not_of(' '));
      c.erase(c.find_last_not_of(' ') + 1);
      if (!c.empty()) classes.insert(c);
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
  }
};

double Round(double v, double scale) { return std::round(v * scale) / scale; }

int64_t UnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

}  // namespace

int main() {
  Plugin plugin;
  const json settings = plugin.Settings();
  const std::string host = settings.value("host", "127.0.0.1");
  const int port = settings.value("port", 9100);

  if (host.empty()) plugin.Fail("目标地址为空，请在设置中填写");
  WSADATA wsa;
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) plugin.Fail("无法初始化 Winsock");
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;
  addrinfo* addr = nullptr;
  if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &addr) != 0 || !addr) {
    plugin.Fail("无法解析目标地址 " + host + "，请检查设置");
  }
  SOCKET sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock == INVALID_SOCKET) plugin.Fail("无法创建 UDP 套接字");
  u_long nonBlocking = 1;
  ioctlsocket(sock, FIONBIO, &nonBlocking);  // 发送缓冲满时丢弃而不是阻塞
  const std::string target = host + ":" + std::to_string(port);

  std::mutex configMutex;
  Config config(settings);
  std::atomic<uint64_t> sent{0}, bytes{0}, errors{0}, truncated{0};
  std::atomic<int64_t> lastErrorMs{0};
  std::atomic<int> lastError{0};

  const auto send = [&](const json& message) {
    const std::string data = message.dump(-1, ' ', false, json::error_handler_t::replace);
    if (sendto(sock, data.data(), static_cast<int>(data.size()), 0, addr->ai_addr, static_cast<int>(addr->ai_addrlen)) ==
        SOCKET_ERROR) {
      ++errors;
      lastError = WSAGetLastError();
      lastErrorMs = UnixMs();
      return false;
    }
    ++sent;
    bytes += data.size();
    return true;
  };

  plugin.OnSettings([&](const json& values) {
    std::lock_guard lock(configMutex);
    config = Config(values);
    plugin.Log("设置已更新：速率上限 " + std::to_string(static_cast<int>(config.maxFps)) + " 帧/秒");
  });
  plugin.OnAction("test", [&] {
    if (!send({{"type", "test"}, {"ts", UnixMs()}})) {
      throw std::runtime_error("发送失败（Winsock 错误 " + std::to_string(lastError.load()) + "）");
    }
    return json("已发送测试数据报到 " + target);
  });
  plugin.OnDiagnose([&] {
    return json{{"target", target}, {"sent", sent.load()}, {"bytes", bytes.load()}, {"errors", errors.load()},
                {"lastError", lastError.load()}};
  });

  // 指标与健康状态每秒报告一次，没有新帧时也更新
  std::thread([&] {
    uint64_t lastSent = 0, lastBytes = 0;
    while (!plugin.WaitStop(1000)) {
      const uint64_t s = sent, b = bytes;
      plugin.Metric("sent", s, "已发送数据报");
      plugin.Metric("rate", s - lastSent, "发送速率", "个/秒");
      plugin.Metric("bandwidth", Round((b - lastBytes) / 1024.0, 10), "带宽", "KB/s");
      plugin.Metric("errors", errors.load(), "发送失败");
      if (truncated) plugin.Metric("truncated", truncated.load(), "截断的帧");
      lastSent = s;
      lastBytes = b;
      if (UnixMs() - lastErrorMs < 5000) {
        plugin.Health("degraded", "发送失败（Winsock 错误 " + std::to_string(lastError.load()) + "）");
      } else {
        plugin.Health("ok");
      }
      plugin.Status("发送到 " + target);
    }
  }).detach();

  plugin.Log("发送到 " + target);
  int64_t metaSession = -1;
  auto lastMeta = std::chrono::steady_clock::time_point{};
  double rate = -1;
  nr::sdk::ReadOptions options;
  {
    std::lock_guard lock(configMutex);
    options.maxFps = rate = config.maxFps;
  }
  // 限速在读取方实现（7.1：消费速率是插件自己的选择）；速率设置改变时重新开始读取
  while (!plugin.StopRequested()) {
    nr::sdk::ForEachFrame(plugin, options, [&](const Frame& f, const json& meta) {
      Config c = [&] {
        std::lock_guard lock(configMutex);
        return config;
      }();
      if (c.maxFps != rate) {
        rate = c.maxFps;
        options.maxFps = rate;
        return false;  // 由外层循环以新速率重新读取
      }
      const auto now = std::chrono::steady_clock::now();
      if (f.session != metaSession || now - lastMeta > std::chrono::seconds(5)) {
        metaSession = f.session;
        lastMeta = now;
        send({{"type", "meta"},
              {"session", f.session},
              {"model", meta.value("model", "")},
              {"classNames", meta.value("classNames", json::object())},
              {"keypoints", meta.value("keypoints", json::object())},
              {"coordinates", meta.value("coordinates", "screen-physical")}});
      }
      const json names = meta.value("classNames", json::object());
      json det = json::array(), kpt = json::array();
      for (size_t i = 0; i < f.det.size(); ++i) {
        const auto& d = f.det[i];
        if (d.score < c.minScore) continue;
        if (!c.classes.empty()) {
          const std::string key = std::to_string(d.cls);
          const std::string name = names.contains(key) ? names[key].get<std::string>() : key;
          if (!c.classes.count(name)) continue;
        }
        det.push_back({Round(d.x1, 10), Round(d.y1, 10), Round(d.x2, 10), Round(d.y2, 10), Round(d.score, 1000), d.cls});
        if (c.keypoints && f.numKeypoints) {
          json pts = json::array();
          for (int j = 0; j < f.numKeypoints * 3; ++j) pts.push_back(Round(f.kpt[(i * f.numKeypoints) * 3 + j], 10));
          kpt.push_back(std::move(pts));
        }
      }
      json message = {{"type", "frame"},
                      {"session", f.session},
                      {"frame", f.index},
                      {"ts", UnixMs() - static_cast<int64_t>(f.latencyMs)},
                      {"latency_ms", Round(f.latencyMs, 1000)},
                      {"region", {f.region[0], f.region[1], f.region[2], f.region[3]}},
                      {"det", det}};
      if (c.keypoints && f.numKeypoints) message["kpt"] = kpt;
      // 超过数据报大小：先去掉关键点，再按比例截断目标
      if (message.dump().size() > kMaxDatagram) {
        message.erase("kpt");
        while (message.dump().size() > kMaxDatagram && !message["det"].empty()) {
          auto& list = message["det"];
          list.erase(list.begin() + static_cast<std::ptrdiff_t>(list.size() * 3 / 4), list.end());
        }
        message["truncated"] = true;
        ++truncated;
      }
      send(message);
      return true;
    });
  }
  closesocket(sock);
  freeaddrinfo(addr);
  WSACleanup();
  return 0;
}
