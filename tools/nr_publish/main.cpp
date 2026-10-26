// nr_publish：用宿主的同一个发布实现写结果共享内存（扩展机制方案 7.7）。
// 生成合成目标，或回放 tools/results/read_results.py --jsonl 的记录；可模拟多个会话与会话间隔。
// 没有 GPU 和模型也能开发、测试插件。

#include "common/qpc.h"
#include "common/win_util.h"
#include "results/result_publisher.h"

#include <nlohmann/json.hpp>

#include <windows.h>
#include <timeapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using json = nlohmann::json;

struct Options {
  double fps = 144;
  double duration = 0;  // 每个会话的秒数，0 表示不限
  uint64_t frames = 0;  // 每个会话的帧数，0 表示不限
  int sessions = 1;
  double gap = 1;          // 会话间隔秒数
  double startDelay = 0;   // 写入元数据后、发布第一帧前等待的秒数，便于读取方先接入
  int objects = 3;
  int keypoints = 0;
  std::wstring replay;
  std::wstring mapping = nr::kResultsMappingName;
  bool quiet = false;
};

std::atomic<bool> g_stop{false};

BOOL WINAPI OnCtrl(DWORD) {
  g_stop = true;
  return TRUE;
}

void Usage() {
  std::fwprintf(stderr,
                L"用法：nr_publish [--fps 144] [--duration 秒] [--frames 帧数] [--sessions 1] [--gap 秒]\n"
                L"                 [--start-delay 秒] [--objects 3] [--keypoints 0] [--replay 记录.jsonl]\n"
                L"                 [--mapping 共享内存名] [--quiet]\n"
                L"不指定 --duration 与 --frames 时持续发布，按 Ctrl+C 结束。\n");
}

bool Parse(int argc, wchar_t** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::wstring a = argv[i];
    const auto next = [&]() -> std::wstring { return i + 1 < argc ? argv[++i] : L""; };
    if (a == L"--fps") o.fps = std::stod(next());
    else if (a == L"--duration") o.duration = std::stod(next());
    else if (a == L"--frames") o.frames = std::stoull(next());
    else if (a == L"--sessions") o.sessions = std::stoi(next());
    else if (a == L"--gap") o.gap = std::stod(next());
    else if (a == L"--start-delay") o.startDelay = std::stod(next());
    else if (a == L"--objects") o.objects = std::stoi(next());
    else if (a == L"--keypoints") o.keypoints = std::stoi(next());
    else if (a == L"--replay") o.replay = next();
    else if (a == L"--mapping") o.mapping = next();
    else if (a == L"--quiet") o.quiet = true;
    else return false;
  }
  o.objects = std::clamp(o.objects, 0, nr::kMaxDetections);
  o.keypoints = std::clamp(o.keypoints, 0, nr::kMaxKeypoints);
  return o.fps > 0 && o.sessions > 0;
}

struct ReplayFrame {
  std::vector<nr::Detection> det;
  std::vector<nr::Keypoint> kpt;  // 每个目标 numKeypoints 个点，按目标顺序
  int numKeypoints = 0;
  int candidates = 0, overflow = 0;
  RECT region{};
};

std::vector<ReplayFrame> LoadReplay(const std::wstring& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("无法读取 " + nr::ToUtf8(path));
  std::vector<ReplayFrame> frames;
  for (std::string line; std::getline(f, line);) {
    if (line.empty()) continue;
    const json j = json::parse(line);
    ReplayFrame r;
    for (const auto& d : j.at("det")) {
      if (r.det.size() >= nr::kMaxDetections) break;
      r.det.push_back({d[0].get<float>(), d[1].get<float>(), d[2].get<float>(), d[3].get<float>(), d[4].get<float>(),
                       d[5].get<int32_t>()});
    }
    if (j.contains("kpt") && !j["kpt"].empty()) {
      r.numKeypoints = std::min<int>(static_cast<int>(j["kpt"][0].size() / 3), nr::kMaxKeypoints);
      for (size_t i = 0; i < r.det.size(); ++i) {
        const auto& pts = j["kpt"][i];
        for (int k = 0; k < r.numKeypoints; ++k) {
          r.kpt.push_back({pts[k * 3].get<float>(), pts[k * 3 + 1].get<float>(), pts[k * 3 + 2].get<float>()});
        }
      }
    }
    r.candidates = j.value("candidates", static_cast<int>(r.det.size()));
    r.overflow = j.value("overflow", 0);
    const auto region = j.value("region", std::vector<int>{0, 0, 1920, 1080});
    r.region = {region[0], region[1], region[0] + region[2], region[1] + region[3]};
    frames.push_back(std::move(r));
  }
  if (frames.empty()) throw std::runtime_error("回放记录为空");
  return frames;
}

// 合成目标：在 1920×1080 区域内沿不同的周期轨迹移动，分数缓慢变化
void Synthesize(const Options& o, uint64_t frame, nr::DetectionOutput& out, RECT& region) {
  region = {0, 0, 1920, 1080};
  out.count = o.objects;
  out.numKeypoints = o.keypoints;
  out.candidates = o.objects;
  out.overflow = 0;
  const double t = static_cast<double>(frame) / o.fps;
  for (int i = 0; i < o.objects; ++i) {
    const double cx = 960 + 700 * std::sin(t * (0.5 + 0.13 * i) + i);
    const double cy = 540 + 380 * std::cos(t * (0.4 + 0.11 * i) + 2 * i);
    const double w = 60 + 20 * i, h = 120 + 10 * i;
    nr::Detection& d = out.detections[i];
    d.x1 = static_cast<float>(cx - w / 2);
    d.y1 = static_cast<float>(cy - h / 2);
    d.x2 = static_cast<float>(cx + w / 2);
    d.y2 = static_cast<float>(cy + h / 2);
    d.score = static_cast<float>(0.6 + 0.35 * std::sin(t * 0.7 + i));
    d.cls = i % 3;
    for (int k = 0; k < o.keypoints; ++k) {
      out.keypoints[i * o.keypoints + k] = {static_cast<float>(cx + 10 * std::cos(k)),
                                            static_cast<float>(cy + 10 * std::sin(k)), 0.9f};
    }
  }
}

std::string Metadata(const Options& o, int64_t session, const char* state, int numKeypoints) {
  json names = json::array();
  for (int k = 0; k < numKeypoints; ++k) names.push_back("k" + std::to_string(k));
  const json j = {{"layout", nr::kResultsLayoutVersion},
                  {"session", session},
                  {"state", state},
                  {"pid", GetCurrentProcessId()},
                  {"model", o.replay.empty() ? "synthetic" : "replay"},
                  {"decoder", "nr_publish"},
                  {"inputSize", 640},
                  {"numClasses", 3},
                  {"coordinates", "screen-physical"},
                  {"clock", "qpc"},
                  {"synthetic", true},
                  {"classNames", {{"0", "person"}, {"1", "car"}, {"2", "dog"}}},
                  {"keypoints", {{"count", numKeypoints}, {"names", names}, {"skeleton", json::array()}}}};
  return j.dump();
}

int64_t UnixMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// 等到指定 QPC 时刻：较远时睡眠，最后 1.5 ms 自旋
void WaitUntil(int64_t target) {
  for (;;) {
    const double remaining = nr::QpcToMs(target - nr::QpcNow());
    if (remaining <= 0 || g_stop) return;
    if (remaining > 1.5) Sleep(static_cast<DWORD>(remaining - 1));
    else YieldProcessor();
  }
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  Options o;
  try {
    if (!Parse(argc, argv, o)) {
      Usage();
      return 2;
    }
  } catch (const std::exception&) {
    Usage();
    return 2;
  }
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCtrlHandler(OnCtrl, TRUE);
  timeBeginPeriod(1);

  try {
    std::vector<ReplayFrame> replay;
    if (!o.replay.empty()) replay = LoadReplay(o.replay);
    auto out = std::make_unique<nr::DetectionOutput>();
    json sessions = json::array();
    uint64_t total = 0;

    for (int s = 0; s < o.sessions && !g_stop; ++s) {
      if (s > 0) {
        const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(o.gap);
        while (!g_stop && std::chrono::steady_clock::now() < until) Sleep(10);
        if (g_stop) break;
      }
      const int numKeypoints = replay.empty() ? o.keypoints : replay.front().numKeypoints;
      const int64_t session = UnixMs();
      uint64_t published = 0;
      {
        nr::ResultPublisher publisher(o.mapping);
        publisher.PublishMetadata(Metadata(o, session, "running", numKeypoints));
        if (!o.quiet) std::fprintf(stderr, "会话 %lld 开始发布\n", static_cast<long long>(session));
        if (o.startDelay > 0) Sleep(static_cast<DWORD>(o.startDelay * 1000));

        const int64_t period = static_cast<int64_t>(static_cast<double>(nr::QpcFrequency()) / o.fps);
        const int64_t start = nr::QpcNow();
        int64_t secondStart = start;
        uint64_t secondFrames = 0;
        for (uint64_t i = 0; !g_stop; ++i) {
          if (o.frames && i >= o.frames) break;
          if (o.duration > 0 && nr::QpcToMs(nr::QpcNow() - start) >= o.duration * 1000) break;
          WaitUntil(start + static_cast<int64_t>(i) * period);
          nr::PublishedFrame f;
          f.session = session;
          f.frameIndex = i;
          f.acquireQpc = nr::QpcNow();
          if (replay.empty()) {
            Synthesize(o, i, *out, f.region);
          } else {
            const ReplayFrame& r = replay[i % replay.size()];
            out->count = static_cast<int32_t>(r.det.size());
            out->numKeypoints = r.numKeypoints;
            out->candidates = r.candidates;
            out->overflow = r.overflow;
            std::copy(r.det.begin(), r.det.end(), out->detections);
            std::copy(r.kpt.begin(), r.kpt.end(), out->keypoints);
            f.region = r.region;
          }
          f.resultQpc = nr::QpcNow();
          publisher.Publish(f, *out);
          ++published;
          ++secondFrames;
          const int64_t now = nr::QpcNow();
          if (!o.quiet && nr::QpcToMs(now - secondStart) >= 1000) {
            std::fprintf(stderr, "会话 %lld：已发布 %llu 帧，%.1f 帧/秒\n", static_cast<long long>(session),
                         static_cast<unsigned long long>(published),
                         static_cast<double>(secondFrames) * 1000.0 / nr::QpcToMs(now - secondStart));
            secondStart = now;
            secondFrames = 0;
          }
        }
        publisher.PublishMetadata(Metadata(o, session, "stopped", numKeypoints));
      }
      total += published;
      sessions.push_back({{"session", session}, {"frames", published}});
    }
    // 汇总写到标准输出，便于测试脚本读取
    std::printf("%s\n", json{{"published", total}, {"sessions", sessions}}.dump().c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    timeEndPeriod(1);
    return 1;
  }
  timeEndPeriod(1);
  return 0;
}
