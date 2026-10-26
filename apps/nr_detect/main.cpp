// 检测程序：采集 → D3D11/CUDA 交接 → GPU 预处理 → TensorRT for RTX → GPU 后处理 → 覆盖层。
// --image-list 为离线模式：逐张图片运行同一条 GPU 流水线并输出 JSON，用于与
// Python 参考流水线逐项对比（tools/model_export/compare_cpp.py）。

#include "common/d3d_util.h"
#include "common/qpc.h"
#include "common/wic_image.h"
#include "common/win_util.h"
#include "inference/d3d_cuda_interop.h"
#include "inference/detector.h"
#include "pipeline/session.h"

#include <cuda_runtime.h>
#include <unknwn.h>
#include <winrt/base.h>

#include <clocale>
#include <cstdio>
#include <cwchar>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

enum class Mode { kWgcWindow, kDuplication };

struct Options {
  std::filesystem::path model = L"models/reference/yolo26n-640-nms-fp16.onnx";
  std::filesystem::path cacheDir = L"cache/engines";
  float conf = 0.25f;
  float iou = 0.7f;
  bool cudaGraph = true;
  // 离线模式
  std::filesystem::path imageList;
  std::filesystem::path jsonOut;
  std::filesystem::path dumpDir;
  // 实时模式
  Mode mode = Mode::kWgcWindow;
  std::wstring windowTitle;  // 空则使用合成测试源
  double duration = 30;
  bool overlay = true;
  // 调度实验项
  int inflight = 1;
  nr::GpuPriority gpuPriority = nr::GpuPriority::kNormal;
  bool mmcss = false;
  std::wstring sync = L"auto";  // CUDA 等待方式：auto | spin | yield | block
  std::filesystem::path summaryJson;
};

void PrintUsage() {
  std::fwprintf(stderr,
                L"用法：\n"
                L"  nr_detect [--model x.onnx] [--conf 0.25] [--iou 0.7] [--no-cuda-graph]\n"
                L"            [--mode wgc|dd] [--window 标题] [--duration 秒] [--no-overlay]\n"
                L"  nr_detect --image-list 列表.txt --json 结果.jsonl [--dump-dir 目录]\n");
}

bool ParseOptions(int argc, wchar_t** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const wchar_t* { return i + 1 < argc ? argv[++i] : nullptr; };
    const std::wstring a = argv[i];
    const wchar_t* v = nullptr;
    if (a == L"--model" && (v = next())) o.model = v;
    else if (a == L"--cache-dir" && (v = next())) o.cacheDir = v;
    else if (a == L"--conf" && (v = next())) o.conf = std::wcstof(v, nullptr);
    else if (a == L"--iou" && (v = next())) o.iou = std::wcstof(v, nullptr);
    else if (a == L"--no-cuda-graph") o.cudaGraph = false;
    else if (a == L"--image-list" && (v = next())) o.imageList = v;
    else if (a == L"--json" && (v = next())) o.jsonOut = v;
    else if (a == L"--dump-dir" && (v = next())) o.dumpDir = v;
    else if (a == L"--mode" && (v = next())) {
      if (!std::wcscmp(v, L"wgc")) o.mode = Mode::kWgcWindow;
      else if (!std::wcscmp(v, L"dd")) o.mode = Mode::kDuplication;
      else return false;
    } else if (a == L"--window" && (v = next())) o.windowTitle = v;
    else if (a == L"--duration" && (v = next())) o.duration = std::wcstod(v, nullptr);
    else if (a == L"--no-overlay") o.overlay = false;
    else if (a == L"--inflight" && (v = next())) o.inflight = _wtoi(v);
    else if (a == L"--gpu-priority" && (v = next())) {
      std::string name;
      for (const wchar_t* c = v; *c; ++c) name += static_cast<char>(*c);
      if (!nr::ParseGpuPriority(name, o.gpuPriority)) return false;
    } else if (a == L"--mmcss") o.mmcss = true;
    else if (a == L"--sync" && (v = next())) o.sync = v;
    else if (a == L"--summary-json" && (v = next())) o.summaryJson = v;
    else return false;
  }
  return true;
}

nr::DetectorOptions MakeDetectorOptions(const Options& o) {
  nr::DetectorOptions d;
  d.engine.onnx = o.model;
  d.engine.cacheDir = o.cacheDir;
  d.engine.cudaGraph = o.cudaGraph;
  d.confThreshold = o.conf;
  d.iouThreshold = o.iou;
  return d;
}

std::string Utf8(const std::wstring& w) {
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
  std::string s(static_cast<size_t>(n > 0 ? n - 1 : 0), '\0');
  if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), n, nullptr, nullptr);
  return s;
}

// 离线模式：同一条 GPU 流水线处理图片列表
int RunImages(const Options& o) {
  winrt::init_apartment(winrt::apartment_type::multi_threaded);
  nr::ComPtr<IDXGIFactory1> factory;
  nr::Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
  nr::ComPtr<IDXGIAdapter1> adapter;
  nr::Check(factory->EnumAdapters1(0, &adapter), "EnumAdapters1");
  DXGI_ADAPTER_DESC1 ad{};
  adapter->GetDesc1(&ad);
  nr::ComPtr<ID3D11DeviceContext> context;
  auto device = nr::CreateDevice(adapter.Get(), &context);
  nr::Detector detector(device.Get(), nr::CudaDeviceForLuid(ad.AdapterLuid), MakeDetectorOptions(o));

  std::wifstream list(o.imageList);
  std::ofstream json(o.jsonOut, std::ios::binary);
  if (!list || !json) {
    std::fprintf(stderr, "无法打开图片列表或输出文件\n");
    return 1;
  }
  if (!o.dumpDir.empty()) std::filesystem::create_directories(o.dumpDir);
  std::vector<float> input(3ull * detector.InputSize() * detector.InputSize());

  std::wstring line;
  size_t count = 0;
  while (std::getline(list, line)) {
    if (line.empty()) continue;
    const nr::BgraImage image = nr::LoadBgraImage(line);
    auto texture = nr::CreateTexture(device.Get(), image);
    const D3D11_BOX region{0, 0, 0, image.width, image.height, 1};
    nr::StageTimes t;
    detector.Submit(context.Get(), texture.Get(), region, 0, 0, 0);
    const nr::DetectionOutput& r = detector.Wait(0, &t);

    std::string s = "{\"image\":\"";
    for (char c : Utf8(line)) s += (c == '\\') ? std::string("/") : std::string(1, c);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "\",\"w\":%u,\"h\":%u,\"candidates\":%d,\"overflow\":%d,\"times\":[%.4f,%.4f,%.4f],\"det\":[",
                  image.width, image.height, r.candidates, r.overflow, t.preprocessMs, t.inferenceMs, t.postprocessMs);
    s += buf;
    for (int i = 0; i < r.count; ++i) {
      const nr::Detection& d = r.detections[i];
      std::snprintf(buf, sizeof(buf), "%s[%.4f,%.4f,%.4f,%.4f,%.6f,%d]", i ? "," : "", d.x1, d.y1, d.x2, d.y2,
                    d.score, d.cls);
      s += buf;
    }
    s += "]}\n";
    json << s;

    if (!o.dumpDir.empty()) {
      detector.ReadInput(input.data());
      std::ofstream dump(o.dumpDir / (std::filesystem::path(line).stem().wstring() + L".bin"), std::ios::binary);
      dump.write(reinterpret_cast<const char*>(input.data()), static_cast<std::streamsize>(input.size() * sizeof(float)));
    }
    ++count;
  }
  detector.SaveCaches();
  std::printf("已处理 %zu 张图片\n", count);
  return 0;
}

void PrintPercentiles(const char* label, const nr::Percentiles& p) {
  std::printf("  %-26s n=%-7zu P50 %7.3f  P95 %7.3f  P99 %7.3f  最大 %7.3f ms\n", label, p.count, p.p50, p.p95,
              p.p99, p.max);
}

// 实时模式：与图形界面共用 Session
int RunLive(const Options& o) {
  nr::EnableDpiAwareness();
  nr::DisablePowerThrottling();

  HWND target = o.windowTitle.empty() ? FindWindowW(L"NeuralReflexTestSource", nullptr)
                                      : FindWindowW(nullptr, o.windowTitle.c_str());
  if (!target) {
    std::fprintf(stderr, "未找到目标窗口\n");
    return 1;
  }

  nr::SessionConfig config;
  config.window = target;
  config.mode = o.mode == Mode::kWgcWindow ? nr::CaptureMode::kWgc : nr::CaptureMode::kDuplication;
  config.detector = MakeDetectorOptions(o);
  config.overlay = o.overlay;
  config.inflight = o.inflight;
  config.gpuPriority = o.gpuPriority;
  config.mmcss = o.mmcss;

  // CUDA 等待方式需在上下文创建前设置，作用于整个进程（设计方案 6.2）
  if (o.sync != L"auto") {
    const unsigned flags = o.sync == L"spin" ? cudaDeviceScheduleSpin
                           : o.sync == L"yield" ? cudaDeviceScheduleYield
                                                : cudaDeviceScheduleBlockingSync;
    int count = 0;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; ++i) {
      cudaSetDevice(i);
      cudaSetDeviceFlags(flags);
    }
  }

  nr::Session session;
  session.Start(config);
  const int64_t start = nr::QpcNow();
  double lastPrinted = 0;
  nr::SessionSnapshot s;
  bool announced = false;
  for (;;) {
    Sleep(200);
    s = session.Snapshot(5);
    if (s.phase == nr::SessionPhase::kFailed) {
      std::fprintf(stderr, "错误：%s\n", s.error.c_str());
      return 1;
    }
    if (s.phase == nr::SessionPhase::kRunning && !announced) {
      std::printf("适配器 %s，模型 %s，准备耗时 %.0f ms\n", s.adapter.c_str(), o.model.string().c_str(), s.prepareMs);
      announced = true;
    }
    for (const nr::SessionSecond& sec : s.seconds) {
      if (sec.t <= lastPrinted) continue;
      lastPrinted = sec.t;
      std::printf("[%5.1fs] 检测 %6.1f/s  取得→结果 P50 %6.2f P95 %6.2f ms  推理 %5.2f ms  框 %.1f\n", sec.t,
                  sec.detectionsPerSec, sec.latency.p50, sec.latency.p95, sec.inferMs, sec.boxes);
    }
    if (s.phase == nr::SessionPhase::kRunning && s.elapsed >= o.duration) break;
    if (nr::QpcToMs(nr::QpcNow() - start) > (o.duration + 120) * 1000) break;  // 准备阶段异常耗时的保护
  }
  session.Stop();
  s = session.Snapshot(0);
  if (s.phase == nr::SessionPhase::kFailed) {
    std::fprintf(stderr, "错误：%s\n", s.error.c_str());
    return 1;
  }

  std::printf("\n==== 汇总（%.1f 秒）====\n", s.elapsed);
  std::printf("  完成检测 %llu 次（%.1f/s），候选溢出帧 %llu，本进程显存 %.0f MB / 预算 %.0f MB\n",
              static_cast<unsigned long long>(s.detections), s.elapsed > 0 ? s.detections / s.elapsed : 0.0,
              static_cast<unsigned long long>(s.overflowFrames), s.vramUsedBytes / 1048576.0,
              s.vramBudgetBytes / 1048576.0);
  PrintPercentiles("取得→结果可用（主机）", s.latency);
  PrintPercentiles("GPU 预处理", s.pre);
  PrintPercentiles("GPU 推理", s.infer);
  PrintPercentiles("GPU 后处理+回读", s.post);
  std::printf("  GPU 调度优先级 请求 %s（状态 0x%08lX），生效 %s\n", nr::GpuPriorityName(o.gpuPriority),
              static_cast<unsigned long>(s.gpuPriorityStatus),
              s.gpuPriorityEffective >= 0 && s.gpuPriorityEffective <= 5
                  ? nr::GpuPriorityName(static_cast<nr::GpuPriority>(s.gpuPriorityEffective))
                  : "未知");

  if (!o.summaryJson.empty()) {
    FILE* f = nullptr;
    if (_wfopen_s(&f, o.summaryJson.c_str(), L"w") == 0 && f) {
      auto pct = [&](const char* name, const nr::Percentiles& p, bool comma) {
        std::fprintf(f, "\"%s\":{\"n\":%zu,\"p50\":%.4f,\"p95\":%.4f,\"p99\":%.4f,\"max\":%.4f}%s", name, p.count, p.p50,
                     p.p95, p.p99, p.max, comma ? "," : "");
      };
      // 稳态：去掉开头 3 秒的预热，按秒取检测速率
      double rateSum = 0;
      int rateN = 0;
      for (const auto& sec : session.Snapshot(100000).seconds) {
        if (sec.t > 3.0) {
          rateSum += sec.detectionsPerSec;
          ++rateN;
        }
      }
      std::fprintf(f, "{\"elapsed\":%.3f,\"detections\":%llu,\"steady_rate\":%.3f,\"overflow_frames\":%llu,"
                      "\"vram_used_mb\":%.1f,\"vram_budget_mb\":%.1f,\"gpu_priority_status\":%ld,\"gpu_priority_effective\":%d,",
                   s.elapsed, static_cast<unsigned long long>(s.detections), rateN ? rateSum / rateN : 0.0,
                   static_cast<unsigned long long>(s.overflowFrames), s.vramUsedBytes / 1048576.0,
                   s.vramBudgetBytes / 1048576.0, s.gpuPriorityStatus, s.gpuPriorityEffective);
      pct("latency", s.latency, true);
      pct("pre", s.pre, true);
      pct("infer", s.infer, true);
      pct("post", s.post, false);
      std::fprintf(f, "}\n");
      std::fclose(f);
    }
  }
  return 0;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  SetConsoleOutputCP(CP_UTF8);
  std::setlocale(LC_ALL, ".UTF-8");
  Options o;
  if (!ParseOptions(argc, argv, o)) {
    PrintUsage();
    return 2;
  }
  try {
    return o.imageList.empty() ? RunLive(o) : RunImages(o);
  } catch (const winrt::hresult_error& e) {
    std::fwprintf(stderr, L"WinRT 错误 0x%08X：%ls\n", static_cast<unsigned>(e.code()), e.message().c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
  }
  return 1;
}
