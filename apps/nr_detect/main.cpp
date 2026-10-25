// 检测程序：采集 → D3D11/CUDA 交接 → GPU 预处理 → TensorRT for RTX → GPU 后处理 → 覆盖层。
// --image-list 为离线模式：逐张图片运行同一条 GPU 流水线并输出 JSON，用于与
// Python 参考流水线逐项对比（tools/model_export/compare_cpp.py）。

#include "capture/dd_capture.h"
#include "capture/wgc_capture.h"
#include "common/d3d_util.h"
#include "common/latest_slot.h"
#include "common/qpc.h"
#include "common/stats.h"
#include "common/wic_image.h"
#include "common/win_util.h"
#include "inference/d3d_cuda_interop.h"
#include "inference/detector.h"
#include "overlay/overlay.h"

#include <unknwn.h>
#include <winrt/base.h>

#include <atomic>
#include <clocale>
#include <cstdio>
#include <cwchar>
#include <exception>
#include <fstream>
#include <string>
#include <thread>
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
    const nr::DetectionOutput& r = detector.Run(context.Get(), texture.Get(), region, 0, 0);
    const nr::StageTimes& t = detector.LastTimes();

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

struct OverlayResult {
  std::vector<RECT> boxes;
  std::vector<int> classes;
  int64_t acquireQpc = 0;
};

void OverlayThread(IDXGIAdapter1* adapter, RECT screenRect, const nr::LatestSlot<OverlayResult>& slot,
                   const std::atomic<bool>& running, std::atomic<uint64_t>& presents, std::atomic<bool>& failed) {
  nr::SetCurrentThreadName(L"nr-overlay");
  try {
    nr::Overlay overlay(adapter, screenRect);
    HANDLE waitable = overlay.FrameWaitable();
    uint64_t seen = 0;
    OverlayResult current;
    std::vector<nr::ColorRect> rects;
    while (running) {
      const DWORD r = MsgWaitForMultipleObjectsEx(1, &waitable, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
      if (r == WAIT_OBJECT_0 + 1) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) DispatchMessageW(&msg);
        continue;
      }
      if (r != WAIT_OBJECT_0) continue;
      slot.TakeNewer(seen, current);
      // 结果过期则清除（设计方案 5.2）
      const bool stale = nr::QpcToMs(nr::QpcNow() - current.acquireQpc) > 100;
      rects.clear();
      if (!stale) {
        // 高饱和度固定调色板，按类别循环取色
        static constexpr float kPalette[][3] = {{0.0f, 1.0f, 0.3f}, {1.0f, 0.2f, 0.6f}, {0.1f, 0.7f, 1.0f},
                                                {1.0f, 0.85f, 0.0f}, {1.0f, 0.45f, 0.0f}, {0.7f, 0.3f, 1.0f}};
        for (size_t i = 0; i < current.boxes.size(); ++i) {
          const RECT& b = current.boxes[i];
          const float* c = kPalette[static_cast<size_t>(current.classes[i]) % std::size(kPalette)];
          nr::AppendOutline(rects, static_cast<float>(b.left), static_cast<float>(b.top),
                            static_cast<float>(b.right - b.left), static_cast<float>(b.bottom - b.top), 3.0f,
                            c[0], c[1], c[2], 1.0f);
        }
      }
      overlay.Present(rects);
      presents++;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "覆盖层错误：%s\n", e.what());
    failed = true;
  }
}

void PrintPercentiles(const char* label, const std::vector<double>& ms) {
  const nr::Percentiles p = nr::ComputePercentiles(ms);
  std::printf("  %-26s n=%-7zu P50 %7.3f  P95 %7.3f  P99 %7.3f  最大 %7.3f ms\n", label, p.count, p.p50, p.p95,
              p.p99, p.max);
}

// 实时模式
int RunLive(const Options& o) {
  nr::EnableDpiAwareness();
  nr::DisablePowerThrottling();
  nr::SetCurrentThreadName(L"nr-detect");
  winrt::init_apartment(winrt::apartment_type::multi_threaded);

  HWND target = o.windowTitle.empty() ? FindWindowW(L"NeuralReflexTestSource", nullptr)
                                      : FindWindowW(nullptr, o.windowTitle.c_str());
  if (!target) {
    std::fprintf(stderr, "未找到目标窗口\n");
    return 1;
  }
  HMONITOR monitor = MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST);
  MONITORINFO mi{sizeof(mi)};
  GetMonitorInfoW(monitor, &mi);
  nr::AdapterInfo adapter = nr::FindAdapterForMonitor(monitor);
  nr::ComPtr<ID3D11DeviceContext> context;
  auto device = nr::CreateDevice(adapter.adapter.Get(), &context);
  const int cudaDevice = nr::CudaDeviceForLuid(adapter.luid);
  std::wprintf(L"适配器：%ls [LUID %ls]，CUDA 设备 %d\n", adapter.name.c_str(), nr::FormatLuid(adapter.luid).c_str(),
               cudaDevice);

  const int64_t setupStart = nr::QpcNow();
  nr::Detector detector(device.Get(), cudaDevice, MakeDetectorOptions(o));
  std::printf("模型 %s，准备耗时 %.0f ms\n", o.model.string().c_str(), nr::QpcToMs(nr::QpcNow() - setupStart));

  std::unique_ptr<nr::CaptureSource> source =
      o.mode == Mode::kWgcWindow ? nr::CreateWgcWindowCapture(device.Get(), target)
                                 : nr::CreateDuplicationCapture(device.Get(), adapter.output.Get());
  std::printf("采集方式：%s，时长 %.0f 秒\n", source->Name(), o.duration);

  nr::LatestSlot<OverlayResult> slot;
  std::atomic<bool> running{true}, overlayFailed{false};
  std::atomic<uint64_t> presents{0};
  std::thread overlayThread;
  if (o.overlay) {
    overlayThread = std::thread(OverlayThread, adapter.adapter.Get(), mi.rcMonitor, std::cref(slot),
                                std::cref(running), std::ref(presents), std::ref(overlayFailed));
  }

  std::vector<double> latency, pre, inf, post;
  uint64_t frames = 0, detections = 0, overflowFrames = 0, outOfFrame = 0;
  const int64_t freq = nr::QpcFrequency();
  const int64_t start = nr::QpcNow();
  const int64_t end = start + static_cast<int64_t>(o.duration * freq);
  int64_t secStart = start;
  uint64_t secFrames = 0;
  std::vector<double> secLatency;
  OverlayResult result;

  while (nr::QpcNow() < end && !overlayFailed && IsWindow(target)) {
    nr::CapturedFrame frame;
    const nr::AcquireStatus status = source->AcquireLatest(100, frame);
    if (status == nr::AcquireStatus::kLost) {
      if (!source->Recreate()) break;
      continue;
    }
    if (status == nr::AcquireStatus::kNewFrame) {
      // 检测区域为目标窗口客户区（设计方案 4.4）
      RECT client{};
      GetClientRect(target, &client);
      POINT origin{0, 0};
      ClientToScreen(target, &origin);
      const LONG x = origin.x - frame.screenOrigin.x, y = origin.y - frame.screenOrigin.y;
      if (x < 0 || y < 0 || x + client.right > static_cast<LONG>(frame.width) ||
          y + client.bottom > static_cast<LONG>(frame.height) || client.right <= 0 || client.bottom <= 0) {
        ++outOfFrame;
      } else {
        const D3D11_BOX region{static_cast<UINT>(x), static_cast<UINT>(y), 0, static_cast<UINT>(x + client.right),
                               static_cast<UINT>(y + client.bottom), 1};
        const nr::DetectionOutput& r = detector.Run(context.Get(), frame.texture, region,
                                                    static_cast<float>(origin.x), static_cast<float>(origin.y));
        const int64_t done = nr::QpcNow();

        result.boxes.clear();
        result.classes.clear();
        for (int i = 0; i < r.count; ++i) {
          const nr::Detection& d = r.detections[i];
          result.boxes.push_back({static_cast<LONG>(d.x1), static_cast<LONG>(d.y1), static_cast<LONG>(d.x2),
                                  static_cast<LONG>(d.y2)});
          result.classes.push_back(d.cls);
        }
        result.acquireQpc = frame.acquireQpc;
        slot.Publish(result);

        ++frames;
        ++secFrames;
        detections += static_cast<uint64_t>(r.count);
        if (r.overflow) ++overflowFrames;
        const double ms = nr::QpcToMs(done - frame.acquireQpc);
        latency.push_back(ms);
        secLatency.push_back(ms);
        const nr::StageTimes& t = detector.LastTimes();
        pre.push_back(t.preprocessMs);
        inf.push_back(t.inferenceMs);
        post.push_back(t.postprocessMs);
      }
    }

    const int64_t now = nr::QpcNow();
    if (now - secStart >= freq) {
      const double secs = static_cast<double>(now - secStart) / freq;
      const nr::Percentiles p = nr::ComputePercentiles(secLatency);
      std::printf("[%5.1fs] 检测 %6.1f/s  取得→结果 P50 %6.2f P95 %6.2f ms  框 %d\n", nr::QpcToMs(now - start) / 1000,
                  secFrames / secs, p.p50, p.p95, static_cast<int>(result.boxes.size()));
      secStart = now;
      secFrames = 0;
      secLatency.clear();
    }
  }

  running = false;
  if (overlayThread.joinable()) overlayThread.join();
  detector.SaveCaches();

  const double total = nr::QpcToMs(nr::QpcNow() - start) / 1000.0;
  std::printf("\n==== 汇总（%.1f 秒，%s）====\n", total, source->Name());
  std::printf("  完成检测 %llu 次（%.1f/s），平均每帧 %.1f 个框，候选溢出帧 %llu，区域不在画面内 %llu\n",
              static_cast<unsigned long long>(frames), frames / total,
              frames ? static_cast<double>(detections) / frames : 0.0, static_cast<unsigned long long>(overflowFrames),
              static_cast<unsigned long long>(outOfFrame));
  PrintPercentiles("取得→结果可用（主机）", latency);
  PrintPercentiles("GPU 预处理", pre);
  PrintPercentiles("GPU 推理", inf);
  PrintPercentiles("GPU 后处理+回读", post);
  if (o.overlay) {
    std::printf("  覆盖层呈现 %llu 次（%.1f/s）\n", static_cast<unsigned long long>(presents.load()), presents / total);
  }
  return overlayFailed ? 1 : 0;
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
