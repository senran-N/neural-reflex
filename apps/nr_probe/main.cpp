// 链路探针：采集合成测试源 → 解码帧标签 → 覆盖层画框。
// 推理接入前用标签里的真值框代替检测结果，验证采集、坐标还原、
// 最新帧交接和覆盖层呈现，并测量各段延迟。解码框不是检测结果。

#include "capture/dd_capture.h"
#include "capture/wgc_capture.h"
#include "common/d3d_util.h"
#include "common/frame_tag.h"
#include "common/latest_slot.h"
#include "common/qpc.h"
#include "common/stats.h"
#include "common/win_util.h"
#include "overlay/overlay.h"

#include <unknwn.h>
#include <winrt/base.h>

#include <atomic>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

enum class Mode { kWgcWindow, kWgcMonitor, kDuplication };

struct Options {
  Mode mode = Mode::kWgcWindow;
  double duration = 30;
  bool overlay = true;
  double staleMs = 100;  // 超过该时间没有新结果就清除旧框
  const wchar_t* csvPath = nullptr;
};

void PrintUsage() {
  std::fwprintf(stderr,
                L"用法：nr_probe [--mode wgc|wgc-monitor|dd] [--duration 秒] [--no-overlay]\n"
                L"               [--stale-ms N] [--csv 路径.csv]\n"
                L"需先运行 nr_testsource。\n");
}

bool ParseOptions(int argc, wchar_t** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const wchar_t* { return i + 1 < argc ? argv[++i] : nullptr; };
    const wchar_t* a = argv[i];
    const wchar_t* v = nullptr;
    if (!std::wcscmp(a, L"--mode") && (v = next())) {
      if (!std::wcscmp(v, L"wgc")) o.mode = Mode::kWgcWindow;
      else if (!std::wcscmp(v, L"wgc-monitor")) o.mode = Mode::kWgcMonitor;
      else if (!std::wcscmp(v, L"dd")) o.mode = Mode::kDuplication;
      else return false;
    } else if (!std::wcscmp(a, L"--duration") && (v = next())) {
      o.duration = std::wcstod(v, nullptr);
    } else if (!std::wcscmp(a, L"--no-overlay")) {
      o.overlay = false;
    } else if (!std::wcscmp(a, L"--stale-ms") && (v = next())) {
      o.staleMs = std::wcstod(v, nullptr);
    } else if (!std::wcscmp(a, L"--csv") && (v = next())) {
      o.csvPath = v;
    } else {
      return false;
    }
  }
  return o.duration > 0;
}

struct Result {
  uint32_t frameId = 0;
  int64_t sourceRenderQpc = 0;  // 测试源生成该帧的时刻（来自标签）
  int64_t acquireQpc = 0;
  RECT box{};                   // 屏幕坐标
};

std::vector<double> ToMs(const std::vector<int64_t>& ticks) {
  std::vector<double> ms;
  ms.reserve(ticks.size());
  for (int64_t t : ticks) ms.push_back(nr::QpcToMs(t));
  return ms;
}

void PrintPercentiles(const char* label, const std::vector<int64_t>& ticks) {
  const nr::Percentiles p = nr::ComputePercentiles(ToMs(ticks));
  std::printf("  %-28s n=%-7zu P50 %7.3f  P95 %7.3f  P99 %7.3f  最大 %7.3f ms\n", label, p.count,
              p.p50, p.p95, p.p99, p.max);
}

// 覆盖层线程：在呈现时机取最新结果画框，不等待未就绪的结果（设计方案 6.2）
struct OverlayStats {
  std::atomic<uint64_t> presents{0};
  std::atomic<uint64_t> freshPresents{0};  // 采用了新结果的呈现
  std::mutex mutex;
  std::vector<int64_t> freshness;  // 呈现返回时，结果距测试源生成的时间
};

void OverlayThread(IDXGIAdapter1* adapter, RECT screenRect, const nr::LatestSlot<Result>& slot,
                   const std::atomic<bool>& running, double staleMs, OverlayStats& stats,
                   std::atomic<bool>& failed) {
  nr::SetCurrentThreadName(L"nr-overlay");
  try {
    nr::Overlay overlay(adapter, screenRect);
    HANDLE waitable = overlay.FrameWaitable();
    uint64_t seen = 0;
    Result current{};
    bool hasResult = false;
    std::vector<nr::ColorRect> rects;

    while (running) {
      const DWORD r = MsgWaitForMultipleObjectsEx(1, &waitable, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
      if (r == WAIT_OBJECT_0 + 1) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) DispatchMessageW(&msg);
        continue;
      }
      if (r != WAIT_OBJECT_0) continue;

      const bool fresh = slot.TakeNewer(seen, current);
      if (fresh) hasResult = true;
      const int64_t now = nr::QpcNow();
      if (hasResult && nr::QpcToMs(now - current.acquireQpc) > staleMs) hasResult = false;

      rects.clear();
      if (hasResult) {
        const RECT& b = current.box;
        nr::AppendOutline(rects, static_cast<float>(b.left - 3), static_cast<float>(b.top - 3),
                          static_cast<float>(b.right - b.left + 6), static_cast<float>(b.bottom - b.top + 6),
                          2.0f, 0.1f, 0.95f, 0.3f, 1.0f);
      }
      overlay.Present(rects);
      const int64_t presented = nr::QpcNow();

      stats.presents++;
      if (fresh) {
        stats.freshPresents++;
        std::lock_guard lock(stats.mutex);
        stats.freshness.push_back(presented - current.sourceRenderQpc);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "覆盖层错误：%s\n", e.what());
    failed = true;
  }
}

std::unique_ptr<nr::CaptureSource> CreateSource(Mode mode, ID3D11Device* device, HWND target,
                                                HMONITOR monitor, IDXGIOutput* output) {
  switch (mode) {
    case Mode::kWgcWindow: return nr::CreateWgcWindowCapture(device, target);
    case Mode::kWgcMonitor: return nr::CreateWgcMonitorCapture(device, monitor);
    case Mode::kDuplication: return nr::CreateDuplicationCapture(device, output);
  }
  return nullptr;
}

int Run(const Options& o) {
  nr::EnableDpiAwareness();
  const bool unthrottled = nr::DisablePowerThrottling();
  nr::SetCurrentThreadName(L"nr-capture");
  winrt::init_apartment(winrt::apartment_type::multi_threaded);

  HWND target = FindWindowW(L"NeuralReflexTestSource", nullptr);
  if (!target) {
    std::fprintf(stderr, "未找到测试源窗口，请先运行 nr_testsource\n");
    return 1;
  }
  HMONITOR monitor = MonitorFromWindow(target, MONITOR_DEFAULTTONEAREST);
  MONITORINFO mi{sizeof(mi)};
  GetMonitorInfoW(monitor, &mi);

  // 采集与呈现使用驱动该显示器的适配器，以 LUID 核对（设计方案第 2 节）
  nr::AdapterInfo adapter = nr::FindAdapterForMonitor(monitor);
  nr::ComPtr<ID3D11DeviceContext> context;
  nr::ComPtr<ID3D11Device> device = nr::CreateDevice(adapter.adapter.Get(), &context);
  std::wprintf(L"适配器：%ls [LUID %ls]\n", adapter.name.c_str(), nr::FormatLuid(adapter.luid).c_str());

  auto source = CreateSource(o.mode, device.Get(), target, monitor, adapter.output.Get());
  std::printf("采集方式：%s，时长 %.0f 秒，覆盖层 %s，电源节流 %s\n", source->Name(), o.duration,
              o.overlay ? "开" : "关", unthrottled ? "已退出" : "退出失败");

  // 标签区域读回用的小块暂存纹理（仅测量工具使用，不属于实时图像路径）
  D3D11_TEXTURE2D_DESC sd{};
  sd.Width = nr::tag::kWidthPx;
  sd.Height = nr::tag::kHeightPx;
  sd.MipLevels = 1;
  sd.ArraySize = 1;
  sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  sd.SampleDesc.Count = 1;
  sd.Usage = D3D11_USAGE_STAGING;
  sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  nr::ComPtr<ID3D11Texture2D> staging;
  nr::Check(device->CreateTexture2D(&sd, nullptr, &staging), "CreateTexture2D(暂存)");

  FILE* csv = nullptr;
  if (o.csvPath) {
    if (_wfopen_s(&csv, o.csvPath, L"w") != 0 || !csv) {
      std::fwprintf(stderr, L"无法写入 CSV：%ls\n", o.csvPath);
      return 1;
    }
    std::fprintf(csv, "frame_id,source_render_qpc,capture_api_qpc,acquire_qpc,decoded_qpc,accumulated\n");
  }

  nr::LatestSlot<Result> slot;
  std::atomic<bool> running{true};
  std::atomic<bool> overlayFailed{false};
  OverlayStats overlayStats;
  std::thread overlayThread;
  if (o.overlay) {
    overlayThread = std::thread(OverlayThread, adapter.adapter.Get(), mi.rcMonitor, std::cref(slot),
                                std::cref(running), o.staleMs, std::ref(overlayStats), std::ref(overlayFailed));
  }

  // 统计
  std::vector<int64_t> captureLatency;   // 测试源生成 → 本程序取得
  std::vector<int64_t> apiLatency;       // 测试源生成 → 采集 API 时间戳
  std::vector<int64_t> processLatency;   // 取得 → 结果发布（含标签读回）
  uint64_t newFrames = 0, decoded = 0, undecodable = 0, outOfFrame = 0;
  uint64_t duplicates = 0, skippedIds = 0, lostEvents = 0;
  uint32_t lastId = 0;

  const int64_t freq = nr::QpcFrequency();
  const int64_t start = nr::QpcNow();
  const int64_t end = start + static_cast<int64_t>(o.duration * freq);
  int64_t secStart = start;
  uint64_t secFrames = 0, secDecoded = 0;
  std::vector<int64_t> secLatency;

  while (nr::QpcNow() < end && !overlayFailed && IsWindow(target)) {
    nr::CapturedFrame frame;
    const nr::AcquireStatus status = source->AcquireLatest(100, frame);
    if (status == nr::AcquireStatus::kLost) {
      ++lostEvents;
      if (!source->Recreate()) {
        std::fprintf(stderr, "采集丢失且无法恢复\n");
        break;
      }
      continue;
    }

    if (status == nr::AcquireStatus::kNewFrame) {
      ++newFrames;
      ++secFrames;

      // 测试源客户区原点 → 采集纹理坐标
      POINT client{0, 0};
      ClientToScreen(target, &client);
      const LONG tagX = client.x + nr::tag::kOriginX - frame.screenOrigin.x;
      const LONG tagY = client.y + nr::tag::kOriginY - frame.screenOrigin.y;
      if (tagX < 0 || tagY < 0 || tagX + nr::tag::kWidthPx > static_cast<LONG>(frame.width) ||
          tagY + nr::tag::kHeightPx > static_cast<LONG>(frame.height)) {
        ++outOfFrame;
      } else {
        D3D11_BOX box{static_cast<UINT>(tagX), static_cast<UINT>(tagY), 0,
                      static_cast<UINT>(tagX + nr::tag::kWidthPx), static_cast<UINT>(tagY + nr::tag::kHeightPx), 1};
        context->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, frame.texture, 0, &box);
        source->ReleaseFrame();
        D3D11_MAPPED_SUBRESOURCE mapped;
        nr::Check(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped), "Map(暂存)");
        const auto payload = nr::tag::Decode(static_cast<const uint8_t*>(mapped.pData), mapped.RowPitch);
        context->Unmap(staging.Get(), 0);
        const int64_t decodedQpc = nr::QpcNow();

        if (!payload) {
          ++undecodable;
        } else {
          ++decoded;
          ++secDecoded;
          if (payload->frameId == lastId) {
            ++duplicates;
          } else if (lastId != 0 && payload->frameId > lastId + 1) {
            skippedIds += payload->frameId - lastId - 1;
          }
          lastId = payload->frameId;

          Result r;
          r.frameId = payload->frameId;
          r.sourceRenderQpc = payload->qpc;
          r.acquireQpc = frame.acquireQpc;
          r.box = {client.x + payload->boxX, client.y + payload->boxY,
                   client.x + payload->boxX + payload->boxW, client.y + payload->boxY + payload->boxH};
          slot.Publish(r);

          captureLatency.push_back(frame.acquireQpc - payload->qpc);
          secLatency.push_back(frame.acquireQpc - payload->qpc);
          if (frame.sourceQpc) apiLatency.push_back(frame.sourceQpc - payload->qpc);
          processLatency.push_back(decodedQpc - frame.acquireQpc);
          if (csv) {
            std::fprintf(csv, "%u,%lld,%lld,%lld,%lld,%u\n", payload->frameId, payload->qpc,
                         frame.sourceQpc, frame.acquireQpc, decodedQpc, frame.accumulated);
          }
        }
      }
    }

    const int64_t now = nr::QpcNow();
    if (now - secStart >= freq) {
      const double secs = static_cast<double>(now - secStart) / freq;
      const nr::Percentiles p = nr::ComputePercentiles(ToMs(secLatency));
      std::printf("[%5.1fs] 新画面 %6.1f/s  解码 %6.1f/s  采集延迟 P50 %6.2f P95 %6.2f ms  覆盖层 %llu 次\n",
                  nr::QpcToMs(now - start) / 1000.0, secFrames / secs, secDecoded / secs, p.p50, p.p95,
                  static_cast<unsigned long long>(overlayStats.presents.load()));
      secStart = now;
      secFrames = secDecoded = 0;
      secLatency.clear();
    }
  }

  running = false;
  if (overlayThread.joinable()) overlayThread.join();
  if (csv) std::fclose(csv);

  const double total = nr::QpcToMs(nr::QpcNow() - start) / 1000.0;
  std::printf("\n==== 汇总（%.1f 秒，%s）====\n", total, source->Name());
  std::printf("  新画面 %llu（%.1f/s），解码成功 %llu（%.1f/s），无法解码 %llu，标签不在画面内 %llu\n",
              static_cast<unsigned long long>(newFrames), newFrames / total,
              static_cast<unsigned long long>(decoded), decoded / total,
              static_cast<unsigned long long>(undecodable), static_cast<unsigned long long>(outOfFrame));
  std::printf("  重复帧 %llu，跳过的测试源帧 %llu，采集丢失 %llu\n", static_cast<unsigned long long>(duplicates),
              static_cast<unsigned long long>(skippedIds), static_cast<unsigned long long>(lostEvents));
  PrintPercentiles("测试源生成→采集 API 时间戳", apiLatency);
  PrintPercentiles("测试源生成→本程序取得", captureLatency);
  PrintPercentiles("取得→结果发布（含读回）", processLatency);
  if (o.overlay) {
    std::lock_guard lock(overlayStats.mutex);
    std::printf("  覆盖层呈现 %llu 次（%.1f/s），其中采用新结果 %llu 次（%.1f/s）\n",
                static_cast<unsigned long long>(overlayStats.presents.load()), overlayStats.presents / total,
                static_cast<unsigned long long>(overlayStats.freshPresents.load()), overlayStats.freshPresents / total);
    PrintPercentiles("测试源生成→覆盖层 Present 返回", overlayStats.freshness);
    std::printf("  注：Present 返回不等于像素上屏（设计方案 7.1）\n");
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
    return Run(o);
  } catch (const winrt::hresult_error& e) {
    std::fwprintf(stderr, L"WinRT 错误 0x%08X：%ls\n", static_cast<unsigned>(e.code()), e.message().c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
  }
  return 1;
}
