// 合成测试源（设计方案 7.4）：按设定速率呈现一个移动目标，并在左上角
// 编码帧编号、生成时刻和目标框，供采集端做帧对应、延迟和坐标验证。

#include "common/d3d_util.h"
#include "common/frame_tag.h"
#include "common/qpc.h"
#include "common/quad_renderer.h"
#include "common/win_util.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <exception>
#include <vector>

namespace {

constexpr wchar_t kClassName[] = L"NeuralReflexTestSource";

struct Options {
  int width = 1280;
  int height = 720;
  double fps = 0;          // 0 表示跟随垂直同步
  double duration = 0;     // 秒，0 表示直到关闭窗口
  const wchar_t* logPath = nullptr;
};

void PrintUsage() {
  std::fwprintf(stderr,
                L"用法：nr_testsource [--fps N] [--size WxH] [--duration 秒] [--log 路径.csv]\n"
                L"  --fps 0（默认）跟随显示器垂直同步；N>0 时按 N FPS 定时呈现并允许撕裂\n");
}

bool ParseOptions(int argc, wchar_t** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() -> const wchar_t* { return i + 1 < argc ? argv[++i] : nullptr; };
    const wchar_t* a = argv[i];
    const wchar_t* v = nullptr;
    if (!std::wcscmp(a, L"--fps") && (v = next())) {
      o.fps = std::wcstod(v, nullptr);
    } else if (!std::wcscmp(a, L"--size") && (v = next())) {
      if (swscanf_s(v, L"%dx%d", &o.width, &o.height) != 2) return false;
    } else if (!std::wcscmp(a, L"--duration") && (v = next())) {
      o.duration = std::wcstod(v, nullptr);
    } else if (!std::wcscmp(a, L"--log") && (v = next())) {
      o.logPath = v;
    } else {
      return false;
    }
  }
  return o.width >= nr::tag::kOriginX + nr::tag::kWidthPx + 8 && o.height >= 240;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  if (msg == WM_DESTROY) {
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProcW(hwnd, msg, wp, lp);
}

HWND CreateSourceWindow(int clientW, int clientH) {
  WNDCLASSEXW wc{sizeof(wc)};
  wc.lpfnWndProc = WndProc;
  wc.hInstance = GetModuleHandleW(nullptr);
  wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
  wc.lpszClassName = kClassName;
  RegisterClassExW(&wc);

  const DWORD style = WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
  RECT r{0, 0, clientW, clientH};
  AdjustWindowRectExForDpi(&r, style, FALSE, 0, GetDpiForSystem());
  HWND hwnd = CreateWindowExW(0, kClassName, L"NR Test Source", style, 100, 100, r.right - r.left,
                              r.bottom - r.top, nullptr, nullptr, wc.hInstance, nullptr);
  if (!hwnd) nr::Fail("CreateWindowExW(测试源)", HRESULT_FROM_WIN32(GetLastError()));

  // 窗口实际所在显示器的 DPI 可能不同，按实际 DPI 校正客户区为物理像素尺寸
  RECT client{};
  GetClientRect(hwnd, &client);
  if (client.right != clientW || client.bottom != clientH) {
    RECT fix{0, 0, clientW, clientH};
    AdjustWindowRectExForDpi(&fix, style, FALSE, 0, GetDpiForWindow(hwnd));
    SetWindowPos(hwnd, nullptr, 0, 0, fix.right - fix.left, fix.bottom - fix.top,
                 SWP_NOMOVE | SWP_NOZORDER);
  }
  ShowWindow(hwnd, SW_SHOW);
  return hwnd;
}

bool TearingSupported() {
  nr::ComPtr<IDXGIFactory5> factory;
  BOOL allow = FALSE;
  if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
    factory->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allow, sizeof(allow));
  }
  return allow != FALSE;
}

// 目标在客户区内往返移动，避开左上角标签区域
void TargetAt(double t, const Options& o, float& x, float& y, float& w, float& h) {
  w = 120;
  h = 90;
  const float top = static_cast<float>(nr::tag::kOriginY + nr::tag::kHeightPx + 24);
  const float spanX = static_cast<float>(o.width) - w;
  const float spanY = static_cast<float>(o.height) - h - top;
  auto bounce = [](double v) { double f = std::fmod(v, 2.0); return f < 1.0 ? f : 2.0 - f; };
  x = static_cast<float>(bounce(t * 0.35) * spanX);
  y = top + static_cast<float>(bounce(t * 0.23) * spanY);
}

int Run(const Options& o) {
  nr::EnableDpiAwareness();
  nr::DisablePowerThrottling();
  HWND hwnd = CreateSourceWindow(o.width, o.height);

  RECT client{};
  GetClientRect(hwnd, &client);
  const UINT w = static_cast<UINT>(client.right), h = static_cast<UINT>(client.bottom);

  nr::AdapterInfo adapter = nr::FindAdapterForMonitor(MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST));
  nr::ComPtr<ID3D11DeviceContext> context;
  nr::ComPtr<ID3D11Device> device = nr::CreateDevice(adapter.adapter.Get(), &context);

  const bool paced = o.fps > 0;
  const bool tearing = paced && TearingSupported();
  nr::ComPtr<IDXGIFactory2> factory;
  nr::Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
  DXGI_SWAP_CHAIN_DESC1 desc{};
  desc.Width = w;
  desc.Height = h;
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.BufferCount = 3;
  desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
  desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
               (tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
  nr::ComPtr<IDXGISwapChain1> sc1;
  nr::Check(factory->CreateSwapChainForHwnd(device.Get(), hwnd, &desc, nullptr, nullptr, &sc1),
            "CreateSwapChainForHwnd");
  factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
  nr::ComPtr<IDXGISwapChain2> swapChain;
  nr::Check(sc1.As(&swapChain), "IDXGISwapChain2");
  // 帧生成时刻尽量贴近实际呈现，避免测试源自身排队污染延迟测量
  swapChain->SetMaximumFrameLatency(1);
  HANDLE frameWaitable = swapChain->GetFrameLatencyWaitableObject();

  nr::ComPtr<ID3D11Texture2D> backBuffer;
  nr::Check(swapChain->GetBuffer(0, IID_PPV_ARGS(&backBuffer)), "GetBuffer");
  nr::ComPtr<ID3D11RenderTargetView> rtv;
  nr::Check(device->CreateRenderTargetView(backBuffer.Get(), nullptr, &rtv), "CreateRenderTargetView");

  nr::QuadRenderer renderer;
  renderer.Init(device.Get(), nr::tag::kBits + 8);
  std::vector<nr::ColorRect> rects;
  rects.reserve(nr::tag::kBits + 8);

  FILE* log = nullptr;
  if (o.logPath) {
    if (_wfopen_s(&log, o.logPath, L"w") != 0 || !log) {
      std::fwprintf(stderr, L"无法写入日志：%ls\n", o.logPath);
      return 1;
    }
    std::fprintf(log, "frame_id,render_qpc,present_return_qpc,box_x,box_y,box_w,box_h\n");
  }

  HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                        TIMER_ALL_ACCESS);
  const int64_t freq = nr::QpcFrequency();
  const int64_t period = paced ? static_cast<int64_t>(freq / o.fps) : 0;
  const int64_t start = nr::QpcNow();
  int64_t nextFrame = start;
  int64_t statStart = start;
  uint32_t statFrames = 0;
  uint32_t frameId = 0;

  std::wprintf(L"测试源：客户区 %ux%u，%ls，适配器 %ls [%ls]\n", w, h,
               paced ? L"定时呈现" : L"垂直同步", adapter.name.c_str(), nr::FormatLuid(adapter.luid).c_str());

  for (;;) {
    MSG msg;
    bool quit = false;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT) quit = true;
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
    if (quit) break;
    if (o.duration > 0 && nr::QpcToMs(nr::QpcNow() - start) >= o.duration * 1000) break;

    WaitForSingleObjectEx(frameWaitable, 1000, TRUE);
    if (paced) {
      const int64_t now = nr::QpcNow();
      if (nextFrame - now > 0) {
        LARGE_INTEGER due;
        due.QuadPart = -static_cast<LONGLONG>((nextFrame - now) * 10'000'000 / freq);
        SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0);
        WaitForSingleObject(timer, INFINITE);
      } else if (now - nextFrame > period) {
        nextFrame = now;  // 落后超过一帧时重新对齐，不补帧
      }
      nextFrame += period;
    }

    const int64_t renderQpc = nr::QpcNow();
    float tx, ty, tw, th;
    TargetAt(nr::QpcToMs(renderQpc - start) / 1000.0, o, tx, ty, tw, th);

    nr::tag::Payload payload;
    payload.frameId = ++frameId;
    payload.qpc = renderQpc;
    payload.boxX = static_cast<uint16_t>(tx);
    payload.boxY = static_cast<uint16_t>(ty);
    payload.boxW = static_cast<uint16_t>(tw);
    payload.boxH = static_cast<uint16_t>(th);
    const auto bits = nr::tag::Encode(payload);

    rects.clear();
    rects.push_back({tx, ty, tw, th, 0.95f, 0.30f, 0.15f, 1});
    constexpr float kCell = nr::tag::kCellPx;
    rects.push_back({nr::tag::kOriginX - 4.0f, nr::tag::kOriginY - 4.0f, nr::tag::kWidthPx + 8.0f,
                     nr::tag::kHeightPx + 8.0f, 0, 0, 0, 1});
    for (int i = 0; i < nr::tag::kBits; ++i) {
      if (!bits[i]) continue;
      rects.push_back({nr::tag::kOriginX + (i % nr::tag::kCols) * kCell,
                       nr::tag::kOriginY + (i / nr::tag::kCols) * kCell, kCell, kCell, 1, 1, 1, 1});
    }

    const float background[4] = {0.18f, 0.20f, 0.24f, 1};
    context->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
    context->ClearRenderTargetView(rtv.Get(), background);
    renderer.Draw(context.Get(), static_cast<float>(w), static_cast<float>(h), rects.data(), rects.size());
    nr::Check(swapChain->Present(paced ? 0 : 1, tearing ? DXGI_PRESENT_ALLOW_TEARING : 0), "Present");
    const int64_t presentReturnQpc = nr::QpcNow();

    if (log) {
      std::fprintf(log, "%u,%lld,%lld,%u,%u,%u,%u\n", payload.frameId, renderQpc, presentReturnQpc,
                   payload.boxX, payload.boxY, payload.boxW, payload.boxH);
    }

    ++statFrames;
    const int64_t statElapsed = presentReturnQpc - statStart;
    if (statElapsed >= freq) {
      wchar_t title[128];
      std::swprintf(title, 128, L"NR Test Source - %.1f FPS - 帧 %u",
                    statFrames * static_cast<double>(freq) / statElapsed, frameId);
      SetWindowTextW(hwnd, title);
      statFrames = 0;
      statStart = presentReturnQpc;
    }
  }

  if (log) std::fclose(log);
  CloseHandle(timer);
  CloseHandle(frameWaitable);
  std::wprintf(L"测试源结束：共 %u 帧\n", frameId);
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
    return Run(o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
