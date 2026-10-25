#include "overlay/overlay.h"

#include "common/d3d_util.h"
#include "common/win_util.h"

namespace nr {
namespace {

constexpr wchar_t kClassName[] = L"NeuralReflexOverlay";

// 覆盖层的生命周期只由会话控制：外部发来的关闭请求（例如 taskkill）一律忽略
LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
  if (msg == WM_CLOSE) return 0;
  return DefWindowProcW(hwnd, msg, wp, lp);
}

void RegisterOverlayClass() {
  static const bool registered = [] {
    WNDCLASSEXW wc{sizeof(wc)};
    wc.lpfnWndProc = OverlayProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = kClassName;
    return RegisterClassExW(&wc) != 0;
  }();
  (void)registered;
}

}  // namespace

Overlay::Overlay(IDXGIAdapter1* adapter, const RECT& screenRect, size_t maxRects)
    : screenRect_(screenRect) {
  RegisterOverlayClass();
  const int w = screenRect.right - screenRect.left;
  const int h = screenRect.bottom - screenRect.top;

  // 无重定向位图 + 分层 + 透明：内容完全由 DirectComposition 提供，鼠标点击穿透
  hwnd_ = CreateWindowExW(WS_EX_NOREDIRECTIONBITMAP | WS_EX_TOPMOST | WS_EX_LAYERED |
                              WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                          kClassName, L"", WS_POPUP, screenRect.left, screenRect.top, w, h,
                          nullptr, nullptr, GetModuleHandleW(nullptr), nullptr);
  if (!hwnd_) Fail("CreateWindowExW(覆盖层)", HRESULT_FROM_WIN32(GetLastError()));
  SetLayeredWindowAttributes(hwnd_, 0, 255, LWA_ALPHA);

  device_ = CreateDevice(adapter, &context_);

  Microsoft::WRL::ComPtr<IDXGIDevice> dxgiDevice;
  Check(device_.As(&dxgiDevice), "ID3D11Device -> IDXGIDevice");
  Microsoft::WRL::ComPtr<IDXGIFactory2> factory;
  Check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");

  DXGI_SWAP_CHAIN_DESC1 desc{};
  desc.Width = static_cast<UINT>(w);
  desc.Height = static_cast<UINT>(h);
  desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
  desc.SampleDesc.Count = 1;
  desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
  desc.BufferCount = 2;
  desc.Scaling = DXGI_SCALING_STRETCH;
  desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
  desc.AlphaMode = DXGI_ALPHA_MODE_PREMULTIPLIED;
  desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;

  Microsoft::WRL::ComPtr<IDXGISwapChain1> swapChain1;
  Check(factory->CreateSwapChainForComposition(device_.Get(), &desc, nullptr, &swapChain1),
        "CreateSwapChainForComposition");
  Check(swapChain1.As(&swapChain_), "IDXGISwapChain1 -> IDXGISwapChain2");
  Check(swapChain_->SetMaximumFrameLatency(1), "SetMaximumFrameLatency");
  waitable_ = swapChain_->GetFrameLatencyWaitableObject();

  // D3D11 翻转模型下缓冲 0 的视图始终指向当前后台缓冲
  Microsoft::WRL::ComPtr<ID3D11Texture2D> backBuffer;
  Check(swapChain_->GetBuffer(0, IID_PPV_ARGS(&backBuffer)), "GetBuffer");
  Check(device_->CreateRenderTargetView(backBuffer.Get(), nullptr, &rtv_), "CreateRenderTargetView");

  Check(DCompositionCreateDevice(dxgiDevice.Get(), IID_PPV_ARGS(&dcomp_)), "DCompositionCreateDevice");
  Check(dcomp_->CreateTargetForHwnd(hwnd_, TRUE, &target_), "CreateTargetForHwnd");
  Check(dcomp_->CreateVisual(&visual_), "CreateVisual");
  Check(visual_->SetContent(swapChain_.Get()), "SetContent");
  Check(target_->SetRoot(visual_.Get()), "SetRoot");
  Check(dcomp_->Commit(), "Commit");

  renderer_.Init(device_.Get(), maxRects);
  local_.reserve(maxRects);
  ShowWindow(hwnd_, SW_SHOWNOACTIVATE);
}

Overlay::~Overlay() {
  if (waitable_) CloseHandle(waitable_);
  if (hwnd_) DestroyWindow(hwnd_);
}

void Overlay::Present(const std::vector<ColorRect>& screenRects) {
  local_.clear();
  for (const ColorRect& r : screenRects) {
    ColorRect l = r;
    l.x -= static_cast<float>(screenRect_.left);
    l.y -= static_cast<float>(screenRect_.top);
    local_.push_back(l);
  }

  const float clear[4] = {0, 0, 0, 0};
  context_->OMSetRenderTargets(1, rtv_.GetAddressOf(), nullptr);
  context_->ClearRenderTargetView(rtv_.Get(), clear);
  renderer_.Draw(context_.Get(), static_cast<float>(screenRect_.right - screenRect_.left),
                 static_cast<float>(screenRect_.bottom - screenRect_.top), local_.data(), local_.size());
  Check(swapChain_->Present(1, 0), "Present(覆盖层)");
}

}  // namespace nr
