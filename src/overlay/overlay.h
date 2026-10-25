#pragma once

#include "common/quad_renderer.h"

#include <d3d11.h>
#include <dcomp.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <vector>

namespace nr {

// 透明、置顶、鼠标穿透的覆盖层：Direct3D 绘制，DirectComposition 合成，
// 可等待交换链且最大帧延迟为 1（设计方案 5.2）。
// 必须在创建它的线程上调用全部方法，并由该线程处理窗口消息。
class Overlay {
 public:
  Overlay(IDXGIAdapter1* adapter, const RECT& screenRect, size_t maxRects = 1024);
  ~Overlay();
  Overlay(const Overlay&) = delete;
  Overlay& operator=(const Overlay&) = delete;

  // 呈现时机到来时变为有信号，等待它之后再取最新结果并绘制。
  HANDLE FrameWaitable() const { return waitable_; }

  // rects 为屏幕坐标、预乘 alpha。空列表即清除所有框。
  void Present(const std::vector<ColorRect>& screenRects);

 private:
  HWND hwnd_ = nullptr;
  RECT screenRect_{};
  Microsoft::WRL::ComPtr<ID3D11Device> device_;
  Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
  Microsoft::WRL::ComPtr<IDXGISwapChain2> swapChain_;
  Microsoft::WRL::ComPtr<ID3D11RenderTargetView> rtv_;
  Microsoft::WRL::ComPtr<IDCompositionDevice> dcomp_;
  Microsoft::WRL::ComPtr<IDCompositionTarget> target_;
  Microsoft::WRL::ComPtr<IDCompositionVisual> visual_;
  HANDLE waitable_ = nullptr;
  QuadRenderer renderer_;
  std::vector<ColorRect> local_;
};

}  // namespace nr
