#pragma once

#include <d3d11.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <string>

namespace nr {

using Microsoft::WRL::ComPtr;

struct AdapterInfo {
  ComPtr<IDXGIAdapter1> adapter;
  ComPtr<IDXGIOutput> output;  // 与显示器对应的输出
  std::wstring name;
  LUID luid{};
};

// 找到实际驱动该显示器的适配器，采集、计算和呈现都应使用它（设计方案第 2 节）。
AdapterInfo FindAdapterForMonitor(HMONITOR monitor);

ComPtr<ID3D11Device> CreateDevice(IDXGIAdapter1* adapter, ComPtr<ID3D11DeviceContext>* context);

std::wstring FormatLuid(const LUID& luid);

}  // namespace nr
