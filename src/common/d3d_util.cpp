#include "common/d3d_util.h"

#include "common/win_util.h"

#include <cstdio>
#include <stdexcept>

namespace nr {

AdapterInfo FindAdapterForMonitor(HMONITOR monitor) {
  ComPtr<IDXGIFactory1> factory;
  Check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");

  ComPtr<IDXGIAdapter1> adapter;
  for (UINT a = 0; factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
    ComPtr<IDXGIOutput> output;
    for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
      DXGI_OUTPUT_DESC desc{};
      if (SUCCEEDED(output->GetDesc(&desc)) && desc.Monitor == monitor) {
        DXGI_ADAPTER_DESC1 ad{};
        adapter->GetDesc1(&ad);
        return AdapterInfo{adapter, output, ad.Description, ad.AdapterLuid};
      }
    }
  }
  throw std::runtime_error("未找到驱动目标显示器的 DXGI 适配器");
}

ComPtr<ID3D11Device> CreateDevice(IDXGIAdapter1* adapter, ComPtr<ID3D11DeviceContext>* context) {
  const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
  ComPtr<ID3D11Device> device;
  Check(D3D11CreateDevice(adapter, adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
                          nullptr, flags, levels, ARRAYSIZE(levels), D3D11_SDK_VERSION, &device,
                          nullptr, context ? context->ReleaseAndGetAddressOf() : nullptr),
        "D3D11CreateDevice");
  return device;
}

std::wstring FormatLuid(const LUID& luid) {
  wchar_t buf[32];
  std::swprintf(buf, 32, L"%08X:%08X", static_cast<unsigned>(luid.HighPart), luid.LowPart);
  return buf;
}

}  // namespace nr
