// 并行负载模拟（设计方案 7.5）：在同一块 GPU 上不限帧率地渲染重着色器，并常驻指定大小的显存，
// 代表三维建模与渲染类软件。每秒输出一行帧率，供测试脚本统计检测开启前后的变化。
// 这是可重复的模拟负载，不等同于任何具体软件。

#include "common/d3d_util.h"
#include "common/qpc.h"
#include "common/win_util.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  double vramGb = 0;      // 常驻显存
  int width = 1920, height = 1080;
  int iterations = 256;   // 每像素着色器循环次数，决定负载强度
  double duration = 60;   // 秒
  bool info = false;      // 只输出 GPU 与调度信息
  // 着色器从常驻显存的第一块采样：带宽压力大，单帧 GPU 时间成倍变长。
  // 默认采样小纹理，此时常驻显存只占容量，便于把容量压力与长时间占用 GPU 分开测量。
  bool sampleResident = false;
  // 每帧拆成的 draw 数（按水平条带划分），总计算量不变。用于区分单次 GPU 工作的长度
  // 与总负载对其他进程的影响（设计方案 6.4）。
  int draws = 1;
};

constexpr char kShader[] = R"(
Texture2D<float4> big : register(t0);
SamplerState samp : register(s0);
cbuffer C : register(b0) { float time; uint iterations; float2 pad; };
float4 vs_main(uint id : SV_VertexID) : SV_Position {
  float2 uv = float2((id << 1) & 2, id & 2);
  return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}
float4 ps_main(float4 pos : SV_Position) : SV_Target {
  float2 p = pos.xy * 0.001 + time;
  float3 acc = 0;
  [loop] for (uint i = 0; i < iterations; ++i) {
    p = float2(sin(p.x * 1.7 + p.y), cos(p.y * 1.3 - p.x)) + 0.01 * i;
    // 相邻像素读取相邻纹素，带宽压力真实但不出现随机访问的病态情形
    acc += big.SampleLevel(samp, frac(pos.xy / 4096.0 + 0.013 * i), 0).rgb * 0.001 + abs(p.xyx) * 0.0005;
  }
  return float4(frac(acc), 1);
}
)";

Microsoft::WRL::ComPtr<ID3DBlob> Compile(const char* entry, const char* target) {
  Microsoft::WRL::ComPtr<ID3DBlob> code, errors;
  if (FAILED(D3DCompile(kShader, sizeof(kShader) - 1, "gpuload", nullptr, nullptr, entry, target,
                        D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors))) {
    throw std::runtime_error(errors ? static_cast<const char*>(errors->GetBufferPointer()) : "着色器编译失败");
  }
  return code;
}

bool Parse(int argc, wchar_t** argv, Options& o) {
  for (int i = 1; i < argc; ++i) {
    const std::wstring a = argv[i];
    const wchar_t* v = i + 1 < argc ? argv[i + 1] : nullptr;
    if (a == L"--info") o.info = true;
    else if (a == L"--vram-gb" && v) o.vramGb = std::wcstod(argv[++i], nullptr);
    else if (a == L"--sample-resident") o.sampleResident = true;
    else if (a == L"--draws" && v) o.draws = std::max(1, _wtoi(argv[++i]));
    else if (a == L"--iterations" && v) o.iterations = _wtoi(argv[++i]);
    else if (a == L"--duration" && v) o.duration = std::wcstod(argv[++i], nullptr);
    else if (a == L"--size" && v) { if (swscanf_s(argv[++i], L"%dx%d", &o.width, &o.height) != 2) return false; }
    else return false;
  }
  return true;
}

int Run(const Options& o) {
  nr::AdapterInfo adapter = nr::FindAdapterForMonitor(MonitorFromPoint(POINT{0, 0}, MONITOR_DEFAULTTOPRIMARY));
  if (o.info) {
    const nr::HagsState h = nr::QueryHags(adapter.luid);
    std::printf("{\"adapter_luid\":\"%ls\",\"hags_queried\":%s,\"hags_supported\":%s,\"hags_enabled\":%s,\"hags_default\":%s}\n",
                nr::FormatLuid(adapter.luid).c_str(), h.queried ? "true" : "false", h.supported ? "true" : "false",
                h.enabled ? "true" : "false", h.enabledByDefault ? "true" : "false");
    return 0;
  }
  nr::ComPtr<ID3D11DeviceContext> context;
  auto device = nr::CreateDevice(adapter.adapter.Get(), &context);

  // 常驻显存：每块 256 MB（4096 x 4096 RGBA32F），全部清零一次使其真正分配，并在着色器中采样第一块
  std::vector<nr::ComPtr<ID3D11Texture2D>> blocks;
  const int count = static_cast<int>(o.vramGb * 4 + 0.5);
  for (int i = 0; i < count; ++i) {
    D3D11_TEXTURE2D_DESC td{};
    td.Width = td.Height = 4096;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    nr::ComPtr<ID3D11Texture2D> t;
    if (FAILED(device->CreateTexture2D(&td, nullptr, &t))) {
      std::fprintf(stderr, "显存分配在第 %d 块（%.2f GB）失败\n", i, i / 4.0);
      break;
    }
    nr::ComPtr<ID3D11RenderTargetView> rtv;
    nr::Check(device->CreateRenderTargetView(t.Get(), nullptr, &rtv), "CreateRenderTargetView");
    const float c[4] = {0.1f * (i % 7), 0.2f, 0.3f, 1};
    context->ClearRenderTargetView(rtv.Get(), c);
    blocks.push_back(t);
  }
  context->Flush();

  D3D11_TEXTURE2D_DESC rd{};
  rd.Width = o.width;
  rd.Height = o.height;
  rd.MipLevels = rd.ArraySize = 1;
  rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  rd.SampleDesc.Count = 1;
  rd.BindFlags = D3D11_BIND_RENDER_TARGET;
  nr::ComPtr<ID3D11Texture2D> target;
  nr::Check(device->CreateTexture2D(&rd, nullptr, &target), "CreateTexture2D(渲染目标)");
  nr::ComPtr<ID3D11RenderTargetView> rtv;
  nr::Check(device->CreateRenderTargetView(target.Get(), nullptr, &rtv), "CreateRenderTargetView");

  const bool sampleResident = o.sampleResident && !blocks.empty();
  nr::ComPtr<ID3D11Texture2D> sampled = sampleResident ? blocks.front() : target;
  if (!sampleResident) {
    D3D11_TEXTURE2D_DESC sd = rd;
    sd.Width = sd.Height = 256;
    sd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    nr::Check(device->CreateTexture2D(&sd, nullptr, &sampled), "CreateTexture2D(采样)");
  }
  nr::ComPtr<ID3D11ShaderResourceView> srv;
  nr::Check(device->CreateShaderResourceView(sampled.Get(), nullptr, &srv), "CreateShaderResourceView");

  auto vsCode = Compile("vs_main", "vs_5_0");
  auto psCode = Compile("ps_main", "ps_5_0");
  nr::ComPtr<ID3D11VertexShader> vs;
  nr::ComPtr<ID3D11PixelShader> ps;
  device->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &vs);
  device->CreatePixelShader(psCode->GetBufferPointer(), psCode->GetBufferSize(), nullptr, &ps);
  D3D11_SAMPLER_DESC sdesc{};
  sdesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
  sdesc.AddressU = sdesc.AddressV = sdesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
  nr::ComPtr<ID3D11SamplerState> sampler;
  device->CreateSamplerState(&sdesc, &sampler);
  D3D11_BUFFER_DESC cbd{16, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0};
  nr::ComPtr<ID3D11Buffer> cb;
  device->CreateBuffer(&cbd, nullptr, &cb);

  // 最多两帧在途，与普通渲染程序的排队深度相当
  D3D11_QUERY_DESC qd{D3D11_QUERY_EVENT, 0};
  nr::ComPtr<ID3D11Query> fences[2];
  for (auto& f : fences) device->CreateQuery(&qd, &f);

  // 每个条带一个视口；SV_Position 为渲染目标坐标，着色结果与单次 draw 相同
  std::vector<D3D11_VIEWPORT> strips;
  for (int i = 0; i < o.draws; ++i) {
    const float y0 = std::floor(static_cast<float>(o.height) * i / o.draws);
    const float y1 = std::floor(static_cast<float>(o.height) * (i + 1) / o.draws);
    if (y1 > y0) strips.push_back({0, y0, static_cast<float>(o.width), y1 - y0, 0, 1});
  }

  DXGI_QUERY_VIDEO_MEMORY_INFO vram{};
  nr::ComPtr<IDXGIAdapter3> a3;
  if (SUCCEEDED(adapter.adapter.As(&a3))) a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vram);
  std::printf("{\"event\":\"start\",\"vram_gb\":%.2f,\"process_vram_mb\":%.0f,\"budget_mb\":%.0f,\"iterations\":%d,\"sample_resident\":%s,\"draws\":%zu}\n",
              blocks.size() / 4.0, vram.CurrentUsage / 1048576.0, vram.Budget / 1048576.0, o.iterations,
              sampleResident ? "true" : "false", strips.size());
  std::fflush(stdout);

  const int64_t freq = nr::QpcFrequency(), start = nr::QpcNow();
  int64_t secStart = start;
  uint64_t frames = 0, secFrames = 0;
  while (nr::QpcToMs(nr::QpcNow() - start) < o.duration * 1000) {
    ID3D11Query* fence = fences[frames % 2].Get();
    if (frames >= 2) while (context->GetData(fence, nullptr, 0, 0) == S_FALSE) SwitchToThread();

    D3D11_MAPPED_SUBRESOURCE m;
    context->Map(cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m);
    struct { float time; uint32_t iterations; float pad[2]; } c{static_cast<float>(frames) * 0.01f, static_cast<uint32_t>(o.iterations), {}};
    std::memcpy(m.pData, &c, sizeof(c));
    context->Unmap(cb.Get(), 0);

    context->OMSetRenderTargets(1, rtv.GetAddressOf(), nullptr);
    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->VSSetShader(vs.Get(), nullptr, 0);
    context->PSSetShader(ps.Get(), nullptr, 0);
    context->PSSetShaderResources(0, 1, srv.GetAddressOf());
    context->PSSetSamplers(0, 1, sampler.GetAddressOf());
    context->PSSetConstantBuffers(0, 1, cb.GetAddressOf());
    for (const D3D11_VIEWPORT& vp : strips) {
      context->RSSetViewports(1, &vp);
      context->Draw(3, 0);
    }
    context->End(fence);
    context->Flush();
    ++frames;
    ++secFrames;

    const int64_t now = nr::QpcNow();
    if (now - secStart >= freq) {
      const auto unixMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch()).count();
      std::printf("{\"event\":\"second\",\"unix_ms\":%lld,\"fps\":%.2f}\n", static_cast<long long>(unixMs),
                  secFrames * static_cast<double>(freq) / (now - secStart));
      std::fflush(stdout);
      secFrames = 0;
      secStart = now;
    }
  }
  return 0;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  SetConsoleOutputCP(CP_UTF8);
  Options o;
  if (!Parse(argc, argv, o)) {
    std::fwprintf(stderr, L"用法：nr_gpuload [--vram-gb N] [--sample-resident] [--draws N] [--iterations N] [--size WxH] [--duration 秒]\n");
    return 2;
  }
  try {
    return Run(o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "错误：%s\n", e.what());
    return 1;
  }
}
