#include "common/quad_renderer.h"

#include "common/win_util.h"

#include <d3dcompiler.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace nr {
namespace {

constexpr char kShader[] = R"(
cbuffer Viewport : register(b0) { float2 invHalfSize; float2 pad; };
struct VSIn { float2 pos : POSITION; float4 col : COLOR; };
struct VSOut { float4 pos : SV_Position; float4 col : COLOR; };
VSOut vs_main(VSIn i) {
  VSOut o;
  o.pos = float4(i.pos.x * invHalfSize.x - 1.0, 1.0 - i.pos.y * invHalfSize.y, 0.0, 1.0);
  o.col = i.col;
  return o;
}
float4 ps_main(VSOut i) : SV_Target { return i.col; }
)";

struct Vertex {
  float x, y;
  float r, g, b, a;
};

constexpr size_t kVerticesPerRect = 6;

Microsoft::WRL::ComPtr<ID3DBlob> Compile(const char* entry, const char* target) {
  Microsoft::WRL::ComPtr<ID3DBlob> code, errors;
  HRESULT hr = D3DCompile(kShader, sizeof(kShader) - 1, "quad_renderer", nullptr, nullptr, entry,
                          target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
  if (FAILED(hr)) {
    std::string msg = "着色器编译失败：";
    if (errors) msg.append(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
    throw std::runtime_error(msg);
  }
  return code;
}

}  // namespace

void AppendOutline(std::vector<ColorRect>& out, float x, float y, float w, float h,
                   float thickness, float r, float g, float b, float a) {
  const float t = thickness;
  out.push_back({x, y, w, t, r, g, b, a});
  out.push_back({x, y + h - t, w, t, r, g, b, a});
  out.push_back({x, y + t, t, h - 2 * t, r, g, b, a});
  out.push_back({x + w - t, y + t, t, h - 2 * t, r, g, b, a});
}

void QuadRenderer::Init(ID3D11Device* device, size_t maxRects) {
  maxRects_ = maxRects;

  auto vsCode = Compile("vs_main", "vs_5_0");
  auto psCode = Compile("ps_main", "ps_5_0");
  Check(device->CreateVertexShader(vsCode->GetBufferPointer(), vsCode->GetBufferSize(), nullptr, &vs_),
        "CreateVertexShader");
  Check(device->CreatePixelShader(psCode->GetBufferPointer(), psCode->GetBufferSize(), nullptr, &ps_),
        "CreatePixelShader");

  const D3D11_INPUT_ELEMENT_DESC elements[] = {
      {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
      {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0},
  };
  Check(device->CreateInputLayout(elements, ARRAYSIZE(elements), vsCode->GetBufferPointer(),
                                  vsCode->GetBufferSize(), &layout_),
        "CreateInputLayout");

  D3D11_BUFFER_DESC vb{};
  vb.ByteWidth = static_cast<UINT>(maxRects * kVerticesPerRect * sizeof(Vertex));
  vb.Usage = D3D11_USAGE_DYNAMIC;
  vb.BindFlags = D3D11_BIND_VERTEX_BUFFER;
  vb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
  Check(device->CreateBuffer(&vb, nullptr, &vertices_), "CreateBuffer(vertices)");

  D3D11_BUFFER_DESC cb{};
  cb.ByteWidth = 16;
  cb.Usage = D3D11_USAGE_DYNAMIC;
  cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  cb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
  Check(device->CreateBuffer(&cb, nullptr, &constants_), "CreateBuffer(constants)");

  D3D11_BLEND_DESC bd{};
  auto& rt = bd.RenderTarget[0];
  rt.BlendEnable = TRUE;
  rt.SrcBlend = D3D11_BLEND_ONE;  // 预乘 alpha
  rt.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
  rt.BlendOp = D3D11_BLEND_OP_ADD;
  rt.SrcBlendAlpha = D3D11_BLEND_ONE;
  rt.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
  rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
  rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
  Check(device->CreateBlendState(&bd, &blend_), "CreateBlendState");

  D3D11_RASTERIZER_DESC rd{};
  rd.FillMode = D3D11_FILL_SOLID;
  rd.CullMode = D3D11_CULL_NONE;
  rd.DepthClipEnable = TRUE;
  Check(device->CreateRasterizerState(&rd, &raster_), "CreateRasterizerState");
}

void QuadRenderer::Draw(ID3D11DeviceContext* context, float viewportW, float viewportH,
                        const ColorRect* rects, size_t count) {
  count = std::min(count, maxRects_);
  if (count == 0) return;

  D3D11_MAPPED_SUBRESOURCE mapped;
  Check(context->Map(vertices_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped), "Map(vertices)");
  auto* v = static_cast<Vertex*>(mapped.pData);
  for (size_t i = 0; i < count; ++i) {
    const ColorRect& q = rects[i];
    const float x0 = q.x, y0 = q.y, x1 = q.x + q.w, y1 = q.y + q.h;
    const Vertex c[4] = {{x0, y0, q.r, q.g, q.b, q.a},
                         {x1, y0, q.r, q.g, q.b, q.a},
                         {x0, y1, q.r, q.g, q.b, q.a},
                         {x1, y1, q.r, q.g, q.b, q.a}};
    *v++ = c[0]; *v++ = c[1]; *v++ = c[2];
    *v++ = c[2]; *v++ = c[1]; *v++ = c[3];
  }
  context->Unmap(vertices_.Get(), 0);

  Check(context->Map(constants_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped), "Map(constants)");
  const float cbData[4] = {2.0f / viewportW, 2.0f / viewportH, 0, 0};
  std::memcpy(mapped.pData, cbData, sizeof(cbData));
  context->Unmap(constants_.Get(), 0);

  D3D11_VIEWPORT vp{0, 0, viewportW, viewportH, 0, 1};
  const UINT stride = sizeof(Vertex), offset = 0;
  context->RSSetViewports(1, &vp);
  context->RSSetState(raster_.Get());
  context->OMSetBlendState(blend_.Get(), nullptr, 0xFFFFFFFF);
  context->IASetInputLayout(layout_.Get());
  context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->IASetVertexBuffers(0, 1, vertices_.GetAddressOf(), &stride, &offset);
  context->VSSetShader(vs_.Get(), nullptr, 0);
  context->VSSetConstantBuffers(0, 1, constants_.GetAddressOf());
  context->PSSetShader(ps_.Get(), nullptr, 0);
  context->Draw(static_cast<UINT>(count * kVerticesPerRect), 0);
}

}  // namespace nr
