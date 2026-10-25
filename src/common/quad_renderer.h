#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include <cstddef>
#include <vector>

namespace nr {

// 像素坐标下的纯色矩形，颜色为预乘 alpha。
struct ColorRect {
  float x, y, w, h;
  float r, g, b, a;
};

// 用 4 个矩形组成空心框。
void AppendOutline(std::vector<ColorRect>& out, float x, float y, float w, float h,
                   float thickness, float r, float g, float b, float a);

// 批量绘制纯色矩形：一次 Map 写顶点、一次 Draw。容量在 Init 时固定。
class QuadRenderer {
 public:
  void Init(ID3D11Device* device, size_t maxRects);
  void Draw(ID3D11DeviceContext* context, float viewportW, float viewportH,
            const ColorRect* rects, size_t count);

 private:
  Microsoft::WRL::ComPtr<ID3D11VertexShader> vs_;
  Microsoft::WRL::ComPtr<ID3D11PixelShader> ps_;
  Microsoft::WRL::ComPtr<ID3D11InputLayout> layout_;
  Microsoft::WRL::ComPtr<ID3D11Buffer> vertices_;
  Microsoft::WRL::ComPtr<ID3D11Buffer> constants_;
  Microsoft::WRL::ComPtr<ID3D11BlendState> blend_;
  Microsoft::WRL::ComPtr<ID3D11RasterizerState> raster_;
  size_t maxRects_ = 0;
};

}  // namespace nr
