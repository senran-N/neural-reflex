#pragma once

#include <d3d11.h>
#include <wrl/client.h>

#include <cstdint>
#include <filesystem>
#include <vector>

namespace nr {

struct BgraImage {
  uint32_t width = 0, height = 0;
  std::vector<uint8_t> pixels;  // BGRA8，行距 width * 4
};

// 用 WIC 解码图片文件（JPEG/PNG 等）。调用线程需已初始化 COM。
BgraImage LoadBgraImage(const std::filesystem::path& path);

Microsoft::WRL::ComPtr<ID3D11Texture2D> CreateTexture(ID3D11Device* device, const BgraImage& image);

}  // namespace nr
