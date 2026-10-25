#pragma once

// 合成测试源的帧标签（设计方案 7.4）：把帧编号、生成时刻和目标框编码成
// 黑白方格，采集端解码后即可做帧对应、延迟测量和坐标还原验证。

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace nr::tag {

constexpr int kCellPx = 8;
constexpr int kCols = 48;
constexpr int kRows = 4;
constexpr int kBits = kCols * kRows;  // 6 个 32 位字

// 标签左上角在测试源客户区中的位置
constexpr int kOriginX = 8;
constexpr int kOriginY = 8;
constexpr int kWidthPx = kCols * kCellPx;
constexpr int kHeightPx = kRows * kCellPx;

struct Payload {
  uint32_t frameId = 0;
  int64_t qpc = 0;  // 测试源生成该帧时的 QPC
  // 目标框，测试源客户区坐标
  uint16_t boxX = 0, boxY = 0, boxW = 0, boxH = 0;
};

std::array<bool, kBits> Encode(const Payload& payload);

// 输入为标签区域（kWidthPx x kHeightPx）的 BGRA8 像素。校验失败或像素
// 不是清晰的黑白（缩放、混合、撕裂）时返回空。
std::optional<Payload> Decode(const uint8_t* bgra, size_t rowPitch);

}  // namespace nr::tag
