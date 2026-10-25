#include "common/frame_tag.h"

namespace nr::tag {
namespace {

constexpr int kWords = kBits / 32;
static_assert(kWords == 6);

uint32_t Checksum(const uint32_t* words) {
  uint32_t h = 2166136261u;  // FNV-1a
  for (int i = 0; i < kWords - 1; ++i) {
    for (int b = 0; b < 4; ++b) {
      h ^= (words[i] >> (b * 8)) & 0xFFu;
      h *= 16777619u;
    }
  }
  return h;
}

void ToWords(const Payload& p, uint32_t* w) {
  w[0] = p.frameId;
  w[1] = static_cast<uint32_t>(static_cast<uint64_t>(p.qpc));
  w[2] = static_cast<uint32_t>(static_cast<uint64_t>(p.qpc) >> 32);
  w[3] = p.boxX | (static_cast<uint32_t>(p.boxY) << 16);
  w[4] = p.boxW | (static_cast<uint32_t>(p.boxH) << 16);
  w[5] = Checksum(w);
}

}  // namespace

std::array<bool, kBits> Encode(const Payload& payload) {
  uint32_t words[kWords];
  ToWords(payload, words);
  std::array<bool, kBits> bits{};
  for (int i = 0; i < kBits; ++i) bits[i] = (words[i / 32] >> (i % 32)) & 1u;
  return bits;
}

std::optional<Payload> Decode(const uint8_t* bgra, size_t rowPitch) {
  uint32_t words[kWords] = {};
  for (int i = 0; i < kBits; ++i) {
    const int cx = (i % kCols) * kCellPx + kCellPx / 2;
    const int cy = (i / kCols) * kCellPx + kCellPx / 2;
    const uint8_t g = bgra[cy * rowPitch + cx * 4 + 1];
    if (g > 64 && g < 192) return std::nullopt;  // 不清晰的像素
    if (g >= 128) words[i / 32] |= 1u << (i % 32);
  }
  if (Checksum(words) != words[5]) return std::nullopt;

  Payload p;
  p.frameId = words[0];
  p.qpc = static_cast<int64_t>(words[1] | (static_cast<uint64_t>(words[2]) << 32));
  p.boxX = static_cast<uint16_t>(words[3]);
  p.boxY = static_cast<uint16_t>(words[3] >> 16);
  p.boxW = static_cast<uint16_t>(words[4]);
  p.boxH = static_cast<uint16_t>(words[4] >> 16);
  return p;
}

}  // namespace nr::tag
