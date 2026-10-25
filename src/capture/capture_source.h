#pragma once

#include <d3d11.h>
#include <windows.h>

#include <cstdint>

namespace nr {

struct CapturedFrame {
  // 程序自有纹理，采集帧已复制进来并释放。下一次 AcquireLatest 前有效。
  ID3D11Texture2D* texture = nullptr;
  uint32_t width = 0;
  uint32_t height = 0;
  POINT screenOrigin{};    // 纹理 (0,0) 对应的屏幕物理像素坐标
  int64_t sourceQpc = 0;   // 采集 API 给出的画面时间戳（QPC），0 表示没有
  int64_t acquireQpc = 0;  // 本程序取得该帧的时刻
  uint32_t accumulated = 1;  // 自上次取得后累积的画面更新数（>1 表示中间有更新被合并）
};

enum class AcquireStatus { kNewFrame, kTimeout, kLost };

class CaptureSource {
 public:
  virtual ~CaptureSource() = default;
  virtual const char* Name() const = 0;

  // 等待新画面，返回期间到达的最新一帧；旧帧直接丢弃，不排队。
  virtual AcquireStatus AcquireLatest(uint32_t timeoutMs, CapturedFrame& out) = 0;

  // kLost 后调用，成功返回 true（设计方案 6.6）。
  virtual bool Recreate() = 0;
};

}  // namespace nr
