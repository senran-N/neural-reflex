#pragma once

#include <d3d11.h>
#include <windows.h>

#include <cstdint>

namespace nr {

struct CapturedFrame {
  // 采集接口交出的原始画面（不经过额外复制）。使用方发出自己的 GPU 复制后调用
  // CaptureSource::ReleaseFrame()，之后不得再访问（设计方案 4.2：只允许一次 GPU 内复制）。
  ID3D11Texture2D* texture = nullptr;
  uint32_t width = 0;   // 有效内容尺寸，纹理本身可能更大
  uint32_t height = 0;
  POINT screenOrigin{};    // 画面 (0,0) 对应的屏幕物理像素坐标
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
  // 上一帧未释放时先自动释放。
  virtual AcquireStatus AcquireLatest(uint32_t timeoutMs, CapturedFrame& out) = 0;

  // 使用方已发出对画面的复制命令后调用，尽快把缓冲还给系统。
  virtual void ReleaseFrame() = 0;

  // kLost 后调用，成功返回 true（设计方案 6.6）。
  virtual bool Recreate() = 0;
};

}  // namespace nr
