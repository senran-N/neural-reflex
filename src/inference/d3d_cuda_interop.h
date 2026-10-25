#pragma once

// Direct3D 11 与 CUDA 的资源交接（设计方案 4.2）：启动时以 NT 句柄共享并导入一次，
// 之后每帧只做 fence 的 signal/wait，不做图形资源映射。

#include <cuda_runtime.h>
#include <d3d11_4.h>
#include <wrl/client.h>

#include <cstdint>

namespace nr {

// CUDA 设备与 D3D 适配器必须是同一块 GPU，按 LUID 匹配。
int CudaDeviceForLuid(const LUID& luid);

// 可被 CUDA 以纹理对象读取的 D3D11 纹理（BGRA8，逐元素读取、坐标不归一化）。
class SharedTexture {
 public:
  SharedTexture(ID3D11Device* device, uint32_t width, uint32_t height);
  ~SharedTexture();
  SharedTexture(const SharedTexture&) = delete;
  SharedTexture& operator=(const SharedTexture&) = delete;

  ID3D11Texture2D* D3D() const { return texture_.Get(); }
  cudaTextureObject_t Cuda() const { return textureObject_; }
  uint32_t Width() const { return width_; }
  uint32_t Height() const { return height_; }

 private:
  Microsoft::WRL::ComPtr<ID3D11Texture2D> texture_;
  HANDLE sharedHandle_ = nullptr;
  cudaExternalMemory_t externalMemory_ = nullptr;
  cudaMipmappedArray_t mipmappedArray_ = nullptr;
  cudaTextureObject_t textureObject_ = 0;
  uint32_t width_, height_;
};

// D3D11 与 CUDA 共用的单调递增 fence。
class SharedFence {
 public:
  explicit SharedFence(ID3D11Device* device);
  ~SharedFence();
  SharedFence(const SharedFence&) = delete;
  SharedFence& operator=(const SharedFence&) = delete;

  // D3D 侧：在即时上下文的命令流中 signal / wait。
  void D3DSignal(ID3D11DeviceContext* context, uint64_t value);
  void D3DWait(ID3D11DeviceContext* context, uint64_t value);
  // CUDA 侧：在 stream 上 wait / signal。
  void CudaWait(cudaStream_t stream, uint64_t value);
  void CudaSignal(cudaStream_t stream, uint64_t value);

  uint64_t Next() { return ++value_; }

 private:
  Microsoft::WRL::ComPtr<ID3D11Fence> fence_;
  HANDLE sharedHandle_ = nullptr;
  cudaExternalSemaphore_t semaphore_ = nullptr;
  uint64_t value_ = 0;
};

}  // namespace nr
