#pragma once

#include "inference/d3d_cuda_interop.h"
#include "inference/letterbox.h"
#include "inference/postprocess.h"
#include "inference/trt_engine.h"

#include <cuda_runtime.h>
#include <d3d11.h>

#include <array>
#include <memory>

namespace nr {

struct DetectorOptions {
  EngineOptions engine;
  float confThreshold = 0.25f;
  float iouThreshold = 0.7f;
};

struct StageTimes {
  float preprocessMs = 0, inferenceMs = 0, postprocessMs = 0;  // GPU 事件计时
};

// 检测流水线：检测区域复制进共享纹理 → fence → CUDA 预处理 → 推理 → 后处理 → 回读结果。
// 提交与取结果分开，最多 kSlots 帧在途（设计方案 6.1）：GPU 处理第 N 帧时，第 N+1 帧
// 的采集与复制已可进行。GPU 上所有工作在同一个 stream 中按序执行。
// Submit 只能在调用方独占的 D3D 即时上下文所在线程调用；Wait 可在另一个线程调用。
class Detector {
 public:
  static constexpr int kSlots = 2;

  Detector(ID3D11Device* device, int cudaDevice, const DetectorOptions& options);
  ~Detector();
  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;

  // 把 source 中的 region 复制进槽位并提交全部 GPU 工作，不等待。返回槽位号。
  // 调用方须保证该槽位上一次的结果已经 Wait 过。复制命令发出后即可释放 source。
  int Submit(ID3D11DeviceContext* context, ID3D11Texture2D* source, const D3D11_BOX& region, float offsetX,
             float offsetY, int slot);

  // 等待槽位结果。返回的引用指向该槽位的页锁定内存，下次提交到该槽位前有效。
  const DetectionOutput& Wait(int slot, StageTimes* times = nullptr);

  // 同步执行一帧（离线模式）
  const DetectionOutput& Run(ID3D11DeviceContext* context, ID3D11Texture2D* source, const D3D11_BOX& region,
                             float offsetX, float offsetY);

  int InputSize() const { return inputSize_; }
  int NumClasses() const { return post_.numClasses; }

  // 下一次 Submit 起生效
  void SetThresholds(float conf, float iou) {
    post_.confThreshold = conf;
    post_.iouThreshold = iou;
  }

  // 调试用：读回最近一次的模型输入张量
  void ReadInput(float* host) const;

  void SaveCaches() { engine_->SaveRuntimeCache(); }

 private:
  struct Slot {
    std::unique_ptr<SharedTexture> texture;
    uint64_t released = 0;  // CUDA 读完该槽位纹理后 signal 的 fence 值
    DetectionOutput* host = nullptr;
    cudaEvent_t events[4] = {};
  };

  ID3D11Device* device_;
  DetectorOptions options_;
  std::unique_ptr<TrtEngine> engine_;
  std::unique_ptr<SharedFence> fence_;
  cudaStream_t stream_ = nullptr;
  std::array<Slot, kSlots> slots_;
  PostprocessBuffers buffers_;
  PostprocessParams post_;
  int inputSize_ = 0;
};

}  // namespace nr
