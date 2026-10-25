#pragma once

#include "inference/d3d_cuda_interop.h"
#include "inference/letterbox.h"
#include "inference/postprocess.h"
#include "inference/trt_engine.h"

#include <cuda_runtime.h>
#include <d3d11.h>

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
// 所有方法在同一线程调用，D3D 即时上下文由调用方独占。
class Detector {
 public:
  Detector(ID3D11Device* device, int cudaDevice, const DetectorOptions& options);
  ~Detector();
  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;

  // region 为 source 纹理中的检测区域；结果坐标为区域内坐标加 offset。
  // 返回的指针指向页锁定内存，下一次 Run 前有效。
  const DetectionOutput& Run(ID3D11DeviceContext* context, ID3D11Texture2D* source, const D3D11_BOX& region,
                             float offsetX, float offsetY);

  const StageTimes& LastTimes() const { return times_; }
  const LetterboxParams& LastLetterbox() const { return letterbox_; }
  int InputSize() const { return inputSize_; }
  int NumClasses() const { return post_.numClasses; }

  // 下一次 Run 起生效
  void SetThresholds(float conf, float iou) {
    post_.confThreshold = conf;
    post_.iouThreshold = iou;
  }

  // 调试用：读回最近一次的模型输入张量
  void ReadInput(float* host) const;

  void SaveCaches() { engine_->SaveRuntimeCache(); }

 private:
  ID3D11Device* device_;
  DetectorOptions options_;
  std::unique_ptr<TrtEngine> engine_;
  std::unique_ptr<SharedTexture> texture_;
  std::unique_ptr<SharedFence> fence_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t events_[4] = {};
  PostprocessBuffers buffers_;
  DetectionOutput* hostOutput_ = nullptr;
  PostprocessParams post_;
  LetterboxParams letterbox_;
  StageTimes times_;
  int inputSize_ = 0;
  uint64_t textureReleased_ = 0;  // CUDA 读完共享纹理后 signal 的值
};

}  // namespace nr
