#pragma once

// 输出解码器：把推理输出张量解码为统一结果（扩展机制方案 4.3）。
// 实时路径约束：只在给定 stream 上排队工作，不同步；显存在构造时一次分配；结果不超过 DetectionOutput 的容量。

#include "inference/letterbox.h"
#include "inference/model_manifest.h"
#include "inference/postprocess.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nr {

struct DecodeParams {
  float confThreshold = 0.25f;
  float iouThreshold = 0.7f;
  float offsetX = 0, offsetY = 0;  // 结果坐标的平移量，例如检测区域的屏幕原点
  LetterboxParams letterbox;
};

class OutputDecoder {
 public:
  virtual ~OutputDecoder() = default;
  virtual const char* Name() const = 0;
  virtual int NumClasses() const = 0;
  virtual int NumKeypoints() const = 0;
  virtual void Enqueue(const float* modelOutput, const DecodeParams& params, DetectionOutput* output,
                       cudaStream_t stream) = 0;
};

// 清单的 decoder 字段；未声明时按输出形状推断（只能推断检测头），无法推断时抛出异常。
std::string ResolveDecoderName(const ModelManifest& manifest, const std::vector<int64_t>& outputShape);

// 按 ResolveDecoderName 的结果创建。形状与解码器要求不符时抛出异常。
std::unique_ptr<OutputDecoder> CreateDecoder(const ModelManifest& manifest, const std::vector<int64_t>& outputShape);

}  // namespace nr
