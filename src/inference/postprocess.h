#pragma once

#include "inference/letterbox.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace nr {

// 固定容量（设计方案 5.1）：候选上限与 NMS 单个线程块的大小一致。
constexpr int kMaxCandidates = 1024;
constexpr int kMaxDetections = 300;

struct Detection {
  float x1, y1, x2, y2;  // 还原到检测区域坐标后再加上 offset
  float score;
  int32_t cls;
};

struct DetectionOutput {
  int32_t count;       // 最终框数
  int32_t candidates;  // 超过置信度阈值的候选总数（可能大于容量）
  int32_t overflow;    // 因容量不足被丢弃的候选数
  int32_t reserved;
  Detection detections[kMaxDetections];
};

struct PostprocessParams {
  int numAnchors = 0;
  int numClasses = 0;
  float confThreshold = 0.25f;
  float iouThreshold = 0.7f;
  int maxDetections = kMaxDetections;
  float offsetX = 0, offsetY = 0;  // 结果坐标的平移量，例如检测区域的屏幕原点
};

struct PostprocessBuffers {
  int numAnchors = 0;
  void* candidates = nullptr;    // kMaxCandidates 个候选
  int32_t* counters = nullptr;   // 候选总数、槽位计数与截断格
  int32_t* histogram = nullptr;  // 候选分数直方图，用于溢出时按分数保留
  float* bestScore = nullptr;    // 每个锚点的最高类别分数
  int32_t* bestClass = nullptr;
  DetectionOutput* output = nullptr;
};

PostprocessBuffers AllocatePostprocessBuffers(int numAnchors);
void FreePostprocessBuffers(PostprocessBuffers& buffers);

// 解码一对多检测头输出 [1, 4 + nc, N]，单标签筛选，按类别 NMS，还原坐标。
// 候选超过容量时保留分数最高的 kMaxCandidates 个。
// 与 tools/model_export/reference_pipeline.py 的 postprocess 一致。
void LaunchPostprocess(const float* modelOutput, const PostprocessParams& params,
                       const LetterboxParams& letterbox, const PostprocessBuffers& buffers,
                       cudaStream_t stream);

}  // namespace nr
