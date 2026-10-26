#pragma once

#include "common/detection.h"
#include "inference/letterbox.h"

#include <cuda_runtime.h>

namespace nr {

struct PostprocessParams {
  int numAnchors = 0;
  int numClasses = 0;
  int numKeypoints = 0;  // 类别行之后紧跟 3 * numKeypoints 行 (x, y, score)
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
  DetectionOutput* output = nullptr;  // 结果写入位置（显存），由调用方提供
};

PostprocessBuffers AllocatePostprocessBuffers(int numAnchors);
void FreePostprocessBuffers(PostprocessBuffers& buffers);

// 解码一对多检测头输出 [1, 4 + nc + 3k, N]，单标签筛选，按类别 NMS，还原坐标；k > 0 时附带所选锚点的关键点。
// 候选超过容量时保留分数最高的 kMaxCandidates 个。
// 与 tools/model_export/reference_pipeline.py 的 postprocess / postprocess_pose 一致。
void LaunchPostprocess(const float* modelOutput, const PostprocessParams& params,
                       const LetterboxParams& letterbox, const PostprocessBuffers& buffers,
                       cudaStream_t stream);

// 解码端到端检测头输出 [1, K, 6]（x1, y1, x2, y2, score, cls），只做阈值筛选并还原坐标，保持模型给出的顺序。
// K 不超过 kMaxCandidates。与 reference_pipeline.py 的 postprocess_end2end 一致。
void LaunchEnd2EndPostprocess(const float* modelOutput, int rows, const PostprocessParams& params,
                              const LetterboxParams& letterbox, DetectionOutput* output, cudaStream_t stream);

}  // namespace nr
