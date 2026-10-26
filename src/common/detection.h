#pragma once

// 每帧检测结果（扩展机制方案 4.2）。纯数据结构，不依赖 CUDA：推理模块在显存中写入，
// 结果发布与插件工具在主机端读取。

#include <cstddef>
#include <cstdint>

namespace nr {

// 固定容量（设计方案 5.1）：候选上限与 NMS 单个线程块的大小一致。
constexpr int kMaxCandidates = 1024;
constexpr int kMaxDetections = 300;
constexpr int kMaxKeypoints = 32;  // 每个目标的关键点上限（扩展机制方案 4.2）

struct Detection {
  float x1, y1, x2, y2;  // 还原到检测区域坐标后再加上 offset
  float score;
  int32_t cls;
};

struct Keypoint {
  float x, y;  // 与目标框相同的坐标系
  float score;
};

struct DetectionOutput {
  int32_t count;         // 最终框数
  int32_t candidates;    // 超过置信度阈值的候选总数（可能大于容量）
  int32_t overflow;      // 因容量不足被丢弃的候选数
  int32_t numKeypoints;  // 每个目标的关键点数，0 表示没有关键点
  Detection detections[kMaxDetections];
  // 第 i 个目标的第 j 个点位于 keypoints[i * numKeypoints + j]；没有关键点的模型不回读这一段
  Keypoint keypoints[kMaxDetections * kMaxKeypoints];
};

// 回读时只需复制到这里；有关键点时再复制 kMaxDetections * numKeypoints 个点
constexpr size_t kDetectionOutputHeadBytes = offsetof(DetectionOutput, keypoints);

}  // namespace nr
