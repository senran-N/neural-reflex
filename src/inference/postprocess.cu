#include "inference/postprocess.h"

#include "inference/cuda_check.h"

#include <cfloat>

namespace nr {
namespace {

struct Candidate {
  float x1, y1, x2, y2;
  float score;
  int32_t cls;
};

// 溢出时按分数保留：分数落在 (conf, 1] 的直方图里，先算出截断格，再写入候选。
// 高于截断格的候选占用预留槽位，只有截断格内部的取舍是任意的（格宽约 (1 - conf) / 2048）。
constexpr int kBins = 2048;

// 计数器布局
enum Counter { kTotal = 0, kHigh = 1, kMid = 2, kCutBin = 3, kHighCount = 4, kCounterCount = 5 };

__device__ int ScoreBin(float s, float conf) {
  const int b = static_cast<int>((s - conf) / (1.0f - conf) * kBins);
  return min(max(b, 0), kBins - 1);
}

__global__ void ScoreKernel(const float* out, PostprocessParams p, float* bestScore, int32_t* bestClass,
                            int32_t* histogram, int32_t* counters) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= p.numAnchors) return;

  // 单标签：取最高分类别，并列时取类别号最小者（与 numpy argmax 一致）
  float best = -FLT_MAX;
  int cls = 0;
  for (int c = 0; c < p.numClasses; ++c) {
    const float s = out[(4 + c) * p.numAnchors + a];
    if (s > best) {
      best = s;
      cls = c;
    }
  }
  bestScore[a] = best;
  bestClass[a] = cls;
  if (!(best > p.confThreshold)) return;
  atomicAdd(&counters[kTotal], 1);
  atomicAdd(&histogram[ScoreBin(best, p.confThreshold)], 1);
}

// 单线程：从最高格向下累加，找到使保留数不超过容量的截断格
__global__ void CutoffKernel(const int32_t* histogram, int32_t* counters) {
  const int total = counters[kTotal];
  if (total <= kMaxCandidates) {
    counters[kCutBin] = -1;
    counters[kHighCount] = total;
    return;
  }
  int kept = 0;
  for (int b = kBins - 1; b >= 0; --b) {
    if (kept + histogram[b] > kMaxCandidates) {
      counters[kCutBin] = b;
      counters[kHighCount] = kept;
      return;
    }
    kept += histogram[b];
  }
}

__global__ void WriteKernel(const float* out, PostprocessParams p, const float* bestScore, const int32_t* bestClass,
                            Candidate* candidates, int32_t* counters) {
  const int a = blockIdx.x * blockDim.x + threadIdx.x;
  if (a >= p.numAnchors) return;
  const float best = bestScore[a];
  if (!(best > p.confThreshold)) return;

  const int bin = ScoreBin(best, p.confThreshold);
  const int cut = counters[kCutBin];
  int slot;
  if (bin > cut) {
    slot = atomicAdd(&counters[kHigh], 1);
  } else if (bin == cut) {
    slot = counters[kHighCount] + atomicAdd(&counters[kMid], 1);
    if (slot >= kMaxCandidates) return;
  } else {
    return;
  }
  const float cx = out[a], cy = out[p.numAnchors + a];
  const float w = out[2 * p.numAnchors + a], h = out[3 * p.numAnchors + a];
  candidates[slot] = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, best, bestClass[a]};
}

__device__ float Iou(const float4& a, const float4& b) {
  const float ix = fmaxf(0.0f, fminf(a.z, b.z) - fmaxf(a.x, b.x));
  const float iy = fmaxf(0.0f, fminf(a.w, b.w) - fmaxf(a.y, b.y));
  const float inter = ix * iy;
  const float uni = (a.z - a.x) * (a.w - a.y) + (b.z - b.x) * (b.w - b.y) - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

// 单个线程块：按分数降序排序（双调排序），再逐个做按类别的贪心抑制。
__global__ void __launch_bounds__(kMaxCandidates)
NmsKernel(const Candidate* candidates, const int32_t* counters, PostprocessParams p, LetterboxParams lb,
          DetectionOutput* output) {
  __shared__ float scores[kMaxCandidates];
  __shared__ int16_t order[kMaxCandidates];
  __shared__ float4 boxes[kMaxCandidates];
  __shared__ int32_t classes[kMaxCandidates];
  __shared__ uint8_t suppressed[kMaxCandidates];

  const int t = threadIdx.x;
  const int n = min(counters[kTotal], kMaxCandidates);
  scores[t] = t < n ? candidates[t].score : -FLT_MAX;
  order[t] = static_cast<int16_t>(t);
  __syncthreads();

  for (int k = 2; k <= kMaxCandidates; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int other = t ^ j;
      if (other > t) {
        // 分数高者在前；分数相同时下标小者在前，保证结果确定
        const bool before = scores[t] > scores[other] || (scores[t] == scores[other] && order[t] < order[other]);
        const bool descending = (t & k) == 0;
        if (descending != before) {
          const float s = scores[t]; scores[t] = scores[other]; scores[other] = s;
          const int16_t o = order[t]; order[t] = order[other]; order[other] = o;
        }
      }
      __syncthreads();
    }
  }

  if (t < n) {
    const Candidate& c = candidates[order[t]];
    boxes[t] = make_float4(c.x1, c.y1, c.x2, c.y2);
    classes[t] = c.cls;
  }
  suppressed[t] = 0;
  __syncthreads();

  for (int i = 0; i < n; ++i) {
    if (!suppressed[i] && t > i && t < n && !suppressed[t] && classes[t] == classes[i] &&
        Iou(boxes[i], boxes[t]) > p.iouThreshold) {
      suppressed[t] = 1;
    }
    __syncthreads();
  }

  if (t == 0) {
    int kept = 0;
    for (int i = 0; i < n && kept < p.maxDetections; ++i) {
      if (suppressed[i]) continue;
      const float4 b = boxes[i];
      Detection& d = output->detections[kept++];
      // 去掉填充、除以缩放比例，裁到检测区域内，再平移
      d.x1 = fminf(fmaxf((b.x - lb.left) / lb.scale, 0.0f), static_cast<float>(lb.srcW)) + p.offsetX;
      d.y1 = fminf(fmaxf((b.y - lb.top) / lb.scale, 0.0f), static_cast<float>(lb.srcH)) + p.offsetY;
      d.x2 = fminf(fmaxf((b.z - lb.left) / lb.scale, 0.0f), static_cast<float>(lb.srcW)) + p.offsetX;
      d.y2 = fminf(fmaxf((b.w - lb.top) / lb.scale, 0.0f), static_cast<float>(lb.srcH)) + p.offsetY;
      d.score = scores[i];
      d.cls = classes[i];
    }
    output->count = kept;
    output->candidates = counters[kTotal];
    output->overflow = max(0, counters[kTotal] - kMaxCandidates);
  }
}

}  // namespace

PostprocessBuffers AllocatePostprocessBuffers(int numAnchors) {
  PostprocessBuffers b;
  b.numAnchors = numAnchors;
  CudaCheck(cudaMalloc(&b.candidates, sizeof(Candidate) * kMaxCandidates), "cudaMalloc(candidates)");
  CudaCheck(cudaMalloc(&b.counters, sizeof(int32_t) * kCounterCount), "cudaMalloc(counters)");
  CudaCheck(cudaMalloc(&b.histogram, sizeof(int32_t) * kBins), "cudaMalloc(histogram)");
  CudaCheck(cudaMalloc(&b.bestScore, sizeof(float) * numAnchors), "cudaMalloc(scores)");
  CudaCheck(cudaMalloc(&b.bestClass, sizeof(int32_t) * numAnchors), "cudaMalloc(classes)");
  CudaCheck(cudaMalloc(&b.output, sizeof(DetectionOutput)), "cudaMalloc(detections)");
  return b;
}

void FreePostprocessBuffers(PostprocessBuffers& b) {
  cudaFree(b.candidates);
  cudaFree(b.counters);
  cudaFree(b.histogram);
  cudaFree(b.bestScore);
  cudaFree(b.bestClass);
  cudaFree(b.output);
  b = {};
}

void LaunchPostprocess(const float* modelOutput, const PostprocessParams& params, const LetterboxParams& letterbox,
                       const PostprocessBuffers& buffers, cudaStream_t stream) {
  CudaCheck(cudaMemsetAsync(buffers.counters, 0, sizeof(int32_t) * kCounterCount, stream), "cudaMemsetAsync(counters)");
  CudaCheck(cudaMemsetAsync(buffers.histogram, 0, sizeof(int32_t) * kBins, stream), "cudaMemsetAsync(histogram)");
  const int block = 256;
  const int grid = (params.numAnchors + block - 1) / block;
  ScoreKernel<<<grid, block, 0, stream>>>(modelOutput, params, buffers.bestScore, buffers.bestClass, buffers.histogram,
                                          buffers.counters);
  CutoffKernel<<<1, 1, 0, stream>>>(buffers.histogram, buffers.counters);
  WriteKernel<<<grid, block, 0, stream>>>(modelOutput, params, buffers.bestScore, buffers.bestClass,
                                          static_cast<Candidate*>(buffers.candidates), buffers.counters);
  NmsKernel<<<1, kMaxCandidates, 0, stream>>>(static_cast<const Candidate*>(buffers.candidates), buffers.counters,
                                              params, letterbox, buffers.output);
}

}  // namespace nr
