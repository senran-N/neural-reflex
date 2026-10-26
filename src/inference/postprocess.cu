#include "inference/postprocess.h"

#include "inference/cuda_check.h"

#include <cfloat>
#include <climits>

namespace nr {
namespace {

struct Candidate {
  float x1, y1, x2, y2;
  float score;
  int32_t cls;
  int32_t anchor;  // 取附带数据（关键点）时使用
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
  candidates[slot] = {cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, best, bestClass[a], a};
}

__device__ float Iou(const float4& a, const float4& b) {
  const float ix = fmaxf(0.0f, fminf(a.z, b.z) - fmaxf(a.x, b.x));
  const float iy = fmaxf(0.0f, fminf(a.w, b.w) - fmaxf(a.y, b.y));
  const float inter = ix * iy;
  const float uni = (a.z - a.x) * (a.w - a.y) + (b.z - b.x) * (b.w - b.y) - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

// 去掉填充、除以缩放比例，裁到检测区域内，再平移
__device__ float RestoreX(float x, const LetterboxParams& lb, const PostprocessParams& p) {
  return fminf(fmaxf((x - lb.left) / lb.scale, 0.0f), static_cast<float>(lb.srcW)) + p.offsetX;
}
__device__ float RestoreY(float y, const LetterboxParams& lb, const PostprocessParams& p) {
  return fminf(fmaxf((y - lb.top) / lb.scale, 0.0f), static_cast<float>(lb.srcH)) + p.offsetY;
}

// 单个线程块：按分数降序排序（双调排序），再逐个做按类别的贪心抑制；有关键点时由全部线程并行写出。
__global__ void __launch_bounds__(kMaxCandidates)
NmsKernel(const float* out, const Candidate* candidates, const int32_t* counters, PostprocessParams p,
          LetterboxParams lb, DetectionOutput* output) {
  __shared__ float scores[kMaxCandidates];
  __shared__ int16_t order[kMaxCandidates];
  __shared__ int32_t anchors[kMaxCandidates];  // 排序键的第二位，与原子操作的先后无关
  __shared__ float4 boxes[kMaxCandidates];
  __shared__ int32_t classes[kMaxCandidates];
  __shared__ uint8_t suppressed[kMaxCandidates];
  __shared__ int16_t keptRank[kMaxDetections];  // 第 d 个结果在排序后的位置
  __shared__ int32_t keptCount;

  const int t = threadIdx.x;
  const int n = min(counters[kTotal], kMaxCandidates);
  scores[t] = t < n ? candidates[t].score : -FLT_MAX;
  anchors[t] = t < n ? candidates[t].anchor : INT_MAX;
  order[t] = static_cast<int16_t>(t);
  __syncthreads();

  for (int k = 2; k <= kMaxCandidates; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int other = t ^ j;
      if (other > t) {
        // 分数高者在前；分数相同时锚点号小者在前。候选写入位置取决于原子操作的先后，
        // 不能作为第二键，否则并列分数的顺序（进而 NMS 结果）每次运行都可能不同
        const bool before = scores[t] > scores[other] || (scores[t] == scores[other] && anchors[t] < anchors[other]);
        const bool descending = (t & k) == 0;
        if (descending != before) {
          const float s = scores[t]; scores[t] = scores[other]; scores[other] = s;
          const int32_t a = anchors[t]; anchors[t] = anchors[other]; anchors[other] = a;
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
      keptRank[kept] = static_cast<int16_t>(i);
      Detection& d = output->detections[kept++];
      d.x1 = RestoreX(b.x, lb, p);
      d.y1 = RestoreY(b.y, lb, p);
      d.x2 = RestoreX(b.z, lb, p);
      d.y2 = RestoreY(b.w, lb, p);
      d.score = scores[i];
      d.cls = classes[i];
    }
    keptCount = kept;
    output->count = kept;
    output->candidates = counters[kTotal];
    output->overflow = max(0, counters[kTotal] - kMaxCandidates);
    output->numKeypoints = p.numKeypoints;
  }

  if (p.numKeypoints > 0) {
    __syncthreads();
    const int total = keptCount * p.numKeypoints;
    for (int e = t; e < total; e += blockDim.x) {
      const int d = e / p.numKeypoints, j = e % p.numKeypoints;
      const int anchor = candidates[order[keptRank[d]]].anchor;
      const float* row = out + static_cast<size_t>(4 + p.numClasses + 3 * j) * p.numAnchors + anchor;
      Keypoint& k = output->keypoints[e];
      k.x = RestoreX(row[0], lb, p);
      k.y = RestoreY(row[p.numAnchors], lb, p);
      k.score = row[2 * p.numAnchors];
    }
  }
}

// 单个线程块：每个线程一行，阈值筛选后按行号前缀和确定输出位置，保持模型给出的顺序。
__global__ void __launch_bounds__(kMaxCandidates)
End2EndKernel(const float* out, int rows, PostprocessParams p, LetterboxParams lb, DetectionOutput* output) {
  __shared__ int32_t scan[kMaxCandidates];
  const int t = threadIdx.x;
  const float* r = out + static_cast<size_t>(t) * 6;
  const bool keep = t < rows && r[4] > p.confThreshold;
  scan[t] = keep ? 1 : 0;
  __syncthreads();
  for (int offset = 1; offset < kMaxCandidates; offset <<= 1) {
    const int v = t >= offset ? scan[t - offset] : 0;
    __syncthreads();
    scan[t] += v;
    __syncthreads();
  }
  const int total = scan[kMaxCandidates - 1];
  const int slot = scan[t] - 1;
  if (keep && slot < p.maxDetections) {
    Detection& d = output->detections[slot];
    d.x1 = RestoreX(r[0], lb, p);
    d.y1 = RestoreY(r[1], lb, p);
    d.x2 = RestoreX(r[2], lb, p);
    d.y2 = RestoreY(r[3], lb, p);
    d.score = r[4];
    d.cls = static_cast<int32_t>(r[5]);
  }
  if (t == 0) {
    output->count = min(total, p.maxDetections);
    output->candidates = total;
    output->overflow = max(0, total - p.maxDetections);
    output->numKeypoints = 0;
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
  return b;
}

void FreePostprocessBuffers(PostprocessBuffers& b) {
  cudaFree(b.candidates);
  cudaFree(b.counters);
  cudaFree(b.histogram);
  cudaFree(b.bestScore);
  cudaFree(b.bestClass);
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
  NmsKernel<<<1, kMaxCandidates, 0, stream>>>(modelOutput, static_cast<const Candidate*>(buffers.candidates),
                                              buffers.counters, params, letterbox, buffers.output);
}

void LaunchEnd2EndPostprocess(const float* modelOutput, int rows, const PostprocessParams& params,
                              const LetterboxParams& letterbox, DetectionOutput* output, cudaStream_t stream) {
  End2EndKernel<<<1, kMaxCandidates, 0, stream>>>(modelOutput, rows, params, letterbox, output);
}

}  // namespace nr
