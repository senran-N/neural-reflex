#include "inference/preprocess.h"

namespace nr {
namespace {

constexpr float kPad = 114.0f / 255.0f;

// 与 OpenCV resize 的坐标规则一致：半像素中心对齐，越界时钳到边缘像素
__device__ void SourceCoord(int dst, float invScale, int srcSize, int& i0, int& i1, float& frac) {
  float f = (dst + 0.5f) * invScale - 0.5f;
  int i = static_cast<int>(floorf(f));
  frac = f - i;
  if (i < 0) {
    i = 0;
    frac = 0.0f;
  }
  if (i >= srcSize - 1) {
    i = srcSize - 1;
    frac = 0.0f;
  }
  i0 = i;
  i1 = min(i + 1, srcSize - 1);
}

__global__ void LetterboxKernel(cudaTextureObject_t source, LetterboxParams p, float* output) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= p.size || y >= p.size) return;

  float r = kPad, g = kPad, b = kPad;
  const int ix = x - p.left, iy = y - p.top;
  if (ix >= 0 && iy >= 0 && ix < p.newW && iy < p.newH) {
    int x0, x1, y0, y1;
    float fx, fy;
    SourceCoord(ix, static_cast<float>(p.srcW) / p.newW, p.srcW, x0, x1, fx);
    SourceCoord(iy, static_cast<float>(p.srcH) / p.newH, p.srcH, y0, y1, fy);
    const uchar4 c00 = tex2D<uchar4>(source, x0 + 0.5f, y0 + 0.5f);
    const uchar4 c10 = tex2D<uchar4>(source, x1 + 0.5f, y0 + 0.5f);
    const uchar4 c01 = tex2D<uchar4>(source, x0 + 0.5f, y1 + 0.5f);
    const uchar4 c11 = tex2D<uchar4>(source, x1 + 0.5f, y1 + 0.5f);
    const float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    // 纹理为 BGRA：x=B, y=G, z=R
    const float bb = w00 * c00.x + w10 * c10.x + w01 * c01.x + w11 * c11.x;
    const float gg = w00 * c00.y + w10 * c10.y + w01 * c01.y + w11 * c11.y;
    const float rr = w00 * c00.z + w10 * c10.z + w01 * c01.z + w11 * c11.z;
    r = rintf(rr) * (1.0f / 255.0f);
    g = rintf(gg) * (1.0f / 255.0f);
    b = rintf(bb) * (1.0f / 255.0f);
  }
  const int plane = p.size * p.size;
  const int i = y * p.size + x;
  output[i] = r;
  output[plane + i] = g;
  output[2 * plane + i] = b;
}

}  // namespace

void LaunchLetterbox(cudaTextureObject_t source, const LetterboxParams& params, float* output,
                     cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((params.size + block.x - 1) / block.x, (params.size + block.y - 1) / block.y);
  LetterboxKernel<<<grid, block, 0, stream>>>(source, params, output);
}

}  // namespace nr
