#include "inference/decoder.h"

#include <stdexcept>
#include <string>

namespace nr {
namespace {

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
  return s + "]";
}

// 一对多头 [1, 4 + nc + 3k, N]：yolo-anchors（k = 0）与 yolo-pose-anchors
class AnchorDecoder final : public OutputDecoder {
 public:
  AnchorDecoder(const char* name, int numClasses, int numKeypoints, int numAnchors) : name_(name) {
    params_.numAnchors = numAnchors;
    params_.numClasses = numClasses;
    params_.numKeypoints = numKeypoints;
    buffers_ = AllocatePostprocessBuffers(numAnchors);
  }
  ~AnchorDecoder() override { FreePostprocessBuffers(buffers_); }

  const char* Name() const override { return name_; }
  int NumClasses() const override { return params_.numClasses; }
  int NumKeypoints() const override { return params_.numKeypoints; }

  void Enqueue(const float* modelOutput, const DecodeParams& d, DetectionOutput* output,
               cudaStream_t stream) override {
    PostprocessParams p = params_;
    p.confThreshold = d.confThreshold;
    p.iouThreshold = d.iouThreshold;
    p.offsetX = d.offsetX;
    p.offsetY = d.offsetY;
    PostprocessBuffers b = buffers_;
    b.output = output;
    LaunchPostprocess(modelOutput, p, d.letterbox, b, stream);
  }

 private:
  const char* name_;
  PostprocessParams params_;
  PostprocessBuffers buffers_;
};

// 端到端头 [1, K, 6]：模型内已完成抑制，只做阈值筛选
class End2EndDecoder final : public OutputDecoder {
 public:
  End2EndDecoder(int rows, int numClasses) : rows_(rows), numClasses_(numClasses) {}

  const char* Name() const override { return "yolo-end2end"; }
  int NumClasses() const override { return numClasses_; }
  int NumKeypoints() const override { return 0; }

  void Enqueue(const float* modelOutput, const DecodeParams& d, DetectionOutput* output,
               cudaStream_t stream) override {
    PostprocessParams p;
    p.confThreshold = d.confThreshold;
    p.offsetX = d.offsetX;
    p.offsetY = d.offsetY;
    LaunchEnd2EndPostprocess(modelOutput, rows_, p, d.letterbox, output, stream);
  }

 private:
  int rows_;
  int numClasses_;
};

bool AnchorShape(const std::vector<int64_t>& s) { return s.size() == 3 && s[0] == 1 && s[1] > 4 && s[2] > s[1]; }
bool End2EndShape(const std::vector<int64_t>& s) {
  return s.size() == 3 && s[0] == 1 && s[2] == 6 && s[1] <= kMaxCandidates;
}

}  // namespace

std::string ResolveDecoderName(const ModelManifest& m, const std::vector<int64_t>& shape) {
  if (!m.decoder.empty()) return m.decoder;
  if (AnchorShape(shape)) return "yolo-anchors";
  if (End2EndShape(shape)) return "yolo-end2end";
  throw std::runtime_error("无法按输出形状 " + ShapeText(shape) + " 推断解码器，请在模型清单中声明 decoder");
}

std::unique_ptr<OutputDecoder> CreateDecoder(const ModelManifest& m, const std::vector<int64_t>& shape) {
  const std::string name = ResolveDecoderName(m, shape);
  const int64_t c = shape.size() == 3 ? shape[1] : 0, n = shape.size() == 3 ? shape[2] : 0;
  const bool anchorShape = AnchorShape(shape);
  const bool end2endShape = End2EndShape(shape);

  const auto mismatch = [&](const char* expected) {
    return std::runtime_error("解码器 " + name + " 要求输出形状 " + expected + "，模型为 " + ShapeText(shape));
  };
  if (name == "yolo-anchors") {
    if (!anchorShape) throw mismatch("[1, 4 + nc, N]");
    return std::make_unique<AnchorDecoder>("yolo-anchors", static_cast<int>(c - 4), 0, static_cast<int>(n));
  }
  if (name == "yolo-pose-anchors") {
    const int k = m.numKeypoints;
    if (k <= 0) throw std::runtime_error("解码器 yolo-pose-anchors 需要在模型清单中声明 keypoints.count");
    if (!anchorShape || c != 4 + 1 + 3 * k) throw mismatch(("[1, 5 + 3 × " + std::to_string(k) + ", N]").c_str());
    return std::make_unique<AnchorDecoder>("yolo-pose-anchors", 1, k, static_cast<int>(n));
  }
  if (name == "yolo-end2end") {
    if (!end2endShape) throw mismatch("[1, K, 6]（K ≤ 1024）");
    return std::make_unique<End2EndDecoder>(static_cast<int>(c), static_cast<int>(m.classNames.size()));
  }
  throw std::runtime_error("未知的解码器：" + name);
}

}  // namespace nr
