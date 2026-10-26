#include "inference/model_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace nr {
namespace {

using json = nlohmann::json;

std::string Lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::vector<std::filesystem::path> Candidates(const std::filesystem::path& onnx) {
  const std::string stem = onnx.stem().string();
  std::vector<std::filesystem::path> out = {onnx.parent_path() / (stem + ".json")};
  for (const char* p : {"fp16", "int8", "fp8"}) {
    if (const auto pos = Lower(stem).find(p); pos != std::string::npos) {
      std::string s = stem;
      s.replace(pos, std::strlen(p), "fp32");
      out.push_back(onnx.parent_path() / (s + ".json"));
    }
  }
  return out;
}

}  // namespace

ModelManifest LoadModelManifest(const std::filesystem::path& onnx) {
  ModelManifest m;
  for (const auto& path : Candidates(onnx)) {
    std::ifstream f(path);
    if (!f) continue;
    try {
      const json j = json::parse(f);
      m.source = path;
      m.task = j.value("task", "");
      m.decoder = j.value("decoder", "");
      m.head = j.value("head", "");
      m.imgsz = j.value("imgsz", 0);
      if (j.contains("outputs") && j["outputs"].is_array() && !j["outputs"].empty()) {
        for (const json& d : j["outputs"][0].value("shape", json::array())) {
          if (!d.is_number_integer()) {
            m.outputShape.clear();
            break;
          }
          m.outputShape.push_back(d.get<int64_t>());
        }
      }
      if (j.contains("class_names")) {
        for (const auto& [k, v] : j["class_names"].items()) m.classNames[std::stoi(k)] = v.get<std::string>();
      }
      if (j.contains("keypoints")) {
        const json& k = j["keypoints"];
        m.numKeypoints = k.value("count", 0);
        m.keypointNames = k.value("names", std::vector<std::string>{});
        m.skeleton = k.value("skeleton", std::vector<std::array<int, 2>>{});
      }
      if (j.contains("preprocess")) {
        const json& p = j["preprocess"];
        m.resize = p.value("resize", m.resize);
        m.color = p.value("color", m.color);
        m.scale = p.value("scale", m.scale);
      }
    } catch (const std::exception& e) {
      throw std::runtime_error("模型清单格式错误（" + path.string() + "）：" + e.what());
    }
    if (m.numKeypoints < 0 || m.numKeypoints > 32) throw std::runtime_error("模型清单的关键点数超出范围 0–32");
    return m;
  }
  return m;
}

void CheckSupportedPreprocess(const ModelManifest& m) {
  if (m.resize != "letterbox" || m.color != "rgb" || std::abs(m.scale - 1.0 / 255.0) > 1e-9) {
    throw std::runtime_error("模型清单声明的预处理（" + m.resize + "，" + m.color +
                             "）不受支持；当前只支持 letterbox、rgb、1/255");
  }
}

}  // namespace nr
