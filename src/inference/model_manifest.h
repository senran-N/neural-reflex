#pragma once

// 模型清单：模型文件旁的同名 JSON（扩展机制方案 4.1）。

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nr {

struct ModelManifest {
  std::filesystem::path source;  // 实际读取的清单文件，没有清单时为空
  std::string task;              // detect | pose，未声明时为空
  std::string decoder;           // 解码器名称，未声明时为空（按输出形状推断）
  std::string head;              // 导出脚本记录的检测头说明
  int imgsz = 0;
  std::vector<int64_t> outputShape;  // 导出时记录的第一个输出形状，未记录或含动态维度时为空
  std::map<int, std::string> classNames;
  int numKeypoints = 0;
  std::vector<std::string> keypointNames;
  std::vector<std::array<int, 2>> skeleton;
  // 预处理声明；当前内核只实现缺省组合（letterbox、rgb、1/255）
  std::string resize = "letterbox";
  std::string color = "rgb";
  double scale = 1.0 / 255.0;
};

// 查找顺序：model.json；文件名含 fp16/int8/fp8 时再找对应的 fp32 清单。
// 清单不存在时返回缺省值；清单存在但格式错误时抛出异常。
ModelManifest LoadModelManifest(const std::filesystem::path& onnx);

// 预处理声明不是当前内核支持的组合时抛出异常
void CheckSupportedPreprocess(const ModelManifest& manifest);

}  // namespace nr
