"""参考模型的 Python 参考流水线：预处理、推理后端、解码与 NMS。

C++/CUDA 实现的预处理和后处理以此为准做对比验证（设计方案 5.1、7.4）。
输出契约（一对多头）：[1, 4 + nc, N]，前 4 行为输入像素坐标下的 cx, cy, w, h，
其余为各类别已经过 sigmoid 的置信度。
"""
import ctypes
import pathlib

import cv2
import numpy as np
import torch
import torchvision

INPUT_SIZE = 640
PAD_VALUE = 114


class Letterbox:
    """等比例缩放并居中填充，记录还原参数。"""

    def __init__(self, image_hw, size=INPUT_SIZE):
        h, w = image_hw
        self.scale = min(size / h, size / w)
        self.new_w, self.new_h = round(w * self.scale), round(h * self.scale)
        dw, dh = (size - self.new_w) / 2, (size - self.new_h) / 2
        self.top, self.left = int(round(dh - 0.1)), int(round(dw - 0.1))
        self.bottom = size - self.new_h - self.top
        self.right = size - self.new_w - self.left
        self.size = size

    def apply(self, image_bgr):
        resized = cv2.resize(image_bgr, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, self.top, self.bottom, self.left, self.right,
                                    cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
        rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    def restore(self, boxes_xyxy, image_hw):
        boxes = boxes_xyxy.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - self.left) / self.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - self.top) / self.scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_hw[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_hw[0])
        return boxes


def postprocess(output, conf=0.25, iou=0.7, max_det=300, max_candidates=None):
    """单标签、按类别 NMS。返回 (boxes_xyxy[N,4], scores[N], classes[N])，坐标为模型输入像素。"""
    pred = output[0]  # [4 + nc, N]
    scores_all = pred[4:]
    classes = scores_all.argmax(0)
    scores = scores_all[classes, np.arange(pred.shape[1])]
    keep = scores > conf
    if max_candidates is not None and keep.sum() > max_candidates:
        # 与固定容量 GPU 缓冲一致：只保留置信度最高的候选
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-scores[idx])[:max_candidates]]
        keep = np.zeros_like(keep)
        keep[idx] = True
    cx, cy, w, h = pred[0, keep], pred[1, keep], pred[2, keep], pred[3, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float32)
    scores, classes = scores[keep].astype(np.float32), classes[keep]
    if len(scores) == 0:
        return boxes, scores, classes
    kept = torchvision.ops.batched_nms(torch.from_numpy(boxes), torch.from_numpy(scores),
                                       torch.from_numpy(classes), iou)[:max_det].numpy()
    return boxes[kept], scores[kept], classes[kept]


class OrtBackend:
    name = "onnxruntime-cpu"

    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, tensor):
        return self.session.run(None, {self.input: tensor})[0]


class TrtRtxBackend:
    """TensorRT for RTX，输入输出保持 FP32，内部精度由 ONNX 图决定（强类型）。"""

    def __init__(self, onnx_path, engine_path=None):
        import tensorrt_rtx as trt
        from cuda.bindings import runtime as cudart
        self.trt, self.cudart = trt, cudart
        self.name = f"tensorrt-rtx:{pathlib.Path(onnx_path).stem}"
        logger = trt.Logger(trt.Logger.WARNING)

        engine_path = pathlib.Path(engine_path) if engine_path else pathlib.Path(onnx_path).with_suffix(".engine")
        if engine_path.exists() and engine_path.stat().st_mtime >= pathlib.Path(onnx_path).stat().st_mtime:
            serialized = engine_path.read_bytes()
        else:
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(pathlib.Path(onnx_path).read_bytes()):
                raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            serialized = builder.build_serialized_network(network, builder.create_builder_config())
            if serialized is None:
                raise RuntimeError("引擎构建失败")
            engine_path.write_bytes(bytes(serialized))

        self.engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        self.stream = self._check(cudart.cudaStreamCreate())
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = np.empty(shape, dtype=dtype)
            device = self._check(cudart.cudaMalloc(host.nbytes))
            self.context.set_tensor_address(name, int(device))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.buffers[name] = (host, device, is_input)

    def _check(self, result):
        err, *rest = result if isinstance(result, tuple) else (result,)
        if int(err) != 0:
            raise RuntimeError(f"CUDA 错误 {err}")
        return rest[0] if len(rest) == 1 else rest

    def __call__(self, tensor):
        cudart = self.cudart
        h2d, d2h = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        for host, device, is_input in self.buffers.values():
            if is_input:
                np.copyto(host, tensor)
                self._check(cudart.cudaMemcpyAsync(device, host.ctypes.data, host.nbytes, h2d, self.stream))
        if not self.context.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 失败")
        outputs = []
        for host, device, is_input in self.buffers.values():
            if not is_input:
                self._check(cudart.cudaMemcpyAsync(host.ctypes.data, device, host.nbytes, d2h, self.stream))
                outputs.append(host)
        self._check(cudart.cudaStreamSynchronize(self.stream))
        return outputs[0].copy()


def make_backend(kind, onnx_path):
    if kind == "ort":
        return OrtBackend(onnx_path)
    if kind == "trt":
        return TrtRtxBackend(onnx_path)
    raise ValueError(kind)
