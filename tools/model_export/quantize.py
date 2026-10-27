"""参考模型的低比特量化（设计方案 3.2）：用 NVIDIA ModelOpt 插入显式 Q/DQ 节点，
TensorRT for RTX 按强类型网络执行。

校准集取 COCO val2017 按 id 排序后的最后 --calib 张；精度评估用前 4500 张
（eval_coco.py --limit 4500），两者不重叠。

用法：python quantize.py --coco D:/SDK/datasets/coco --onnx models/reference/yolo26n-640-nms-fp32.onnx \
          --mode int8 --method entropy --out models/reference/yolo26n-640-nms-int8.onnx
"""
import argparse
import contextlib
import io
import pathlib
import subprocess
import sys

import cv2
import numpy as np
from pycocotools.coco import COCO

from reference_pipeline import Letterbox

EVAL_IMAGES = 4500


def build_calibration(coco_root, count, path):
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(coco_root / "annotations" / "instances_val2017.json"))
    ids = sorted(gt.getImgIds())[EVAL_IMAGES:EVAL_IMAGES + count]
    batch = []
    for image_id in ids:
        image = cv2.imread(str(coco_root / "val2017" / gt.loadImgs(image_id)[0]["file_name"]))
        batch.append(Letterbox(image.shape[:2]).apply(image)[0])
    data = np.stack(batch).astype(np.float32)
    np.save(path, data)
    return data.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mode", choices=["int8", "fp8"], required=True)
    ap.add_argument("--method", default="entropy", choices=["max", "entropy"])
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-head", action="store_true", help="检测头最后的解码运算保持高精度")
    args = ap.parse_args()

    calib = pathlib.Path(args.out).with_suffix(f".calib{args.calib}.npy")
    if not calib.exists():
        print("校准数据", build_calibration(pathlib.Path(args.coco), args.calib, calib), flush=True)

    cmd = [sys.executable, "-m", "modelopt.onnx.quantization", "--onnx_path", args.onnx,
           "--quantize_mode", args.mode, "--calibration_method", args.method,
           "--calibration_data_path", str(calib), "--calibration_eps", "cpu",
           "--high_precision_dtype", "fp16", "--output_path", args.out]
    if args.exclude_head:
        # 一对多检测头末端的坐标解码数值范围大（见 FP16 转换日志），量化误差直接落在框坐标上
        cmd += ["--nodes_to_exclude", "/model.23/.*"]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
