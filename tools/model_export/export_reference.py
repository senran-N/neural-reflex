"""导出 Ultralytics 模型为 ONNX，并写出模型清单（设计方案 7.2；扩展机制方案 4.1）。

用法：python export_reference.py [--weights yolo26n.pt] [--out models/reference] [--end2end]
默认导出需要 NMS 的一对多检测头；--end2end 导出无 NMS 的一对一头作为对照配置。
姿态模型（如 yolo26n-pose.pt）导出一对多头，清单中附带关键点名称与连线。
"""
import argparse
import hashlib
import json
import pathlib
import shutil

import onnx
import yaml
from ultralytics import YOLO

DEFAULT_WEIGHTS = "yolo26n.pt"
IMGSZ = 640
OPSET = 17

# COCO 17 点的连线，来自 ultralytics.utils.plotting.Annotator（原为 1 起编号，这里改为 0 起）
COCO_SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
                 [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def tensor_info(value):
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in value.type.tensor_type.shape.dim]
    dtype = onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
    return {"name": value.name, "shape": dims, "dtype": dtype}


def keypoint_info(model):
    count, dims = model.model.kpt_shape
    if dims != 3:
        raise SystemExit(f"仅支持每点 (x, y, score) 的关键点，模型为 {dims} 维")
    info = {"count": count}
    if count == 17:
        coco = pathlib.Path(__import__("ultralytics").__file__).parent / "cfg" / "datasets" / "coco-pose.yaml"
        info["names"] = yaml.safe_load(coco.read_text(encoding="utf-8"))["kpt_names"][0]
        info["skeleton"] = COCO_SKELETON
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", default="models/reference")
    parser.add_argument("--end2end", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    pose = model.task == "pose"
    if model.task not in ("detect", "pose"):
        raise SystemExit(f"暂不支持的任务：{model.task}")
    if pose and args.end2end:
        raise SystemExit("姿态模型暂只支持一对多头")
    exported = model.export(format="onnx", imgsz=IMGSZ, opset=OPSET, batch=1, dynamic=False,
                            simplify=True, half=False, nms=False, end2end=args.end2end)
    suffix = "e2e" if args.end2end else "nms"
    target = out_dir / f"{pathlib.Path(args.weights).stem}-{IMGSZ}-{suffix}-fp32.onnx"
    shutil.move(exported, target)

    graph = onnx.load(str(target)).graph
    contract = {
        "source_weights": pathlib.Path(args.weights).name,
        "source_weights_sha256": sha256(args.weights),
        "onnx": target.name,
        "onnx_sha256": sha256(target),
        "opset": OPSET,
        "imgsz": IMGSZ,
        "task": model.task,
        "decoder": "yolo-end2end" if args.end2end else ("yolo-pose-anchors" if pose else "yolo-anchors"),
        "head": "one-to-one (end2end, no NMS)" if args.end2end else "one-to-many (needs NMS)",
        "class_names": model.names,
        "inputs": [tensor_info(v) for v in graph.input],
        "outputs": [tensor_info(v) for v in graph.output],
    }
    if pose:
        contract["keypoints"] = keypoint_info(model)
    contract_path = target.with_suffix(".json")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: contract[k] for k in ("onnx", "decoder", "inputs", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
