"""导出参考基准模型（设计方案 7.2）为 ONNX，并打印输入输出契约。

用法：python export_reference.py [--out models/reference] [--end2end]
默认导出需要 NMS 的一对多检测头；--end2end 导出无 NMS 的一对一头作为对照配置。
"""
import argparse
import hashlib
import json
import pathlib
import shutil

import onnx
from ultralytics import YOLO

MODEL = "yolo26n.pt"
IMGSZ = 640
OPSET = 17


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def tensor_info(value):
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in value.type.tensor_type.shape.dim]
    dtype = onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type)
    return {"name": value.name, "shape": dims, "dtype": dtype}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/reference")
    parser.add_argument("--end2end", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL)
    exported = model.export(format="onnx", imgsz=IMGSZ, opset=OPSET, batch=1, dynamic=False,
                            simplify=True, half=False, nms=False, end2end=args.end2end)
    suffix = "e2e" if args.end2end else "nms"
    target = out_dir / f"yolo26n-{IMGSZ}-{suffix}-fp32.onnx"
    shutil.move(exported, target)

    graph = onnx.load(str(target)).graph
    contract = {
        "source_weights": MODEL,
        "source_weights_sha256": sha256(MODEL),
        "onnx": target.name,
        "onnx_sha256": sha256(target),
        "opset": OPSET,
        "imgsz": IMGSZ,
        "head": "one-to-one (end2end, no NMS)" if args.end2end else "one-to-many (needs NMS)",
        "class_names": model.names,
        "inputs": [tensor_info(v) for v in graph.input],
        "outputs": [tensor_info(v) for v in graph.output],
    }
    contract_path = target.with_suffix(".json")
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: contract[k] for k in ("onnx", "head", "inputs", "outputs")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
