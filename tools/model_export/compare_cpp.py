"""对比 nr_detect 离线模式与 Python 参考流水线。

1. 预处理：C++ 导出的输入张量 vs Python Letterbox（逐元素差异）。
2. 后处理：把 C++ 的输入张量送入同一引擎，Python 解码 + NMS 的结果 vs C++ 结果，
   单独检验 GPU 后处理。
3. 端到端：Python 预处理 + 推理 + 后处理 vs C++ 结果。

用法：python compare_cpp.py --list images.txt --json cpp.jsonl --dump-dir dumps --onnx model.onnx
"""
import argparse
import json
import pathlib

import cv2
import numpy as np

from eval_coco import match_rate
from reference_pipeline import INPUT_SIZE, Letterbox, TrtRtxBackend, postprocess

LSB = 1.0 / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    args = parser.parse_args()

    backend = TrtRtxBackend(args.onnx, engine_path=pathlib.Path(args.onnx).with_suffix(".py.engine"))
    rows = [json.loads(line) for line in open(args.json, encoding="utf-8")]

    pre_max, pre_frac, post_agree, e2e_agree, worst = [], [], [], [], []
    for row in rows:
        image = cv2.imread(row["image"])
        h, w = image.shape[:2]
        lb = Letterbox((h, w))
        py_tensor = lb.apply(image)
        cpp_tensor = np.fromfile(pathlib.Path(args.dump_dir) / (pathlib.Path(row["image"]).stem + ".bin"),
                                 dtype=np.float32).reshape(1, 3, INPUT_SIZE, INPUT_SIZE)
        diff = np.abs(cpp_tensor - py_tensor)
        pre_max.append(float(diff.max()))
        pre_frac.append(float((diff > 0.5 * LSB).mean()))

        det = np.array(row["det"], dtype=np.float32).reshape(-1, 6)
        cpp = (det[:, :4], det[:, 4], det[:, 5].astype(int))

        def reference(tensor):
            b, s, c = postprocess(backend(tensor), conf=args.conf, iou=args.iou, max_candidates=1024)
            return lb.restore(b, (h, w)), s, c

        post = match_rate(reference(cpp_tensor), cpp, iou_thr=0.99)[0]
        e2e = match_rate(reference(py_tensor), cpp, iou_thr=0.9)[0]
        post_agree.append(post)
        e2e_agree.append(e2e)
        worst.append((post, e2e, row["image"], len(det)))

    def stats(values):
        a = np.array(values)
        return f"均值 {a.mean():.5f}  最小 {a.min():.5f}  最大 {a.max():.5f}"

    print(f"图片数 {len(rows)}")
    print(f"预处理 最大绝对差（1 LSB = {LSB:.5f}）: {stats(pre_max)}")
    print(f"预处理 差异超过 0.5 LSB 的元素比例: {stats(pre_frac)}")
    print(f"后处理 一致率（同一输入，IoU>=0.99）: {stats(post_agree)}；低于 1 的图片 {sum(a < 1 for a in post_agree)}")
    print(f"端到端 一致率（IoU>=0.9）: {stats(e2e_agree)}；低于 0.99 的图片 {sum(a < 0.99 for a in e2e_agree)}")
    for post, e2e, name, n in sorted(worst)[:5]:
        print(f"  最差: 后处理 {post:.3f} 端到端 {e2e:.3f} 框数 {n} {name}")


if __name__ == "__main__":
    main()
