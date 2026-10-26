"""对比 nr_detect 离线模式与 Python 参考流水线。

1. 预处理：C++ 导出的输入张量 vs Python Letterbox（逐元素差异）。
2. 后处理：把 C++ 的输入张量送入同一引擎，Python 解码 + NMS 的结果 vs C++ 结果，
   单独检验 GPU 后处理。
3. 端到端：Python 预处理 + 推理 + 后处理 vs C++ 结果。

解码方式按模型清单的 decoder 字段选择（扩展机制方案 4.4）；姿态模型另比较关键点。

用法：python compare_cpp.py --json cpp.jsonl --dump-dir dumps --onnx model.onnx
"""
import argparse
import json
import pathlib

import cv2
import numpy as np

from eval_coco import match_rate
from reference_pipeline import (INPUT_SIZE, Letterbox, TrtRtxBackend, decoder_name, load_manifest, postprocess,
                                postprocess_end2end, postprocess_pose)

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
    manifest = load_manifest(args.onnx)
    decoder = decoder_name(manifest, backend(np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), np.float32)).shape)
    num_kpt = manifest.get("keypoints", {}).get("count", 0)
    print(f"解码器 {decoder}" + (f"，关键点 {num_kpt}" if num_kpt else ""))

    pre_max, pre_frac, post_agree, e2e_agree, worst, kpt_diff = [], [], [], [], [], []
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

        def reference(tensor, keypoints=False):
            out = backend(tensor)
            if decoder == "yolo-end2end":
                b, s, c = postprocess_end2end(out, conf=args.conf)
            elif decoder == "yolo-pose-anchors":
                b, s, c, k = postprocess_pose(out, num_kpt, conf=args.conf, iou=args.iou, max_candidates=1024)
                if keypoints:
                    return lb.restore(b, (h, w)), s, c, lb.restore_points(k, (h, w))
            else:
                b, s, c = postprocess(out, conf=args.conf, iou=args.iou, max_candidates=1024)
            return lb.restore(b, (h, w)), s, c

        if num_kpt:
            # 同一输入下，C++ 每个目标与 Python 中同类别、IoU 最大的目标比较关键点
            rb, rs, rc, rk = reference(cpp_tensor, keypoints=True)
            ck = np.array(row.get("kpt", []), dtype=np.float32).reshape(-1, num_kpt, 3)
            for i in range(len(det)):
                if len(rb) == 0:
                    break
                x1 = np.maximum(det[i, 0], rb[:, 0]); y1 = np.maximum(det[i, 1], rb[:, 1])
                x2 = np.minimum(det[i, 2], rb[:, 2]); y2 = np.minimum(det[i, 3], rb[:, 3])
                inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                area = (det[i, 2] - det[i, 0]) * (det[i, 3] - det[i, 1])
                iou = inter / np.maximum(area + (rb[:, 2] - rb[:, 0]) * (rb[:, 3] - rb[:, 1]) - inter, 1e-9)
                j = int(iou.argmax())
                if iou[j] >= 0.99:
                    kpt_diff.append((float(np.abs(ck[i, :, :2] - rk[j, :, :2]).max()),
                                     float(np.abs(ck[i, :, 2] - rk[j, :, 2]).max())))

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
    if kpt_diff:
        k = np.array(kpt_diff)
        print(f"关键点（同一输入、已匹配目标 {len(k)} 个）: 坐标最大差 均值 {k[:, 0].mean():.5f} 最大 {k[:, 0].max():.5f} px；"
              f"分数最大差 {k[:, 1].max():.6f}")
    for post, e2e, name, n in sorted(worst)[:5]:
        print(f"  最差: 后处理 {post:.3f} 端到端 {e2e:.3f} 框数 {n} {name}")


if __name__ == "__main__":
    main()
