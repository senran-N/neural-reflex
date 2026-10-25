"""在 COCO val2017 上比较推理后端与精度：mAP、原始输出偏差、检测结果一致性。

用法：
  python eval_coco.py --coco D:/SDK/datasets/coco --limit 5000 \
      --run ort:models/reference/yolo26n-640-nms-fp32.onnx \
      --run trt:models/reference/yolo26n-640-nms-fp32.onnx \
      --run trt:models/reference/yolo26n-640-nms-fp16.onnx

第一个 --run 作为基准，其余与它逐图比较。mAP 使用单标签、按类别 NMS 的
运行时后处理（conf=0.001, iou=0.7, max_det=300），与 Ultralytics 官方验证流程
（多标签、矩形批处理）不同，数值不能直接与官方表格比较，只用于配置间对比。
"""
import argparse
import contextlib
import io
import json
import pathlib
import time

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from reference_pipeline import Letterbox, make_backend, postprocess

# COCO 80 类索引 → 原始 91 类 category_id
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
                85, 86, 87, 88, 89, 90]


def match_rate(a, b, iou_thr=0.9):
    """b 中能与 a 同类别且 IoU>=iou_thr 一一对应的检测比例（以 conf>=0.25 的结果计）。"""
    (ba, sa, ca), (bb, sb, cb) = a, b
    ba, ca = ba[sa >= 0.25], ca[sa >= 0.25]
    bb, cb = bb[sb >= 0.25], cb[sb >= 0.25]
    if len(ba) == 0 and len(bb) == 0:
        return 1.0, 0, 0
    used = np.zeros(len(bb), bool)
    matched = 0
    for box, cls in zip(ba, ca):
        if len(bb) == 0:
            break
        x1 = np.maximum(box[0], bb[:, 0]); y1 = np.maximum(box[1], bb[:, 1])
        x2 = np.minimum(box[2], bb[:, 2]); y2 = np.minimum(box[3], bb[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        union = (box[2] - box[0]) * (box[3] - box[1]) + (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1]) - inter
        iou = np.where((cb == cls) & ~used, inter / np.maximum(union, 1e-9), 0)
        j = int(iou.argmax())
        if iou[j] >= iou_thr:
            used[j] = True
            matched += 1
    return matched / max(len(ba), len(bb)), len(ba), len(bb)


def coco_map(gt, image_ids, detections):
    if not detections:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(detections), "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return round(ev.stats[0] * 100, 2), round(ev.stats[1] * 100, 2)


def evaluate_cpp(gt, image_ids, jsonl):
    by_name = {gt.loadImgs(i)[0]["file_name"]: i for i in image_ids}
    detections, seen, overflow, times = [], set(), 0, []
    for line in open(jsonl, encoding="utf-8"):
        row = json.loads(line)
        image_id = by_name.get(pathlib.Path(row["image"]).name)
        if image_id is None:
            continue
        seen.add(image_id)
        overflow += row["overflow"] > 0
        times.append(row["times"])
        for x1, y1, x2, y2, score, cls in row["det"]:
            detections.append({"image_id": image_id, "category_id": COCO80_TO_91[int(cls)],
                               "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    ids = sorted(seen)
    m, m50 = coco_map(gt, ids, detections)
    t = np.median(np.array(times[10:] or times), axis=0)
    print(json.dumps({"images": len(ids), "mAP50_95": m, "mAP50": m50, "overflow_images": int(overflow),
                      "median_gpu_ms_pre_infer_post": [round(float(v), 4) for v in t]}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--run", action="append", help="后端:onnx 路径，后端为 ort 或 trt")
    parser.add_argument("--cpp-jsonl", help="直接评估 nr_detect --image-list 的输出（需以 --conf 0.001 运行）")
    parser.add_argument("--write-list", help="把前 --limit 张图片的路径写入该文件后退出")
    parser.add_argument("--out", default="cache/eval")
    args = parser.parse_args()

    coco_root = pathlib.Path(args.coco)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(coco_root / "annotations" / "instances_val2017.json"))
    image_ids = sorted(gt.getImgIds())[: args.limit]

    if args.write_list:
        paths = [str((coco_root / "val2017" / gt.loadImgs(i)[0]["file_name"]).resolve()) for i in image_ids]
        pathlib.Path(args.write_list).write_text("\n".join(paths) + "\n", encoding="utf-8")
        return
    if args.cpp_jsonl:
        evaluate_cpp(gt, image_ids, args.cpp_jsonl)
        return

    runs = []
    for spec in args.run or []:
        kind, path = spec.split(":", 1)
        runs.append(make_backend(kind, path))
    names = [r.name for r in runs]

    detections = {n: [] for n in names}
    max_abs = {n: 0.0 for n in names[1:]}
    agreement = {n: [] for n in names[1:]}
    infer_time = {n: 0.0 for n in names}

    for k, image_id in enumerate(image_ids):
        info = gt.loadImgs(image_id)[0]
        image = cv2.imread(str(coco_root / "val2017" / info["file_name"]))
        lb = Letterbox(image.shape[:2])
        tensor = lb.apply(image)

        outputs, results = [], []
        for run in runs:
            t0 = time.perf_counter()
            out = run(tensor)
            infer_time[run.name] += time.perf_counter() - t0
            outputs.append(out)
            boxes, scores, classes = postprocess(out, conf=0.001, iou=0.7, max_det=300)
            results.append((boxes, scores, classes))
            restored = lb.restore(boxes, image.shape[:2])
            for b, s, c in zip(restored, scores, classes):
                detections[run.name].append({
                    "image_id": image_id, "category_id": COCO80_TO_91[int(c)],
                    "bbox": [round(float(b[0]), 3), round(float(b[1]), 3),
                             round(float(b[2] - b[0]), 3), round(float(b[3] - b[1]), 3)],
                    "score": round(float(s), 5)})

        for i in range(1, len(runs)):
            n = names[i]
            max_abs[n] = max(max_abs[n], float(np.abs(outputs[i] - outputs[0]).max()))
            agreement[n].append(match_rate(results[0], results[i])[0])

        if (k + 1) % 500 == 0:
            print(f"{k + 1}/{len(image_ids)}", flush=True)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"images": len(image_ids), "runs": {}}
    for n in names:
        m, m50 = coco_map(gt, image_ids, detections[n])
        entry = {"mAP50_95": m, "mAP50": m50,
                 "mean_infer_ms_incl_copy": round(infer_time[n] / len(image_ids) * 1000, 2)}
        if n in max_abs:
            entry["max_abs_output_diff"] = round(max_abs[n], 5)
            entry["detection_agreement_iou0.9"] = round(float(np.mean(agreement[n])), 4)
            entry["images_below_99pct_agreement"] = int(np.sum(np.array(agreement[n]) < 0.99))
        summary["runs"][n] = entry

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
