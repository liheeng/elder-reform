# -*- coding: utf-8 -*-
"""
eval_hard_cases.py

在 `hard_cases/` 难例集上评测姿态模型的**检出能力**，用于回答：
"这个权重在夜间/倒地/遮挡场景上，比上一个权重好还是差？"

真值来自 `hard_cases/labels.csv`：
    person_present  1=画面里有人 0=没人（必填）
    x1,y1,x2,y2     可选，填了就能额外算定位精度

同时报两套数字，其差值即**质量门（`_is_valid_person`）的代价**：
    裸 YOLO        conf 直接取 --raw-conf，不过任何门
    完整 pipeline  走 `PoseExtractor`（conf=0.05 + 三重质量门 + 重叠合并）

指标：
    正样本召回率   有人且标注 person_present=1 的帧中，被检出（至少 1 个目标）的比例
    召回@IoU0.5    有 GT bbox 的正样本中，best IoU >= --iou-thres 的比例
    负样本误报率   标注 person_present=0 的帧中，仍被检出目标的比例

用法：
    python scripts/eval/eval_hard_cases.py
    python scripts/eval/eval_hard_cases.py --model models/yolo26n-pose.pt
    python scripts/eval/eval_hard_cases.py --models a.pt b.pt        # 多权重对比
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pose.pose_extractor import PoseExtractor  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from utils.device import get_device  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo26n-pose.pt"
DEFAULT_DIR = PROJECT_ROOT / "hard_cases"


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def load_labels(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            val = (row.get("person_present") or "").strip()
            if val not in ("0", "1"):
                continue
            gt = None
            try:
                if all((row.get(k) or "").strip() for k in ("x1", "y1", "x2", "y2")):
                    gt = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
            except ValueError:
                gt = None
            rows.append({
                "video": row["video"], "frame": int(row["frame"]),
                "present": int(val), "gt": gt,
            })
    return rows


def best_iou_vs_gt(boxes, gt):
    if gt is None or len(boxes) == 0:
        return 0.0
    return max(iou(b, gt) for b in boxes)


def evaluate_model(model_path, labels, frames_root, raw_conf, iou_thres, device):
    """返回该模型在难例集上的统计。"""
    extractor = PoseExtractor(model_path=str(model_path))
    raw_model = YOLO(str(model_path))

    stat = {
        "pos": 0, "pos_hit_pipe": 0, "pos_hit_raw": 0,
        "neg": 0, "neg_fp_pipe": 0, "neg_fp_raw": 0,
        "gt_n": 0, "gt_hit_pipe": 0, "gt_hit_raw": 0,
        "iou_pipe": [], "iou_raw": [],
        "misses": [], "per_video": defaultdict(lambda: [0, 0, 0]),
    }

    cache = {}
    for row in labels:
        key = (row["video"], row["frame"])
        img_path = frames_root / row["video"] / ("f%06d.jpg" % row["frame"])
        if not img_path.exists():
            continue
        if key not in cache:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            persons = extractor.extract(img)
            pipe_boxes = [tuple(float(v) for v in p["bbox"]) for p in persons]
            res = raw_model(img, verbose=False, conf=raw_conf, iou=0.5, device=device)
            rb = res[0].boxes
            raw_boxes = ([tuple(float(v) for v in b) for b in rb.xyxy.cpu().numpy()]
                         if rb is not None and len(rb) > 0 else [])
            cache[key] = (pipe_boxes, raw_boxes)

        pipe_boxes, raw_boxes = cache[key]
        pv = stat["per_video"][row["video"]]

        if row["present"] == 1:
            stat["pos"] += 1
            pv[0] += 1
            hit_pipe = len(pipe_boxes) > 0
            hit_raw = len(raw_boxes) > 0
            stat["pos_hit_pipe"] += int(hit_pipe)
            stat["pos_hit_raw"] += int(hit_raw)
            pv[1] += int(hit_pipe)
            if not hit_pipe:
                stat["misses"].append((row["video"], row["frame"]))
            if row["gt"] is not None:
                stat["gt_n"] += 1
                i_pipe = best_iou_vs_gt(pipe_boxes, row["gt"])
                i_raw = best_iou_vs_gt(raw_boxes, row["gt"])
                stat["iou_pipe"].append(i_pipe)
                stat["iou_raw"].append(i_raw)
                stat["gt_hit_pipe"] += int(i_pipe >= iou_thres)
                stat["gt_hit_raw"] += int(i_raw >= iou_thres)
        else:
            stat["neg"] += 1
            pv[2] += 1
            stat["neg_fp_pipe"] += int(len(pipe_boxes) > 0)
            stat["neg_fp_raw"] += int(len(raw_boxes) > 0)

    return stat


def pct(a, b):
    return "%d/%d (%.1f%%)" % (a, b, 100.0 * a / b) if b else "0/0 (-)"


def main():
    ap = argparse.ArgumentParser(description="难例集评测")
    ap.add_argument("--models", nargs="*", default=None,
                    help="待评测权重（默认当前 models/yolo26n-pose.pt）")
    ap.add_argument("--hard-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--raw-conf", type=float, default=0.01)
    ap.add_argument("--iou-thres", type=float, default=0.5)
    args = ap.parse_args()

    hard = Path(args.hard_dir)
    labels = load_labels(hard / "labels.csv")
    models = [Path(m) for m in (args.models or [DEFAULT_MODEL])]

    n_pos = sum(1 for r in labels if r["present"] == 1)
    n_neg = sum(1 for r in labels if r["present"] == 0)
    n_gt = sum(1 for r in labels if r["present"] == 1 and r["gt"] is not None)
    print("难例集: 已标注 %d 帧（正样本 %d / 负样本 %d，其中 %d 帧带 bbox）"
          % (len(labels), n_pos, n_neg, n_gt))
    if not labels:
        print("labels.csv 还没有人工标注（person_present 需填 1/0）")
        return

    device = get_device()
    rows = []
    for model in models:
        if not model.exists():
            print("跳过不存在的权重: %s" % model)
            continue
        st = evaluate_model(model, labels, hard / "frames", args.raw_conf, args.iou_thres, device)
        rows.append((model.name, st))

        print()
        print("=== %s ===" % model.name)
        print("  正样本召回率     裸YOLO %-18s 完整pipeline %s"
              % (pct(st["pos_hit_raw"], st["pos"]), pct(st["pos_hit_pipe"], st["pos"])))
        if st["gt_n"]:
            print("  召回@IoU>=%.1f   裸YOLO %-18s 完整pipeline %s"
                  % (args.iou_thres, pct(st["gt_hit_raw"], st["gt_n"]),
                     pct(st["gt_hit_pipe"], st["gt_n"])))
            print("  平均 best IoU    裸YOLO %.3f             完整pipeline %.3f"
                  % (float(np.mean(st["iou_raw"])), float(np.mean(st["iou_pipe"]))))
        print("  负样本误报率     裸YOLO %-18s 完整pipeline %s"
              % (pct(st["neg_fp_raw"], st["neg"]), pct(st["neg_fp_pipe"], st["neg"])))
        if st["misses"]:
            print("  pipeline 漏检帧 (%d): %s" % (len(st["misses"]),
                  ", ".join("%s#%d" % (v, f) for v, f in st["misses"][:12])))
        print("  分视频（正样本 pipeline 召回）:")
        for vid in sorted(st["per_video"]):
            p, hit, neg = st["per_video"][vid]
            print("    %-28s 正样本 %-14s 负样本 %d" % (vid, pct(hit, p), neg))

    if len(rows) > 1:
        print()
        print("=== 对比汇总 ===")
        print("%-26s %-16s %-16s %s" % ("权重", "正样本召回(裸)", "正样本召回(pipe)", "召回@IoU"))
        for name, st in rows:
            print("%-26s %-16s %-16s %s"
                  % (name, pct(st["pos_hit_raw"], st["pos"]),
                     pct(st["pos_hit_pipe"], st["pos"]),
                     pct(st["gt_hit_pipe"], st["gt_n"])))


if __name__ == "__main__":
    main()
