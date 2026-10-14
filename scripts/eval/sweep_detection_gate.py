# -*- coding: utf-8 -*-
"""
sweep_detection_gate.py

在 `hard_cases/` 难例集上扫描**检出链的两个阈值**，输出"召回 vs 误报"曲线，
用于判断放宽 `PoseExtractor` 的过滤到底值不值。

两个待扫的阈值（都在 `pose/pose_extractor.py`）：
    1. `--conf`：YOLO 推理时的 bbox 置信度下限（生产值 0.05）
    2. 关键点平均置信度门 `min_mean_conf`（生产值 0.30，见 `_is_valid_person`
       与 `extract()` 里那条重复的硬过滤）

做法：所有帧只推理一次（用很低的 `--infer-conf` 拿全候选框），
再**复用生产代码里的 `_is_valid_person` / `_merge_overlapping_detections`**
离线组合出各阈值下的结果，避免重写过滤逻辑导致与线上不一致。

用法：
    python scripts/eval/sweep_detection_gate.py
    python scripts/eval/sweep_detection_gate.py --conf-list 0.05 0.02 0.01 --kp-list 0.30 0.20
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pose.pose_extractor import _is_valid_person, _merge_overlapping_detections  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from utils.device import get_device  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo26n-pose.pt"
DEFAULT_DIR = PROJECT_ROOT / "hard_cases"


def load_labels(hard_dir):
    rows = []
    with open(hard_dir / "labels.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            v = (r.get("person_present") or "").strip()
            if v in ("0", "1"):
                rows.append((r["video"], int(r["frame"]), int(v)))
    return rows


def gate(boxes, scores, kpts_xy, kpts_conf, score_thr, min_mean_conf,
         weak_floor=None, weak_box_score=None):
    """复刻 PoseExtractor.extract 的过滤链。

    传入 weak_floor / weak_box_score 时按生产的两段式判定；
    weak_floor=None 时退化为"单一 min_mean_conf 下限"（用 weak_box_score>1 模拟），
    以便对比历史规则。

    注意别再另写一份"平均关键点置信度"过滤 —— 那正是 2026-10 修掉的重复实现
    （extract() 里曾有一份硬编码 0.3，与 _is_valid_person 各改各的）。
    """
    if weak_floor is None:
        weak_floor = min_mean_conf
        weak_box_score = 2.0        # > 1 即"任何 score 都要求达标" == 单一阈值

    keep = []
    for b, s, kp, kc in zip(boxes, scores, kpts_xy, kpts_conf):
        if float(s) < score_thr:
            continue
        if not _is_valid_person(kp, kc, b, score=float(s),
                                min_mean_conf=min_mean_conf,
                                weak_kpts_floor=weak_floor,
                                weak_box_score=weak_box_score):
            continue
        keep.append({"bbox": b, "keypoints": kp, "score": float(s), "kpts_conf": kc})
    return _merge_overlapping_detections(keep, iou_threshold=0.3)



def main():
    ap = argparse.ArgumentParser(description="扫描检出链阈值")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--hard-dir", default=str(DEFAULT_DIR))
    ap.add_argument("--infer-conf", type=float, default=0.01,
                    help="一次推理用的低阈值（要 <= 所有 --conf-list 取值）")
    ap.add_argument("--conf-list", type=float, nargs="*", default=[0.05, 0.02, 0.01])
    ap.add_argument("--kp-list", type=float, nargs="*", default=[0.30, 0.20, 0.15])
    ap.add_argument("--list-fp", action="store_true", help="列出各配置下的误报帧")
    args = ap.parse_args()

    hard = Path(args.hard_dir)
    labels = load_labels(hard)
    if not labels:
        print("labels.csv 还没有人工标注（person_present 需填 1/0）")
        return
    n_pos = sum(1 for _, _, p in labels if p == 1)
    n_neg = len(labels) - n_pos

    model = YOLO(args.model)
    device = get_device()
    cache = {}
    for v, f, _ in labels:
        img = cv2.imread(str(hard / "frames" / v / ("f%06d.jpg" % f)))
        if img is None:
            continue
        r = model(img, verbose=False, conf=args.infer_conf, iou=0.5, device=device)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            cache[(v, f)] = (np.zeros((0, 4)), np.zeros(0), np.zeros((0, 2)), np.zeros((0, 17)))
        else:
            cache[(v, f)] = (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                             r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy())

    # 三种规则：历史单一阈值 / 朴素下调 / 生产复合门
    RULES = [
        ("单一 0.30 (2026-10 前)", dict(min_mean_conf=0.30)),
        ("单一 0.20 (朴素下调)", dict(min_mean_conf=0.20)),
        ("复合门 (当前生产)", dict(min_mean_conf=0.30, weak_floor=0.20, weak_box_score=0.30)),
    ]
    print("阈值扫描：难例集 %d 帧（正 %d / 负 %d），推理用 conf=%.3f"
          % (len(labels), n_pos, n_neg, args.infer_conf))
    print()
    print("%-24s %-8s %-22s %s" % ("规则", "conf", "正样本召回", "负样本误报"))
    for name, kw in RULES:
        for score_thr in args.conf_list:
            hit = fp = 0
            fp_frames = []
            for v, f, present in labels:
                bx, sc, xy, kc = cache.get((v, f), (np.zeros((0, 4)), np.zeros(0),
                                                    np.zeros((0, 2)), np.zeros((0, 17))))
                kept = gate(bx, sc, xy, kc, score_thr, **kw) if len(bx) else []
                det = len(kept) > 0
                if present == 1:
                    hit += int(det)
                else:
                    fp += int(det)
                    if det:
                        fp_frames.append("%s#%d" % (v, f))
            tag = "  <- 生产值" if ("复合门" in name and abs(score_thr - 0.05) < 1e-9) else ""
            print("%-24s %-8.2f %-22s %s%s"
                  % (name, score_thr, "%d/%d (%.1f%%)" % (hit, n_pos, 100.0 * hit / n_pos),
                     "%d/%d" % (fp, n_neg), tag))
            if args.list_fp and fp_frames:
                print("         误报帧: %s" % ", ".join(fp_frames))
        print()


if __name__ == "__main__":
    main()
