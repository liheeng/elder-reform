# -*- coding: utf-8 -*-
"""
export_hard_cases.py

导出**难例候选帧**，用于夜间/倒地/遮挡等场景下量化姿态模型与质量门的检出能力。

背景：`person_falldown-3.mp4` 这类素材上 YOLO 漏检是"丢对象"的主因（全片 22.4% 帧零检出），
而跟踪层修不了 —— 需要一小批人工标注过的难例帧才能量化改进。

挑帧规则（可调）：
    1. 连续零检出段：每段抽首/中/尾（段长 >= --min-run 才计入）
    2. 低置信检出帧：pipeline 检出了但 best score < --low-score
    3. 对照帧：best score >= --control-score 的帧等间隔抽 --controls 张（用于看"正常帧"的表现）

产出：
    hard_cases/frames/<video_key>/f<帧号>.jpg   原始分辨率，**不入 git**
    hard_cases/labels.csv                       人工标注（person_present 必填，bbox 可选）
    hard_cases/detections.csv                   pipeline 与裸 YOLO 的检出记录（对比基线）

重复运行是安全的：`labels.csv` 里已人工填写过的行会被保留，不会被覆盖。

用法：
    python scripts/build/export_hard_cases.py                       # 默认扫 人摔倒/ 目录
    python scripts/build/export_hard_cases.py --videos a.mp4 b.mp4
    python scripts/build/export_hard_cases.py --max-per-video 80 --min-run 5
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pose.pose_extractor import PoseExtractor  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from utils.device import get_device  # noqa: E402

DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo26n-pose.pt"
DEFAULT_SOURCE = "/Volumes/media-backup/AI-data/real-resource/Yolo视觉素材/人摔倒"

LABEL_FIELDS = ["video", "frame", "person_present", "x1", "y1", "x2", "y2", "note"]
DET_FIELDS = ["video", "frame", "pipeline_n_det", "pipeline_best_score", "pipeline_bbox",
              "raw_n_det", "raw_best_bbox", "raw_best_score", "note"]


def video_key(path):
    """给视频生成一个稳定且唯一的名字（不同目录可能有同名文件）。"""
    raw = "%s_%s" % (Path(path).parent.name, Path(path).stem)
    return re.sub(r"[^0-9A-Za-z_\-]+", "_", raw).strip("_")


def scan_video(video_path, extractor, log_every=200):
    """逐帧跑 PoseExtractor，返回每帧的检测摘要。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  [跳过] 无法打开: %s" % video_path)
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        persons = extractor.extract(frame)
        if persons:
            best = max(persons, key=lambda p: float(p.get("score", 0.0)))
            best_score = float(best.get("score", 0.0))
            best_bbox = tuple(int(v) for v in best["bbox"])
        else:
            best_score = -1.0
            best_bbox = ()
        frames.append({
            "frame": idx,
            "n_det": len(persons),
            "best_score": round(best_score, 4),
            "best_bbox": best_bbox,
        })
        if log_every and idx % log_every == 0:
            print("    %d/%d" % (idx, total))
    cap.release()
    return frames


def pick_candidates(frames, min_run, low_score, controls, control_score, max_per_video):
    """按规则挑候选帧，返回 [(frame, note)]。

    超限时按优先级裁剪：零检出帧（真正的漏检）> 低置信检出 > 对照帧。
    """
    picked = {}   # frame -> (priority, note)
    n = len(frames)
    by_frame = {f["frame"]: f for f in frames}

    # 规则 1: 连续零检出段，抽首/中/尾
    run_start = None
    for i in range(n + 1):
        empty = i < n and frames[i]["n_det"] == 0
        if empty and run_start is None:
            run_start = i
        if not empty and run_start is not None:
            length = i - run_start
            if length >= min_run:
                for pos, tag in ((0, "首"), (length // 2, "中"), (length - 1, "尾")):
                    fr = frames[run_start + pos]["frame"]
                    picked.setdefault(fr, (0, "zero_det_run=%d(%s)" % (length, tag)))
            run_start = None

    # 规则 2: 低置信检出帧
    for f in frames:
        if 0 <= f["best_score"] < low_score:
            picked.setdefault(f["frame"], (1, "low_score=%.2f" % f["best_score"]))

    # 规则 3: 对照帧（检出良好）
    strong = [f["frame"] for f in frames if f["best_score"] >= control_score]
    if controls > 0 and strong:
        step = max(1, len(strong) // controls)
        for fr in strong[::step][:controls]:
            picked.setdefault(fr, (2, "control_score=%.2f" % by_frame[fr]["best_score"]))

    ordered = sorted(picked.items(), key=lambda kv: (kv[1][0], kv[0]))
    if len(ordered) > max_per_video:
        ordered = ordered[:max_per_video]
    return [(fr, note) for fr, (_, note) in ordered]


def pick_negatives(video_path, raw_model, extractor, stride, min_score, max_per_video,
                   device, log_every=200):
    """挑"负样本候选"：裸 YOLO 开火了（没人也会开火 → 假阳性来源）。

    对每帧记录 pipeline 是否也检出（`gated`）：
      gated=0  质量门把它拦下了 —— **放宽阈值后这些帧会翻成"有检出"**，最关键
      gated=1  pipeline 今天就已经检出（例如把家具当人），即现有假阳性

    是否真的没有人，仍需人工看图确认（`labels.csv` 的 person_present 留空待填）。

    返回 [(frame, note)]。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("  [跳过] 无法打开: %s" % video_path)
        return []
    candidates = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if stride > 1 and idx % stride:
            continue
        res = raw_model(frame, verbose=False, conf=min_score, iou=0.5, device=device)
        b = res[0].boxes
        if b is None or len(b) == 0:
            continue
        best = float(b.conf.cpu().numpy().max())
        gated = 1 if extractor.extract(frame) else 0
        candidates.append((idx, best, gated))
        if log_every and idx % log_every == 0:
            print("    %d 帧, 候选 %d" % (idx, len(candidates)))
    cap.release()

    # 优先取"被质量门拦下"的帧（gated=0），其次才取现有假阳性
    candidates.sort(key=lambda t: (t[2], -t[1]))
    picked = sorted(candidates[:max_per_video])
    return [(fr, "neg_candidate:raw_score=%.3f,gated=%d" % (sc, g)) for fr, sc, g in picked]


def write_frames(video_path, wanted, out_dir):
    """按帧号抽取并写出原始分辨率 JPEG。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not wanted:
        return 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    written = 0
    idx = 0
    want = set(wanted)
    last = max(want)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx in want:
            cv2.imwrite(str(out_dir / ("f%06d.jpg" % idx)), frame)
            written += 1
        if idx >= last:
            break
    cap.release()
    return written


def read_csv_if_exists(path):
    if not path.exists():
        return []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(description="导出难例候选帧")
    ap.add_argument("--mode", choices=["hard", "negatives"], default="hard",
                    help="hard=零检出/低置信难例帧; negatives=裸YOLO开火但pipeline没检出的负样本候选")
    ap.add_argument("--videos", nargs="*", default=None, help="视频文件；不给则扫 --source 目录")
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="视频目录（递归找 .mp4/.avi）")
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "hard_cases"))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--conf", type=float, default=0.05, help="pipeline 检出的置信度阈值")
    ap.add_argument("--raw-conf", type=float, default=0.01, help="裸 YOLO 记录/负样本候选用的阈值")
    ap.add_argument("--min-run", type=int, default=3, help="连续零检出段的最小长度")
    ap.add_argument("--low-score", type=float, default=0.25, help="低于此分数视为可疑检出")
    ap.add_argument("--controls", type=int, default=3, help="每段视频抽多少张对照帧")
    ap.add_argument("--control-score", type=float, default=0.5, help="对照帧的最低检出分数")
    ap.add_argument("--stride", type=int, default=15, help="negatives 模式的采样间隔")
    ap.add_argument("--max-per-video", type=int, default=60)
    args = ap.parse_args()

    if args.videos:
        videos = [Path(v) for v in args.videos]
    else:
        src = Path(args.source)
        videos = sorted([p for p in src.rglob("*") if p.suffix.lower() in (".mp4", ".avi")])
    if not videos:
        print("没有找到视频")
        return

    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    labels_path = out_dir / "labels.csv"
    det_path = out_dir / "detections.csv"

    # 已有行按 (video, frame) 建索引：已标注必留；未标注的也要留（可能是另一模式刚加入的候选）
    existing = {}
    for row in read_csv_if_exists(labels_path):
        key = (row.get("video", ""), str(row.get("frame", "")))
        if key[0]:
            existing[key] = row

    extractor = PoseExtractor(model_path=args.model, conf=args.conf)
    raw_model = YOLO(args.model)
    device = get_device()

    print("视频数: %d，输出: %s" % (len(videos), out_dir))
    label_rows, det_rows = [], []

    for vp in videos:
        key = video_key(vp)
        print("--- %s" % vp.name)
        if args.mode == "negatives":
            picked = pick_negatives(vp, raw_model, extractor, args.stride,
                                    args.raw_conf, args.max_per_video, device)
            frames = []
        else:
            frames = scan_video(vp, extractor)
            if not frames:
                continue
            picked = pick_candidates(frames, args.min_run, args.low_score,
                                     args.controls, args.control_score, args.max_per_video)
        print("    候选帧 %d 张%s" % (len(picked),
              "" if args.mode == "negatives" else " (总帧数 %d)" % len(frames)))
        if not picked:
            continue

        n_written = write_frames(vp, [f for f, _ in picked], frames_dir / key)
        print("    写出帧 %d 张 -> %s" % (n_written, frames_dir / key))

        by_frame = {f["frame"]: f for f in frames}
        cap = cv2.VideoCapture(str(vp))
        idx = 0
        want = {f: note for f, note in picked}
        last = max(want) if want else 0
        raw_info = {}
        while want:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx in want:
                res = raw_model(frame, verbose=False, conf=args.raw_conf, iou=0.5, device=device)
                boxes = res[0].boxes
                if boxes is not None and len(boxes) > 0:
                    scores = boxes.conf.cpu().numpy()
                    b = int(np.argmax(scores))
                    raw_info[idx] = (len(boxes),
                                     tuple(int(v) for v in boxes.xyxy[b].cpu().numpy()),
                                     round(float(scores[b]), 4))
                else:
                    raw_info[idx] = (0, (), -1.0)
                want.pop(idx)
            if idx >= last:
                break
        cap.release()

        for fr, note in picked:
            f = by_frame.get(fr, {"n_det": 0, "best_score": -1.0, "best_bbox": ()})
            rc, rb, rs = raw_info.get(fr, (0, (), -1.0))
            label_rows.append({
                "video": key, "frame": fr, "person_present": "", "x1": "", "y1": "",
                "x2": "", "y2": "", "note": note,
            })
            det_rows.append({
                "video": key, "frame": fr,
                "pipeline_n_det": f["n_det"], "pipeline_best_score": f["best_score"],
                "pipeline_bbox": f["best_bbox"], "raw_n_det": rc,
                "raw_best_bbox": rb, "raw_best_score": rs, "note": note,
            })

    # 合并：以旧行为准（保留人工标注），本轮新候选补充进去
    merged = {}
    for k, row in existing.items():
        merged[k] = {f: row.get(f, "") for f in LABEL_FIELDS}
    for row in label_rows:
        k = (row["video"], str(row["frame"]))
        if k in merged:
            # 旧行里人工填过的字段不覆盖，只刷新 note
            old = merged[k]
            for f in LABEL_FIELDS:
                if f != "note" and (old.get(f) or "").strip():
                    row[f] = old[f]
        merged[k] = row
    merged = list(merged.values())

    # detections.csv 同样按 (video, frame) 合并, 避免新一轮覆盖掉上一轮的记录
    det_merged = {r["video"] + "#" + str(r["frame"]): r for r in read_csv_if_exists(det_path)}
    for row in det_rows:
        det_merged[row["video"] + "#" + str(row["frame"])] = row

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LABEL_FIELDS)
        w.writeheader()
        w.writerows(sorted(merged, key=lambda r: (r["video"], int(r["frame"]))))
    with open(det_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=DET_FIELDS)
        w.writeheader()
        w.writerows(sorted(det_merged.values(), key=lambda r: (r["video"], int(r["frame"]))))

    print()
    n_labeled = sum(1 for r in merged if (r.get("person_present") or "").strip() in ("0", "1"))
    print("labels.csv      %d 行（其中已人工标注 %d 行）" % (len(merged), n_labeled))
    print("detections.csv  %d 行（本轮新增 %d）" % (len(det_merged), len(det_rows)))
    print("下一步: 在 labels.csv 填 person_present(1/0)，关键帧可选填 x1,y1,x2,y2")


if __name__ == "__main__":
    main()
