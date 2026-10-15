# -*- coding: utf-8 -*-
"""
eval_nonperson_effect.py

验证 NON_PERSON(3) 分片是否真的生效：在**未参与训练**的误检素材上，
比较不同模型把窗口判成 NON_PERSON 的比例。

为什么需要这个检查:
    端到端回归只能看到 SAFE / WARNING / DANGER 三种显示状态，
    而 NON_PERSON 与 SITTING 在 risk_engine_v2 里都被强制零风险，
    两者的屏幕表现完全一样。所以要判断"分片有没有被学到"，
    必须直接看模型的类别输出。

为什么用这些素材:
    默认取登记表里 camera=handheld 的片段 —— 它们因为机位不合格被
    build_dataset_nonperson.py 排除在训练集之外，是天然的 held-out 误检素材。
    这些片段里探测器持续误报（实测检出帧占比接近 1.0）。

用法:
    python scripts/eval/eval_nonperson_effect.py
    python scripts/eval/eval_nonperson_effect.py --suffixes "",_baseline,_np
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.lstm_classifier_v2 import CLASS_NAMES, LSTMClassifierV2  # noqa: E402
from features.feature_builder_v2 import interpolate_missing_frames  # noqa: E402
from pose.pose_extractor import PoseExtractor  # noqa: E402
from scripts.build.build_dataset_nonperson import (  # noqa: E402
    INDEX_CSV,
    load_index,
    load_or_extract,
)
from scripts.build.build_dataset_v2 import STRIDE, WINDOW_SIZE, build_window_features  # noqa: E402
from scripts.build.extract_keypoints import resolve_model  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models"


def build_windows(video_path, extractor, min_det_ratio):
    """返回 (windows (N,30,192), 检出帧占比) 或 (None, None)。"""
    kpts, confs = load_or_extract(video_path, extractor, force=False, log_every=500)
    if kpts is None:
        return None, None
    det_mask = kpts.reshape(kpts.shape[0], -1).sum(axis=1) > 1.0
    kpts, confs = interpolate_missing_frames(kpts, confs)
    feats = build_window_features(kpts, confs)
    windows = [feats[s:s + WINDOW_SIZE]
               for s in range(0, feats.shape[0] - WINDOW_SIZE + 1, STRIDE)
               if det_mask[s:s + WINDOW_SIZE].mean() >= min_det_ratio]
    if not windows:
        return None, float(det_mask.mean())
    return np.stack(windows).astype(np.float32), float(det_mask.mean())


def resolve_model_paths(suffix):
    model = MODEL_DIR / f"lstm_v2{suffix}.pt"
    norm = MODEL_DIR / f"norm_v2{suffix}.npz"
    temp = MODEL_DIR / f"temperature_v2{suffix}.npz"
    if not model.exists() or not norm.exists():
        return None
    return model, norm, (temp if temp.exists() else None)


def main():
    ap = argparse.ArgumentParser(description="测量 NON_PERSON 判定的实际生效情况")
    ap.add_argument("--suffixes", default=",_baseline,_np",
                    help="逗号分隔的模型后缀（空串=生产模型）")
    ap.add_argument("--camera", default="handheld",
                    help="从登记表取哪类机位的片段（默认 handheld=未参与训练的那批）")
    ap.add_argument("--min-det-ratio", type=float, default=0.5)
    args = ap.parse_args()

    suffixes = [s.strip() for s in args.suffixes.split(",")]
    cameras = {c.strip().lower() for c in args.camera.split(",") if c.strip()}
    rows = load_index(INDEX_CSV, cameras)
    if not rows:
        print(f"登记表里没有 camera={sorted(cameras)} 的片段，退出。")
        return 1

    print("=" * 72)
    print("NON_PERSON 生效检查（素材未参与训练）")
    print("=" * 72)
    print(f"片段: {len(rows)} 段，机位 {sorted(cameras)}")
    for row in rows:
        print(f"  {Path(row['video']).name}  ({row.get('duration_s', '?')}s, "
              f"motion={row.get('motion_mean', '?')})")

    extractor = PoseExtractor(resolve_model(None))
    per_clip = {}
    for row in rows:
        video_path = PROJECT_ROOT / row["video"].lstrip("./")
        windows, det_ratio = build_windows(video_path, extractor, args.min_det_ratio)
        if windows is None:
            print(f"[跳过] {video_path.name}: 无窗口")
            continue
        per_clip[video_path.name] = windows
        print(f"  构建 {video_path.name}: {windows.shape[0]} 窗口，检出帧占比 {det_ratio:.3f}")

    if not per_clip:
        print("没有可用窗口，退出。")
        return 1

    all_windows = np.concatenate(list(per_clip.values()), axis=0)
    print(f"\n合计 {all_windows.shape[0]} 个窗口\n")

    print(f"{'模型':<18}{'NON_PERSON 占比':>16}{'SITTING':>10}{'SAFE':>10}"
          f"{'FALLEN':>10}{'NON_PERSON 平均概率':>22}")
    for suffix in suffixes:
        paths = resolve_model_paths(suffix)
        label = suffix if suffix else "(生产)"
        if paths is None:
            print(f"{label:<18}{'模型文件缺失，跳过':>30}")
            continue
        clf = LSTMClassifierV2(str(paths[0]), str(paths[1]),
                               str(paths[2]) if paths[2] else None)
        counts = Counter()
        prob_sum = 0.0
        for i in range(all_windows.shape[0]):
            out = clf.predict(all_windows[i])
            counts[out["class_name"]] += 1
            prob_sum += out["probs"]["NON_PERSON"]
        n = all_windows.shape[0]
        print(f"{label:<18}{counts['NON_PERSON'] / n * 100:>15.1f}%"
              f"{counts['SITTING'] / n * 100:>9.1f}%"
              f"{counts['SAFE'] / n * 100:>9.1f}%"
              f"{counts['FALLEN'] / n * 100:>9.1f}%"
              f"{prob_sum / n:>21.3f}")

    print("\n解读: 这些片段里探测器持续把非人目标当成行人，"
          "\n      NON_PERSON 占比越高说明该类别越有效地吸收了这类误检。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
