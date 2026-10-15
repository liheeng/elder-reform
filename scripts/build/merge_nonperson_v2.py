# -*- coding: utf-8 -*-
"""
merge_nonperson_v2.py

把 NON_PERSON 分片并入 v2 数据集，输出带后缀的新文件，不改动原文件。

为什么单独一步:
    x_v2.npy / class_labels_v2.npy 是 Le2i + PV 的产物，重建要读外置盘、耗时；
    非人分片则由 build_dataset_nonperson.py 独立产出。
    拆开之后可以"只换非人数据、不重跑 Le2i"，也便于回退。

用法:
    python scripts/build/merge_nonperson_v2.py --dry-run
    python scripts/build/merge_nonperson_v2.py
    python scripts/build/merge_nonperson_v2.py --suffix _np2
"""

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "processed"

BASE_FILES = {
    "x": "x_v2.npy",
    "class": "class_labels_v2.npy",
    "risk": "risk_labels_v2.npy",
    "time": "time_labels_v2.npy",
    "vids": "video_ids_v2.npy",
}

CLASS_NAMES = ["SAFE", "FALLEN", "SITTING", "NON_PERSON"]


def describe(y, title):
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    total = len(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(counts == 0, 0.0, total / (counts * len(CLASS_NAMES)))
    print(f"\n{title}（总 {total}）")
    for i, name in enumerate(CLASS_NAMES):
        share = counts[i] / total * 100
        print(f"  {name:<11} {counts[i]:>7}  {share:5.1f}%   权重 {weights[i]:.3f}")


def main():
    ap = argparse.ArgumentParser(description="把 NON_PERSON 分片并入 v2 数据集")
    ap.add_argument("--suffix", default="_np", help="输出文件名后缀")
    ap.add_argument("--shard-prefix", default="nonperson_v2",
                    help="分片文件前缀（见 build_dataset_nonperson.py）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base = {k: np.load(PROCESSED_DIR / v) for k, v in BASE_FILES.items()}
    n_base = base["class"].shape[0]
    for k, arr in base.items():
        if arr.shape[0] != n_base:
            print(f"[错误] {k} 长度 {arr.shape[0]} 与 class {n_base} 不一致，中止。")
            return 1

    shard = {
        "x": np.load(PROCESSED_DIR / f"{args.shard_prefix}_x.npy"),
        "class": np.load(PROCESSED_DIR / f"{args.shard_prefix}_class.npy"),
        "risk": np.load(PROCESSED_DIR / f"{args.shard_prefix}_risk.npy"),
        "time": np.load(PROCESSED_DIR / f"{args.shard_prefix}_time.npy"),
        "vids": np.load(PROCESSED_DIR / f"{args.shard_prefix}_vids.npy"),
    }
    n_shard = shard["class"].shape[0]

    if base["x"].shape[1:] != shard["x"].shape[1:]:
        print(f"[错误] 特征形状不匹配：base {base['x'].shape[1:]} vs shard {shard['x'].shape[1:]}")
        return 1

    # 分片必须完全不与既有 scene_id 重叠，否则折组划分会被意外改动
    base_scenes = set(np.unique(base["vids"][:, 0]).tolist())
    shard_scenes = set(np.unique(shard["vids"][:, 0]).tolist())
    overlap = base_scenes & shard_scenes
    if overlap:
        print(f"[错误] scene_id 重叠 {sorted(overlap)}，会打乱 LOSO 折组，中止。")
        return 1

    print("=" * 60)
    print("合并 NON_PERSON 分片")
    print("=" * 60)
    print(f"基础数据集: {n_base} 样本，scene_id {sorted(base_scenes)}")
    print(f"非人分片  : {n_shard} 样本，scene_id {sorted(shard_scenes)}")
    describe(base["class"], "合并前")
    if args.dry_run:
        print("\n[dry-run] 不写文件。")
        return 0

    merged = {k: np.concatenate([base[k], shard[k]], axis=0) for k in BASE_FILES}
    describe(merged["class"], "合并后")

    out_names = {
        "x": f"x_v2{args.suffix}.npy",
        "class": f"class_labels_v2{args.suffix}.npy",
        "risk": f"risk_labels_v2{args.suffix}.npy",
        "time": f"time_labels_v2{args.suffix}.npy",
        "vids": f"video_ids_v2{args.suffix}.npy",
    }
    print()
    for k, name in out_names.items():
        np.save(PROCESSED_DIR / name, merged[k])
        print(f"  {name:<32} {merged[k].shape}")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
