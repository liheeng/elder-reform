# -*- coding: utf-8 -*-
"""
build_dataset_nonperson.py

从"已确认无人"的固定机位素材构建 NON_PERSON(3) 分片，字段口径与 build_dataset_v2.py 对齐。

为什么要做:
    class_labels_v2.npy 里 NON_PERSON(3) 样本数为 0，而真实场景中
    "电动车/车罩被当成行人"是持续性的整类误检，没有任何监督信号。
    pipeline/risk_engine_v2.py 对 class 3 的处理是直接忽略，
    该类缺失等于这条机制从未生效。

    processed/nonperson_meta.json 曾声称 Home_02 video 47/57 贡献了 95 个该窗口，
    但 class_labels_v2.npy 中 class 3 一个样本都没有 —— 该 meta 从未生效；
    且这两个视频实测"始终有人"。本脚本产出的 meta 会覆盖它。

口径一致性:
    - 复用 extract_keypoints.extract_video / select_person（含质量门、分裂合并）
    - 复用 build_dataset_v2.build_window_features（30 帧窗、192 维 pos+vel+acc）
    - 与 Le2i 侧一样先走 interpolate_missing_frames

关键取舍:
    只保留"窗口内多数帧确实检出目标"的窗口。NON_PERSON 要教的是
    "检测器以为有人、其实没有"；把纯零检出的窗口也标进来，学到的是
    "全零输入 => NON_PERSON"，对抑制误报没有帮助。

用法:
    python scripts/build/build_dataset_nonperson.py --dry-run
    python scripts/build/build_dataset_nonperson.py
    python scripts/build/build_dataset_nonperson.py --force        # 忽略缓存重新抽点
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from features.feature_builder_v2 import interpolate_missing_frames  # noqa: E402
from pose.pose_extractor import PoseExtractor  # noqa: E402
from scripts.build.build_dataset_v2 import (  # noqa: E402
    CLASS_NON_PERSON,
    NORMAL_TIME,
    RISK_SAFE,
    STRIDE,
    WINDOW_SIZE,
    build_window_features,
)
from scripts.build.extract_keypoints import (  # noqa: E402
    extract_video,
    resolve_model,
    save_cache,
)

PROCESSED_DIR = PROJECT_ROOT / "processed"
KEYPOINTS_DIR = PROJECT_ROOT / "yolo_keypoints"
INDEX_CSV = PROJECT_ROOT / "dataset" / "nonperson_index.csv"

# 0-5 是 Le2i 场景、10-12 是 PV 折组；20 未被占用，且不出现在任何折的测试集里，
# 因此这些窗口恒进训练集 —— 这是刻意的：9 段素材同属一个场景，
# 拆进测试集等于同场景泄露，测不出泛化能力。class 3 的效果由端到端回归判定。
NONPERSON_SCENE_ID = 20

# 窗口内至少有这么多比例的帧检出目标，才认为这个窗口在教"检出即非人"
DEFAULT_MIN_DET_RATIO = 0.5


def load_index(path: Path, cameras):
    """读登记表并按红线筛选：只取已确认无人、且机位符合要求的片段。

    两条判据都在这里硬编码，避免"顺手把没确认的素材喂进去"。
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("has_person", "").strip() != "no":
                continue
            if row.get("no_person_verified", "").strip() != "yes":
                continue
            if cameras and row.get("camera", "").strip().lower() not in cameras:
                continue
            rows.append(row)
    return rows


def cache_paths(stem: str):
    return (KEYPOINTS_DIR / f"nonperson_{stem}_keypoints.npy",
            KEYPOINTS_DIR / f"nonperson_{stem}_confs.npy")


def load_or_extract(video_path: Path, extractor, force: bool, log_every: int):
    """取关键点，优先用缓存。返回 (kpts, confs) 或 (None, None)。"""
    kp_path, cf_path = cache_paths(video_path.stem)
    if kp_path.exists() and cf_path.exists() and not force:
        return np.load(kp_path), np.load(cf_path)

    kpts, confs, hit = extract_video(video_path, extractor, log_every=log_every)
    if kpts is None or kpts.shape[0] == 0:
        return None, None
    save_cache(kpts, confs, kp_path, cf_path)
    print(f"    检出 {hit}/{kpts.shape[0]} 帧 ({hit / kpts.shape[0] * 100:.1f}%)，已缓存")
    return kpts, confs


def build_clip(video_path: Path, extractor, force: bool, min_det_ratio: float,
               log_every: int):
    """把单个片段切成窗口。返回 (x_list, 统计字典) 或 (None, 统计字典)。"""
    kpts, confs = load_or_extract(video_path, extractor, force, log_every)
    if kpts is None:
        return None, {"status": "跳过（抽点失败）"}

    # 有效帧判据与 interpolate_missing_frames 内部保持一致
    det_mask = kpts.reshape(kpts.shape[0], -1).sum(axis=1) > 1.0
    kpts, confs = interpolate_missing_frames(kpts, confs)
    feats = build_window_features(kpts, confs)

    total_frames = feats.shape[0]
    windows, kept, dropped = [], 0, 0
    for start in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        if det_mask[start:end].mean() < min_det_ratio:
            dropped += 1
            continue
        windows.append(feats[start:end])
        kept += 1

    stat = {
        "status": "ok",
        "frames": int(total_frames),
        "det_frame_ratio": round(float(det_mask.mean()), 3),
        "windows_total": kept + dropped,
        "windows_kept": kept,
        "windows_dropped_low_det": dropped,
    }
    return (np.stack(windows).astype(np.float32) if windows else None), stat


def main():
    ap = argparse.ArgumentParser(description="构建 NON_PERSON(3) 分片")
    ap.add_argument("--index", default=str(INDEX_CSV), help="素材登记表 CSV")
    ap.add_argument("--cameras", default="fixed",
                    help="允许的机位，逗号分隔；传空串表示不限")
    ap.add_argument("--model", default=None, help="Pose 权重路径（默认自动解析）")
    ap.add_argument("--out-prefix", default=str(PROCESSED_DIR / "nonperson_v2"),
                    help="输出文件前缀")
    ap.add_argument("--min-det-ratio", type=float, default=DEFAULT_MIN_DET_RATIO,
                    help="窗口内检出的最低帧占比，低于此值丢弃该窗口")
    ap.add_argument("--force", action="store_true", help="忽略关键点缓存重新抽点")
    ap.add_argument("--log-every", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true", help="只列出将处理的片段")
    args = ap.parse_args()

    cameras = {c.strip().lower() for c in args.cameras.split(",") if c.strip()}
    rows = load_index(Path(args.index), cameras)

    print("=" * 60)
    print("构建 NON_PERSON(3) 分片")
    print("=" * 60)
    print(f"登记表   : {args.index}")
    print(f"机位筛选 : {sorted(cameras) if cameras else '不限'}")
    print(f"窗口     : {WINDOW_SIZE} 帧 / stride {STRIDE}")
    print(f"丢弃阈值 : 窗口内检出帧占比 < {args.min_det_ratio}")
    print(f"scene_id : {NONPERSON_SCENE_ID}（不进任何折的测试集）")
    print(f"\n通过红线筛选的片段: {len(rows)}")
    for row in rows:
        print(f"  [{row.get('no', '?'):>2}] {Path(row['video']).name}"
              f"  {row.get('lighting', '?'):<5} {row.get('camera', '?'):<8}"
              f" motion={row.get('motion_mean', '?')}")
    if not rows:
        print("没有可用片段，退出。")
        return 1
    if args.dry_run:
        print("\n[dry-run] 不执行抽点。")
        return 0

    KEYPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    extractor = PoseExtractor(resolve_model(args.model))

    all_x, all_vids, meta = [], [], []
    print()
    for idx, row in enumerate(rows):
        video_path = PROJECT_ROOT / row["video"].lstrip("./")
        print(f"[{idx + 1}/{len(rows)}] {video_path.name}")
        if not video_path.exists():
            print("    跳过：文件不存在")
            meta.append(dict(clip=video_path.name, status="跳过（文件不存在）"))
            continue

        windows, stat = build_clip(video_path, extractor, args.force,
                                   args.min_det_ratio, args.log_every)
        stat.update(clip=video_path.name, category=row.get("type", ""),
                    lighting=row.get("lighting", ""), camera=row.get("camera", ""),
                    motion_mean=row.get("motion_mean", ""))
        meta.append(stat)

        if stat["status"] != "ok":
            print(f"    {stat['status']}")
            continue
        print(f"    {stat['frames']} 帧，检出帧占比 {stat['det_frame_ratio']:.3f}，"
              f"窗口 {stat['windows_kept']}/{stat['windows_total']} 保留"
              f"（低检出丢弃 {stat['windows_dropped_low_det']}）")
        if windows is None:
            print("    保留 0 个窗口，跳过")
            continue
        all_x.append(windows)
        all_vids.extend([[NONPERSON_SCENE_ID, idx]] * windows.shape[0])

    if not all_x:
        print("\n没有产出任何窗口，退出。")
        return 1

    xs = np.concatenate(all_x, axis=0).astype(np.float32)
    n = xs.shape[0]
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(prefix.with_name(prefix.name + "_x.npy"), xs)
    np.save(prefix.with_name(prefix.name + "_class.npy"),
            np.full(n, CLASS_NON_PERSON, dtype=np.int64))
    np.save(prefix.with_name(prefix.name + "_risk.npy"),
            np.full(n, RISK_SAFE, dtype=np.float32))
    np.save(prefix.with_name(prefix.name + "_time.npy"),
            np.full(n, NORMAL_TIME, dtype=np.float32))
    np.save(prefix.with_name(prefix.name + "_vids.npy"),
            np.array(all_vids, dtype=np.int64))

    meta_doc = {
        "_note": ("本文件由 build_dataset_nonperson.py 生成，覆盖此前声称 Home_02 "
                  "video 47/57 贡献 95 个 NON_PERSON 窗口的旧内容——那份记录从未生效，"
                  "class_labels_v2.npy 中 class 3 样本数为 0，且那两个视频实测始终有人。"),
        "scene_id": NONPERSON_SCENE_ID,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "min_det_ratio": args.min_det_ratio,
        "cameras": sorted(cameras),
        "index_csv": str(Path(args.index).name),
        "windows": int(n),
        "clips": meta,
    }
    with open(PROCESSED_DIR / "nonperson_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta_doc, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"完成：{n} 个窗口，形状 {xs.shape}")
    print(f"  {prefix.name}_x.npy / _class.npy / _risk.npy / _time.npy / _vids.npy")
    print(f"  来源记录 -> processed/nonperson_meta.json")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
