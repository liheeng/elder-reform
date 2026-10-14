# -*- coding: utf-8 -*-
"""
build_dataset_5class.py

Le2i 5-class dataset builder - V2 with 30-frame windows, step=2 subsampling, 192-dim features.

Classes:
  0 = SAFE, 1 = WARNING, 2 = FALLEN, 3 = NON_PERSON, 4 = SITTING
"""

import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from features.feature_builder_v2 import FeatureBuilderV2, interpolate_missing_frames
from features.normalize import normalize_keypoints

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ARCHIVE = r"/Volumes/media-backup/AI-data/falldown/Le2i-v2"
KEYPOINTS_DIR = os.path.join(PROJECT_ROOT, "dataset/yolo_keypoints")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")

WINDOW_SIZE = 30
STRIDE = 3
TARGET_FPS = 25
INPUT_DIM = 192

CLASS_SAFE = 0
CLASS_WARNING = 1
CLASS_FALLEN = 2
CLASS_NON_PERSON = 3
CLASS_SITTING = 4

_LSHO, _RSHO = 5, 6
_LHIP, _RHIP = 11, 12

LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 2, 7: 2, 8: 2}

SCENES = {
    "Coffee_room_01": {
        "video_dir": os.path.join(ARCHIVE, "Coffee_room_01", "Coffee_room_01", "Videos"),
        "anno_dir": os.path.join(ARCHIVE, "Coffee_room_01", "Coffee_room_01", "Annotation_files"),
    },
    "Coffee_room_02": {
        "video_dir": os.path.join(ARCHIVE, "Coffee_room_02", "Coffee_room_02", "Videos"),
        "anno_dir": os.path.join(ARCHIVE, "Coffee_room_02", "Coffee_room_02", "Annotations_files"),
    },
    "Home_01": {
        "video_dir": os.path.join(ARCHIVE, "Home_01", "Home_01", "Videos"),
        "anno_dir": os.path.join(ARCHIVE, "Home_01", "Home_01", "Annotation_files"),
    },
    "Home_02": {
        "video_dir": os.path.join(ARCHIVE, "Home_02", "Home_02", "Videos"),
        "anno_dir": os.path.join(ARCHIVE, "Home_02", "Home_02", "Annotation_files"),
    },
}

NORMAL_SCENES = {
    "Lecture_room": {
        "video_dir": os.path.join(ARCHIVE, "Lecture_room", "Lecture room"),
        "confirmed_ids": [str(i) for i in range(15, 28)],
    },
    "Office": {
        "video_dir": os.path.join(ARCHIVE, "Office", "Office"),
        "confirmed_ids": [str(i) for i in range(19, 34)],
    },
}

def compute_torso_angle(keypoints):
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.shape != (17, 2):
        return None
    sho_mid = (kp[_LSHO] + kp[_RSHO]) / 2.0
    hip_mid = (kp[_LHIP] + kp[_RHIP]) / 2.0
    dx = sho_mid[0] - hip_mid[0]
    dy = sho_mid[1] - hip_mid[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return float(math.degrees(math.atan2(abs(dx), abs(dy))))


def keypoint_visible(kp, idx):
    return kp[idx, 0] > 0 and kp[idx, 1] > 0


def get_knee_y_avg(kp):
    if keypoint_visible(kp, 13) and keypoint_visible(kp, 14):
        return (kp[13, 1] + kp[14, 1]) / 2.0
    elif keypoint_visible(kp, 13):
        return kp[13, 1]
    elif keypoint_visible(kp, 14):
        return kp[14, 1]
    return None


def get_ankle_y_avg(kp):
    if keypoint_visible(kp, 15) and keypoint_visible(kp, 16):
        return (kp[15, 1] + kp[16, 1]) / 2.0
    elif keypoint_visible(kp, 15):
        return kp[15, 1]
    elif keypoint_visible(kp, 16):
        return kp[16, 1]
    return None


def is_sitting_posture(keypoints):
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.shape != (17, 2):
        return False
    ta = compute_torso_angle(keypoints)
    if ta is None or ta >= 25.0:
        return False
    hip_mid_y = (kp[_LHIP, 1] + kp[_RHIP, 1]) / 2.0
    knee_y = get_knee_y_avg(kp)
    ankle_y = get_ankle_y_avg(kp)
    if knee_y is None:
        return False
    if knee_y <= hip_mid_y:
        return True
    if ankle_y is None or ankle_y <= hip_mid_y:
        return True
    hip_ankle = ankle_y - hip_mid_y
    if hip_ankle <= 0:
        return False
    knee_ratio = (knee_y - hip_mid_y) / hip_ankle
    valid_kpts = kp[(kp[:, 0] > 0) & (kp[:, 1] > 0)]
    if len(valid_kpts) >= 4:
        spread = max(
            valid_kpts[:, 0].max() - valid_kpts[:, 0].min(),
            valid_kpts[:, 1].max() - valid_kpts[:, 1].min(),
        )
        if spread < 80.0:
            return True
    return knee_ratio < 0.3


def subsample_to_10fps(keypoints, confs, orig_fps=25.0):
    # step=1: 原生帧率, 不子采样
    T = keypoints.shape[0]
    step = 1
    indices = np.arange(0, T, step).astype(int)
    if len(indices) > 0 and indices[-1] != T - 1:
        indices = np.append(indices, T - 1)
    return keypoints[indices], confs[indices]


def parse_annotation_perframe(anno_path):
    if not os.path.exists(anno_path):
        return None, None, {}
    with open(anno_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    fall_start, fall_end = None, None
    frame_labels = {}
    pure_ints = []
    for ln in lines:
        parts = ln.split(",")
        if len(parts) == 1:
            try:
                pure_ints.append(int(parts[0]))
            except ValueError:
                pass
        elif len(parts) == 6:
            try:
                fidx = int(parts[0])
                label = int(parts[1])
                frame_labels[fidx] = label
            except ValueError:
                pass
    if len(pure_ints) >= 2:
        fall_start, fall_end = pure_ints[0], pure_ints[1]
    if fall_start == 0 and fall_end == 0:
        return None, None, {}
    return fall_start, fall_end, frame_labels


def get_window_class(frame_labels, start, end, fall_start, fall_end, raw_keypoints, subsample_indices):
    """
    5-class 窗口类别判定 (严格遵循原版 classify_window_v5 的顺序).
    优先级:
      1. fall_end 后的帧 → FALLEN
      2. 坐姿检测 → SITTING
      3. Le2i 逐帧标注众数投票
      4. 回退: risk 公式 WARNING = [fall_start-90, fall_end)
      5. → SAFE
    """
    last_orig_idx = subsample_indices[end - 1]

    # 1. FALLEN: 在 fall_end 之后 (含接触地面后)
    if fall_start is not None and fall_end is not None:
        if last_orig_idx >= fall_end:
            return CLASS_FALLEN

    # 1.5 跌倒过程中 → FALLEN (最后一帧标签确认即标FALLEN)
    if fall_start is not None and fall_end is not None:
        if fall_start <= last_orig_idx < fall_end:
            ol = frame_labels.get(last_orig_idx) if frame_labels else None
            if ol is not None and ol in (3, 4, 7, 8):
                return CLASS_FALLEN

    # 2. SITTING: 窗口后段帧坐姿检测
    window_len = end - start
    check_start = max(0, window_len - 10)
    sitting_count = 0
    check_frames = 0
    for t_s in range(start + check_start, end):
        orig_idx = subsample_indices[t_s]
        if orig_idx < len(raw_keypoints) and is_sitting_posture(raw_keypoints[orig_idx]):
            sitting_count += 1
        check_frames += 1
    if check_frames > 0 and sitting_count >= check_frames * 0.6:
        return CLASS_SITTING

    # 3. Le2i 逐帧标注：仅用于 SITTING 检测辅助，不再用于 FALLEN/WARNING
    # FALLEN 由 Step 1 (fall_end) 和 Step 4 (回退) 决定
    # WARNING 由 Pre-VFall 数据提供，Le2i 不产生 WARNING
    if frame_labels:
        sitting_count = 0
        for t_s in range(start, end):
            orig_idx = subsample_indices[t_s]
            ol = frame_labels.get(orig_idx)
            if ol is not None and ol in (3, 4, 7, 8):
                if orig_idx < len(raw_keypoints) and is_sitting_posture(raw_keypoints[orig_idx]):
                    sitting_count += 1
        if sitting_count >= (end - start) * 0.5:
            return CLASS_SITTING

    # 4. 回退: 只判定 FALLEN vs SAFE（WARNING 由 Pre-VFall 提供）
    if fall_start is not None and fall_end is not None:
        if last_orig_idx >= fall_end:
            return CLASS_FALLEN
        else:
            return CLASS_SAFE

    return CLASS_SAFE


def build_window_features(keypoints_seq, confs_seq=None):
    T = len(keypoints_seq)
    pos_list = []
    for i in range(T):
        k = keypoints_seq[i]
        c = confs_seq[i] if confs_seq is not None else None
        nk, nc = normalize_keypoints(k, c)
        if nk is None:
            nk, nc = k, c
        pos_list.append(FeatureBuilderV2.build(nk, nc))
    pos = np.array(pos_list, dtype=np.float32)
    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([pos, vel, acc], axis=1).astype(np.float32)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_x = []
    all_classes = []
    video_ids_list = []
    metadata = []
    stats = {CLASS_SAFE: 0, CLASS_WARNING: 0, CLASS_FALLEN: 0, CLASS_NON_PERSON: 0, CLASS_SITTING: 0}

    SCENE_ID_MAP = {
        "Coffee_room_01": 0,
        "Coffee_room_02": 1,
        "Home_01": 2,
        "Home_02": 3,
        "Lecture_room": 4,
        "Office": 5,
    }

    for scene_name, scene_info in SCENES.items():
        video_dir = scene_info["video_dir"]
        anno_dir = scene_info["anno_dir"]
        scene_id = SCENE_ID_MAP[scene_name]

        if not os.path.isdir(video_dir):
            print(f"[跳过] 视频目录不存在: {video_dir}")
            continue

        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".avi")])
        print(f"\n{scene_name} (scene_id={scene_id}): {len(video_files)} videos")

        for vf in video_files:
            video_id = vf.replace("video (", "").replace(").avi", "")
            kp_path = os.path.join(KEYPOINTS_DIR, f"{scene_name}_video_{video_id}_keypoints.npy")
            cf_path = os.path.join(KEYPOINTS_DIR, f"{scene_name}_video_{video_id}_confs.npy")

            if not os.path.exists(kp_path):
                print(f"  跳过 video {video_id} (无 keypoints 缓存)")
                continue

            raw_keypoints = np.load(kp_path)
            raw_confs = np.load(cf_path)
            raw_keypoints, raw_confs = interpolate_missing_frames(raw_keypoints, raw_confs)

            kp_10fps, cf_10fps = subsample_to_10fps(raw_keypoints, raw_confs)

            anno_file = os.path.join(anno_dir, f"video ({video_id}).txt")
            fall_start, fall_end, frame_labels = parse_annotation_perframe(anno_file)

            features = build_window_features(kp_10fps, cf_10fps)
            T_10 = features.shape[0]

            T_orig = raw_keypoints.shape[0]
            step = 1
            subsample_indices = np.arange(0, T_orig, step).astype(int)
            if len(subsample_indices) > 0 and subsample_indices[-1] != T_orig - 1:
                subsample_indices = np.append(subsample_indices, T_orig - 1)

            scene_windows = 0
            sitting_windows = 0
            for start in range(0, T_10 - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE
                x_win = features[start:end]

                cls = get_window_class(
                    frame_labels, start, end,
                    fall_start, fall_end,
                    raw_keypoints, subsample_indices,
                )

                all_x.append(x_win)
                all_classes.append(cls)
                video_ids_list.append([scene_id, int(video_id)])
                stats[cls] = stats.get(cls, 0) + 1
                scene_windows += 1
                if cls == CLASS_SITTING:
                    sitting_windows += 1

            metadata.append({
                "scene": scene_name, "video_id": video_id,
                "fall": fall_start is not None, "windows": scene_windows,
                "sitting_windows": sitting_windows,
            })
            print(f"  video {video_id}: {scene_windows} windows, sitting={sitting_windows}")

    for scene_name, scene_info in NORMAL_SCENES.items():
        video_dir = scene_info["video_dir"]
        confirmed_ids = set(scene_info["confirmed_ids"])
        scene_id = SCENE_ID_MAP[scene_name]

        if not os.path.isdir(video_dir):
            print(f"[跳过] 目录不存在: {video_dir}")
            continue

        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".avi")])
        print(f"\n{scene_name} (scene_id={scene_id}, confirmed normal):")

        for vf in video_files:
            video_id = vf.replace("video (", "").replace(").avi", "")
            if video_id not in confirmed_ids:
                continue

            kp_path = os.path.join(KEYPOINTS_DIR, f"{scene_name}_video_{video_id}_keypoints.npy")
            cf_path = os.path.join(KEYPOINTS_DIR, f"{scene_name}_video_{video_id}_confs.npy")

            if not os.path.exists(kp_path):
                print(f"  跳过 video {video_id} (无 keypoints 缓存)")
                continue

            raw_keypoints = np.load(kp_path)
            raw_confs = np.load(cf_path)
            raw_keypoints, raw_confs = interpolate_missing_frames(raw_keypoints, raw_confs)

            kp_10fps, cf_10fps = subsample_to_10fps(raw_keypoints, raw_confs)
            features = build_window_features(kp_10fps, cf_10fps)
            T_10 = features.shape[0]

            T_orig = raw_keypoints.shape[0]
            step = 1
            subsample_indices = np.arange(0, T_orig, step).astype(int)
            if len(subsample_indices) > 0 and subsample_indices[-1] != T_orig - 1:
                subsample_indices = np.append(subsample_indices, T_orig - 1)

            scene_windows = 0
            sitting_windows = 0
            for start in range(0, T_10 - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE
                x_win = features[start:end]

                window_len = end - start
                check_start = max(0, window_len - 10)
                sitting_count = 0
                check_frames = 0
                for t_s in range(start + check_start, end):
                    orig_idx = subsample_indices[t_s]
                    if orig_idx < len(raw_keypoints):
                        if is_sitting_posture(raw_keypoints[orig_idx]):
                            sitting_count += 1
                        check_frames += 1

                if check_frames > 0 and sitting_count > check_frames // 2:
                    cls = CLASS_SITTING
                    sitting_windows += 1
                else:
                    cls = CLASS_SAFE

                all_x.append(x_win)
                all_classes.append(cls)
                video_ids_list.append([scene_id, int(video_id)])
                stats[cls] = stats.get(cls, 0) + 1
                scene_windows += 1

            metadata.append({
                "scene": scene_name, "video_id": video_id,
                "fall": False, "windows": scene_windows,
                "sitting_windows": sitting_windows,
            })
            print(f"  video {video_id}: {scene_windows} windows, sitting={sitting_windows}")

    if len(all_x) == 0:
        print("\n错误: 没有生成任何训练样本!")
        sys.exit(1)

    xs = np.stack(all_x)
    classes = np.array(all_classes, dtype=np.int64)
    vids = np.array(video_ids_list, dtype=np.int64)

    np.save(os.path.join(PROCESSED_DIR, "x.npy"), xs)
    np.save(os.path.join(PROCESSED_DIR, "class_labels_5class.npy"), classes)
    np.save(os.path.join(PROCESSED_DIR, "video_ids.npy"), vids)

    meta_path = os.path.join(PROCESSED_DIR, "class_meta_5class.json")
    with open(meta_path, "w") as f:
        json.dump({"metadata": metadata, "total_windows": len(all_x)}, f, indent=2)

    total = len(all_x)
    print(f"\n{'='*60}")
    print("5-class 训练数据生成完成")
    print(f"{'='*60}")
    print(f"  总样本数: {total}")
    print(f"  特征形状: {xs.shape}")
    print(f"  SAFE(0):      {stats.get(CLASS_SAFE, 0):>6}  ({stats.get(CLASS_SAFE,0)/total*100:5.1f}%)")
    print(f"  WARNING(1):   {stats.get(CLASS_WARNING, 0):>6}  ({stats.get(CLASS_WARNING,0)/total*100:5.1f}%)")
    print(f"  FALLEN(2):    {stats.get(CLASS_FALLEN, 0):>6}  ({stats.get(CLASS_FALLEN,0)/total*100:5.1f}%)")
    print(f"  NON_PERSON(3):{stats.get(CLASS_NON_PERSON, 0):>6}  ({stats.get(CLASS_NON_PERSON,0)/total*100:5.1f}%)")
    print(f"  SITTING(4):   {stats.get(CLASS_SITTING, 0):>6}  ({stats.get(CLASS_SITTING,0)/total*100:5.1f}%)")
    print(f"  视频ID已保存: {os.path.join(PROCESSED_DIR, 'video_ids.npy')}")
    print(f"  元数据已保存: {meta_path}")


if __name__ == "__main__":
    main()

