# -*- coding: utf-8 -*-
"""
build_dataset_v2.py

V2 dataset builder - dual-head (class + risk) version.
4 classes: SAFE(0), FALLEN(1), SITTING(2), NON_PERSON(3)
Risk labels for regression head: 0.0 / 0.3 / 0.5 / 1.0
Merges Le2i data + Pre-VFall data.
"""

import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from features.feature_builder_v2 import FeatureBuilderV2, interpolate_missing_frames
from features.normalize import normalize_keypoints

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ARCHIVE = r"/Volumes/media-backup/AI-data/dataset/falldown/Le2i-v2"
KEYPOINTS_DIR = os.path.join(PROJECT_ROOT, "yolo_keypoints")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")

WINDOW_SIZE = 30
STRIDE = 3
TARGET_FPS = 25
INPUT_DIM = 192

# V2 类定义
CLASS_SAFE = 0
CLASS_FALLEN = 1
CLASS_SITTING = 2
CLASS_NON_PERSON = 3

# V2 风险标签
RISK_SAFE = 0.0
RISK_FALL = 1.0
TAU = 50
PRE_FALL = 120
NORMAL_TIME = 999

_LSHO, _RSHO = 5, 6
_LHIP, _RHIP = 11, 12

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

def is_sitting_posture_v2(keypoints):
    """基于关节角度的坐姿检测（尺度不变）"""
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.shape != (17, 2):
        return False
    nk, _ = normalize_keypoints(kp, np.ones(17, dtype=np.float32))
    if nk is None:
        nk = kp
    feat = FeatureBuilderV2.build(nk, np.ones(17, dtype=np.float32))
    torso_angle = feat[52]
    left_knee_angle = feat[53]
    right_knee_angle = feat[54]
    left_hip_angle = feat[61]
    right_hip_angle = feat[62]
    avg_knee = (left_knee_angle + right_knee_angle) / 2.0
    avg_hip = (left_hip_angle + right_hip_angle) / 2.0
    if torso_angle >= 25.0:
        return False
    if avg_knee >= 130.0:
        return False
    if avg_hip >= 140.0:
        return False
    if avg_knee <= 30.0 and avg_hip <= 30.0:
        return False
    return True


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


def subsample_to_10fps(keypoints, confs, orig_fps=25.0):
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


def get_window_class_and_risk(frame_labels, start, end, fall_start, fall_end,
                               raw_keypoints, subsample_indices):
    """返回 (class_label, risk_label, time_label) 三元组"""
    last_orig_idx = subsample_indices[end - 1]

    # 规则 1: fall_end 之后 �?FALLEN, risk=1.0, time=0
    if fall_start is not None and fall_end is not None:
        if last_orig_idx >= fall_end:
            return CLASS_FALLEN, RISK_FALL, 0.0

    # 规则 2: 跌倒过程中，逐帧标注�?3/4/7/8 �?FALLEN, risk=1.0, time=0
    if fall_start is not None and fall_end is not None:
        if fall_start <= last_orig_idx < fall_end:
            ol = frame_labels.get(last_orig_idx) if frame_labels else None
            if ol is not None and ol in (3, 4, 7, 8):
                return CLASS_FALLEN, RISK_FALL, 0.0

    # 规则 3: 窗口�?10 帧坐姿检�?�?0% �?SITTING, risk=0.0, time=999
    window_len = end - start
    check_start = max(0, window_len - 10)
    sitting_count = 0
    check_frames = 0
    for t_s in range(start + check_start, end):
        orig_idx = subsample_indices[t_s]
        if orig_idx < len(raw_keypoints) and is_sitting_posture_v2(raw_keypoints[orig_idx]):
            sitting_count += 1
        check_frames += 1
    if check_frames > 0 and sitting_count >= check_frames * 0.6:
        return CLASS_SITTING, RISK_SAFE, NORMAL_TIME

    # 规则 4: 逐帧标注坐姿 �?0% �?SITTING, risk=0.0, time=999
    if frame_labels:
        sitting_count = 0
        for t_s in range(start, end):
            orig_idx = subsample_indices[t_s]
            ol = frame_labels.get(orig_idx)
            if ol is not None and ol in (3, 4, 7, 8):
                if orig_idx < len(raw_keypoints) and is_sitting_posture_v2(raw_keypoints[orig_idx]):
                    sitting_count += 1
        if sitting_count >= (end - start) * 0.5:
            return CLASS_SITTING, RISK_SAFE, NORMAL_TIME

    # 规则 5: 指数风险衰减 → 有 fall 标注的视频
    if fall_start is not None and fall_end is not None:
        if last_orig_idx >= fall_end:
            return CLASS_FALLEN, RISK_FALL, 0.0
        elif last_orig_idx >= fall_start - PRE_FALL:
            time_frames = max(0, fall_end - 1 - last_orig_idx)
            risk = float(np.exp(-time_frames / TAU))
            return CLASS_SAFE, risk, float(time_frames) / PRE_FALL
        else:
            return CLASS_SAFE, RISK_SAFE, NORMAL_TIME

    # 无 fall 标注的正常视频
    return CLASS_SAFE, RISK_SAFE, NORMAL_TIME


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
    all_risks = []
    all_times = []
    video_ids_list = []
    metadata = []
    stats_c = {CLASS_SAFE: 0, CLASS_FALLEN: 0, CLASS_SITTING: 0, CLASS_NON_PERSON: 0}
    stats_r = {"risk=0.0": 0, "0<risk�?.3": 0, "0.3<risk�?.5": 0,
               "0.5<risk<1.0": 0, "risk=1.0": 0}

    SCENE_ID_MAP = {
        "Coffee_room_01": 0,
        "Coffee_room_02": 1,
        "Home_01": 2,
        "Home_02": 3,
        "Lecture_room": 4,
        "Office": 5,
    }

    # ======= 处理有标注的场景（fall 视频�?=======
    for scene_name, scene_info in SCENES.items():
        video_dir = scene_info["video_dir"]
        anno_dir = scene_info["anno_dir"]
        scene_id = SCENE_ID_MAP[scene_name]

        if not os.path.isdir(video_dir):
            print(f"[跳过] 视频目录不存�? {video_dir}")
            continue

        video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(".avi")])
        print(f"\n{scene_name} (scene_id={scene_id}): {len(video_files)} videos")

        for vf in video_files:
            video_id = vf.replace("video (", "").replace(").avi", "")
            kp_path = os.path.join(KEYPOINTS_DIR, f"{scene_name}_video_{video_id}_keypoints.npy")
            cf_path = os.path.join(KEYPOINTS_DIR, f"{scene_name}_video_{video_id}_confs.npy")

            if not os.path.exists(kp_path):
                print(f"  跳过 video {video_id} (keypoints 缓存)")
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
            for start in range(0, T_10 - WINDOW_SIZE + 1, STRIDE):
                end = start + WINDOW_SIZE
                x_win = features[start:end]

                cls, risk, time_val = get_window_class_and_risk(
                    frame_labels, start, end,
                    fall_start, fall_end,
                    raw_keypoints, subsample_indices,
                )

                all_x.append(x_win)
                all_classes.append(cls)
                all_risks.append(risk)
                all_times.append(time_val)
                video_ids_list.append([scene_id, int(video_id)])
                stats_c[cls] = stats_c.get(cls, 0) + 1
                if risk == 0.0:
                    stats_r["risk=0.0"] = stats_r.get("risk=0.0", 0) + 1
                elif risk <= 0.3:
                    stats_r["0<risk�?.3"] = stats_r.get("0<risk�?.3", 0) + 1
                elif risk <= 0.5:
                    stats_r["0.3<risk�?.5"] = stats_r.get("0.3<risk�?.5", 0) + 1
                elif risk < 1.0:
                    stats_r["0.5<risk<1.0"] = stats_r.get("0.5<risk<1.0", 0) + 1
                else:
                    stats_r["risk=1.0"] = stats_r.get("risk=1.0", 0) + 1
                scene_windows += 1

            metadata.append({
                "scene": scene_name, "video_id": video_id,
                "fall": fall_start is not None, "windows": scene_windows,
            })
            print(f"  video {video_id}: {scene_windows} windows")

    # ======= 处理无标注的正常视频 =======
    for scene_name, scene_info in NORMAL_SCENES.items():
        video_dir = scene_info["video_dir"]
        confirmed_ids = set(scene_info["confirmed_ids"])
        scene_id = SCENE_ID_MAP[scene_name]

        if not os.path.isdir(video_dir):
            print(f"[跳过] 目录不存�? {video_dir}")
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
                print(f"  跳过 video {video_id} (keypoints 缓存)")
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

                # 对正常视频：坐姿检测或 SAFE
                window_len = end - start
                check_start = max(0, window_len - 10)
                sitting_count = 0
                check_frames = 0
                for t_s in range(start + check_start, end):
                    orig_idx = subsample_indices[t_s]
                    if orig_idx < len(raw_keypoints):
                        if is_sitting_posture_v2(raw_keypoints[orig_idx]):
                            sitting_count += 1
                        check_frames += 1

                if check_frames > 0 and sitting_count > check_frames // 2:
                    cls = CLASS_SITTING
                    risk = RISK_SAFE
                    time_val = NORMAL_TIME
                    sitting_windows += 1
                else:
                    cls = CLASS_SAFE
                    risk = RISK_SAFE
                    time_val = NORMAL_TIME

                all_x.append(x_win)
                all_classes.append(cls)
                all_risks.append(risk)
                all_times.append(time_val)
                video_ids_list.append([scene_id, int(video_id)])
                stats_c[cls] = stats_c.get(cls, 0) + 1
                stats_r["risk=0.0"] = stats_r.get("risk=0.0", 0) + 1
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

    # ======= 合并 Pre-VFall 数据 =======
    pv_x_path = os.path.join(PROCESSED_DIR, "prevfall_x_v2.npy")
    pv_class_path = os.path.join(PROCESSED_DIR, "prevfall_class_v2.npy")
    pv_risk_path = os.path.join(PROCESSED_DIR, "prevfall_risk_v2.npy")
    pv_time_path = os.path.join(PROCESSED_DIR, "prevfall_time_v2.npy")
    pv_ids_path = os.path.join(PROCESSED_DIR, "prevfall_ids_v2.npy")

    if os.path.exists(pv_x_path):
        print(f"\n加载 Pre-VFall 数据...")
        pv_x = np.load(pv_x_path)
        pv_class = np.load(pv_class_path)
        pv_risk = np.load(pv_risk_path)
        pv_time = np.load(pv_time_path)
        pv_ids = np.load(pv_ids_path)
        print(f"  prevfall samples: {len(pv_x)}")

        all_x.extend([pv_x[i] for i in range(len(pv_x))])
        all_classes.extend(pv_class.tolist())
        all_risks.extend(pv_risk.tolist())
        all_times.extend(pv_time.tolist())
        video_ids_list.extend(pv_ids.tolist())
        for c in pv_class:
            stats_c[c] = stats_c.get(c, 0) + 1
        for r in pv_risk:
            r = float(r)
            if r == 0.0:
                stats_r["risk=0.0"] = stats_r.get("risk=0.0", 0) + 1
            elif r <= 0.3:
                stats_r["0<risk�?.3"] = stats_r.get("0<risk�?.3", 0) + 1
            elif r <= 0.5:
                stats_r["0.3<risk�?.5"] = stats_r.get("0.3<risk�?.5", 0) + 1
            elif r < 1.0:
                stats_r["0.5<risk<1.0"] = stats_r.get("0.5<risk<1.0", 0) + 1
            else:
                stats_r["risk=1.0"] = stats_r.get("risk=1.0", 0) + 1
    else:
        print(f"\n[警告] Pre-VFall 数据不存�? {pv_x_path}")
        print("  跳过合并，仅使用 Le2i 数据")

    # ======= 保存 =======
    xs = np.stack(all_x).astype(np.float32)
    classes = np.array(all_classes, dtype=np.int64)
    risks = np.array(all_risks, dtype=np.float32)
    times = np.array(all_times, dtype=np.float32)
    vids = np.array(video_ids_list, dtype=np.int64)

    np.save(os.path.join(PROCESSED_DIR, "x_v2.npy"), xs)
    np.save(os.path.join(PROCESSED_DIR, "class_labels_v2.npy"), classes)
    np.save(os.path.join(PROCESSED_DIR, "risk_labels_v2.npy"), risks)
    np.save(os.path.join(PROCESSED_DIR, "time_labels_v2.npy"), times)
    np.save(os.path.join(PROCESSED_DIR, "video_ids_v2.npy"), vids)

    total = len(xs)
    print(f"\n{'='*60}")
    print("V2 数据集构建完成!")
    print(f"{'='*60}")
    print(f"  总样本数: {total}")
    print(f"  特征形状: {xs.shape}")
    print(f"  SAFE(0):      {stats_c.get(CLASS_SAFE, 0):>6}  ({stats_c.get(CLASS_SAFE,0)/total*100:5.1f}%)")
    print(f"  FALLEN(1):    {stats_c.get(CLASS_FALLEN, 0):>6}  ({stats_c.get(CLASS_FALLEN,0)/total*100:5.1f}%)")
    print(f"  SITTING(2):   {stats_c.get(CLASS_SITTING, 0):>6}  ({stats_c.get(CLASS_SITTING,0)/total*100:5.1f}%)")
    print(f"  NON_PERSON(3):{stats_c.get(CLASS_NON_PERSON, 0):>6}  ({stats_c.get(CLASS_NON_PERSON,0)/total*100:5.1f}%)")
    print()
    for key in ["risk=0.0", "0<risk�?.3", "0.3<risk�?.5", "0.5<risk<1.0", "risk=1.0"]:
        print(f"  {key}: {stats_r.get(key, 0):>6}")


if __name__ == "__main__":
    main()
