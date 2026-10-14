"""
verify_real_videos.py [REFORM 版]

回归验证: 跌倒视频应触发 DANGER, 正常视频应保持SAFE.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sequence.sequence_buffer import SequenceBufferV3
from classifier.lstm_classifier import LSTMClassifier
from pipeline.risk_engine import RiskEngine
from utils.device import resolve_device

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YOLO_MODEL = os.path.join(PROJECT_ROOT, "models", "yolo26n-pose.pt")
LSTM_MODEL = os.path.join(PROJECT_ROOT, "models", "lstm_5class.pt")
NORM_PARAMS = os.path.join(PROJECT_ROOT, "models", "norm_5class.npz")
TEMP_PARAMS = os.path.join(PROJECT_ROOT, "models", "temperature_scaling.npz")
if not os.path.exists(LSTM_MODEL):
    LSTM_MODEL = os.path.join(PROJECT_ROOT, "models", "lstm_4class.pt")
    NORM_PARAMS = os.path.join(PROJECT_ROOT, "models", "norm_4class.npz")
    if not os.path.exists(LSTM_MODEL):
        LSTM_MODEL = os.path.join(PROJECT_ROOT, "models", "lstm_multitask.pt")
        NORM_PARAMS = os.path.join(PROJECT_ROOT, "models", "norm_params.npz")

ARCHIVE = r"C:\Users\y'y'x\Downloads\archive"
WINDOW = 30
KEYPOINTS_DIR = os.path.join(PROJECT_ROOT, "yolo_keypoints")

# 运行时解析（CUDA -> MPS -> CPU）
YOLO_DEVICE = resolve_device("auto")

FALL_VIDEOS = [
    ("Home_01", "video (1)",
     os.path.join(ARCHIVE, "Home_01", "Home_01", "Videos", "video (1).avi"),
     os.path.join(ARCHIVE, "Home_01", "Home_01", "Annotation_files", "video (1).txt")),
    ("Home_01", "video (10)",
     os.path.join(ARCHIVE, "Home_01", "Home_01", "Videos", "video (10).avi"),
     os.path.join(ARCHIVE, "Home_01", "Home_01", "Annotation_files", "video (10).txt")),
    ("Home_01", "video (12)",
     os.path.join(ARCHIVE, "Home_01", "Home_01", "Videos", "video (12).avi"),
     os.path.join(ARCHIVE, "Home_01", "Home_01", "Annotation_files", "video (12).txt")),
    ("Home_02", "video (1)",
     os.path.join(ARCHIVE, "Home_02", "Home_02", "Videos", "video (1).avi"),
     os.path.join(ARCHIVE, "Home_02", "Home_02", "Annotation_files", "video (1).txt")),
    ("Coffee_room_01", "video (1)",
     os.path.join(ARCHIVE, "Coffee_room_01", "Coffee_room_01", "Videos", "video (1).avi"),
     os.path.join(ARCHIVE, "Coffee_room_01", "Coffee_room_01", "Annotation_files", "video (1).txt")),
    ("Coffee_room_02", "video (1)",
     os.path.join(ARCHIVE, "Coffee_room_02", "Coffee_room_02", "Videos", "video (1).avi"),
     os.path.join(ARCHIVE, "Coffee_room_02", "Coffee_room_02", "Annotations_files", "video (1).txt")),
]

NORMAL_VIDEOS = [
    ("Lecture_room", "video (15)",
     os.path.join(ARCHIVE, "Lecture_room", "Lecture room", "video (15).avi")),
    ("Lecture_room", "video (16)",
     os.path.join(ARCHIVE, "Lecture_room", "Lecture room", "video (16).avi")),
    ("Office", "video (1)",
     os.path.join(ARCHIVE, "Office", "Office", "video (1).avi")),
    ("Office", "video (20)",
     os.path.join(ARCHIVE, "Office", "Office", "video (20).avi")),
]


def extract_kpts(video_path, scene_name, video_id):
    """从训练数据缓存加载关键点（与训练时完全一致）。"""
    kp_path = os.path.join(KEYPOINTS_DIR, f"{scene_name}_video_{video_id}_keypoints.npy")
    if not os.path.exists(kp_path):
        # 无缓存时用YOLO 提取
        from ultralytics import YOLO
        import torch
        model = YOLO(YOLO_MODEL)
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        kpts_all = np.zeros((total, 17, 2), dtype=np.float32)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            res = model(frame, verbose=False, device=YOLO_DEVICE)
            if res[0].keypoints is not None and len(res[0].keypoints) > 0:
                xy = res[0].keypoints.xy.cpu().numpy()
                if xy.ndim == 3 and xy.shape[1] == 17:
                    if res[0].keypoints.conf is not None:
                        conf = res[0].keypoints.conf.cpu().numpy()
                        best = conf.sum(axis=1).argmax()
                    else:
                        best = 0
                    kpts_all[idx] = xy[best]
            idx += 1
        cap.release()
        return kpts_all[:idx], h
    return np.load(kp_path), None
    cap.release()
    return kpts_all[:idx], h


def run_pipeline(kpts, frame_h):
    clf = LSTMClassifier(model_path=LSTM_MODEL, norm_path=NORM_PARAMS, temperature_path=TEMP_PARAMS)
    from features.feature_builder_v2 import FeatureBuilderV2
    from features.normalize import normalize_keypoints
    buffer = SequenceBufferV3(seq_len=WINDOW)
    engine = RiskEngine()
    from collections import Counter
    states = Counter()
    max_risk = 0
    for t in range(len(kpts)):
        kpt = kpts[t]
        if np.all(kpt == 0):
            continue
        nk, nc = normalize_keypoints(kpt)
        if nk is not None:
            feat = FeatureBuilderV2.build(nk, nc)
        else:
            feat = FeatureBuilderV2.build(kpt)
        buffer.update(0, feat)
        if buffer.is_ready(0):
            raw = buffer.get_sequence(0)
            vel = np.zeros_like(raw)
            vel[1:] = raw[1:] - raw[:-1]
            acc = np.zeros_like(vel)
            acc[1:] = vel[1:] - vel[:-1]
            seq192 = np.concatenate([raw, vel, acc], axis=1)
            res = clf.predict(seq192)
            rr = engine.update(
                0,
                class_probs=res.get("probs"),
                class_id=res.get("class_id"),
                danger_prob=res.get("risk"),
            )
            states[rr["state"]] += 1
            max_risk = max(max_risk, rr["risk_score"])
    return dict(states), max_risk


def parse_annotation(anno_path):
    if not os.path.exists(anno_path):
        return None, None
    with open(anno_path, "r") as f:
        lines = f.readlines()
    try:
        fs = int(lines[0].strip())
        fe = int(lines[1].strip())
    except (ValueError, IndexError):
        return None, None
    if fs <= 0 or fe <= 0:
        return None, None
    return fs, fe


def main():
    print("=" * 76)
    print("回归验证 [REFORM]")
    print("=" * 76)
    print(f"Model: {os.path.basename(LSTM_MODEL)}")

    print(f"\n{'='*40}")
    print("A. 跌倒视频→应触发DANGER")
    print(f"{'='*40}")
    fall_pass = fall_total = 0
    for scene, vid, vpath, apath in FALL_VIDEOS:
        if not os.path.exists(vpath):
            print(f"  [跳过] {scene}/{vid}")
            continue
        fall_total += 1
        print(f"  测试: {scene}/{vid} ...", end=" ", flush=True)
        video_id = vid.replace("video (", "").replace(")", "")
        kpts, h = extract_kpts(vpath, scene, video_id)
        states, max_risk = run_pipeline(kpts, h or 240)
        danger = states.get("DANGER", 0)
        ok = danger > 0
        if ok:
            fall_pass += 1
        print(f"DANGER={danger:3d}  max_risk={max_risk:.3f}  {'PASS' if ok else 'FAIL'}")

    print(f"\n{'='*40}")
    print("B. 正常视频 →不应 DANGER")
    print(f"{'='*40}")
    normal_pass = normal_total = 0
    for scene, vid, vpath in NORMAL_VIDEOS:
        if not os.path.exists(vpath):
            print(f"  [跳过] {scene}/{vid}")
            continue
        normal_total += 1
        print(f"  测试: {scene}/{vid} ...", end=" ", flush=True)
        video_id = vid.replace("video (", "").replace(")", "")
        kpts, h = extract_kpts(vpath, scene, video_id)
        states, max_risk = run_pipeline(kpts, h or 240)
        danger = states.get("DANGER", 0)
        ok = danger == 0
        if ok:
            normal_pass += 1
        print(f"DANGER={danger:3d}  max_risk={max_risk:.3f}  {'PASS' if ok else 'FAIL'}")

    print(f"\n{'='*76}")
    print("总结")
    print(f"{'='*76}")
    print(f"  跌倒视频 {fall_pass}/{fall_total}  (目标: 全部PASS)")
    print(f"  正常视频: {normal_pass}/{normal_total}  (目标: 全部PASS)")
    overall = "PASS" if (fall_pass == fall_total and normal_pass == normal_total) else "FAIL"
    print(f"  总体: {overall}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

