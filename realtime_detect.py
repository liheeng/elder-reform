"""
realtime_detect.py [V4 版]

流程: 视频帧 → YOLO-Pose → PersonTracker(分配 ID) → FeatureBuilderV2(64维)
      → SequenceBuffer(每人 30 帧) → 拼接(30,192) → RiskClassifierV2(4-class+risk)
      → RiskEngineV2 → 状态
"""

import os
import cv2
import numpy as np
from pose.pose_extractor import PoseExtractor
from tracker.person_tracker import PersonTracker
from sequence.sequence_buffer import SequenceBufferV3
from features.feature_builder_v2 import FeatureBuilderV2
from features.normalize import normalize_keypoints
from classifier.lstm_classifier_v2 import LSTMClassifierV2
from pipeline.risk_engine_v2 import RiskEngineV2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
POSE_MODEL = os.path.join(MODEL_DIR, "yolo26n-pose.pt")
# 2026-10-15 起启用含 NON_PERSON 分片的 v2 模型（此前该类样本数为 0，等于没有出口）。
# 切换前后对比见 docs/tuning-log.md；改名前的旧模型备份在 models/backup_pre_nonperson/。
LSTM_MODEL = os.path.join(MODEL_DIR, "lstm_v2_nonperson.pt")
NORM_PARAMS = os.path.join(MODEL_DIR, "norm_v2_nonperson.npz")
TEMP_PARAMS = os.path.join(MODEL_DIR, "temperature_v2_nonperson.npz")

WINDOW_SIZE = 30          # 30帧窗口 (原生帧率, ~1.2秒)
FEATURE_DIM = 64           # 每帧64维 (FeatureBuilderV2)
MISSING_TOLERANCE = 60     # 目标连续丢失多少帧后放弃该 ID


def build_192_feature(buffer, person_id):
    """拼接 pos(64) + vel(64) + acc(64) → (T, 192)"""
    raw = buffer.get_sequence(person_id)
    if raw is None:
        return None
    if len(raw) < 2:
        vel = np.zeros_like(raw)
        acc = np.zeros_like(raw)
    else:
        vel = np.zeros_like(raw)
        vel[1:] = raw[1:] - raw[:-1]
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([raw, vel, acc], axis=1).astype(np.float32)


def draw_result(frame, bbox, state, risk_score, person_id, time_to_fall=0):
    x1, y1, x2, y2 = map(int, bbox)
    if state == "DANGER":
        color, status = (0, 0, 255), "FALL"
    elif state == "WARNING":
        color, status = (0, 165, 255), "WARNING"
    else:
        color, status = (0, 255, 0), "SAFE"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID:{person_id} {status}", (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"risk:{risk_score:.3f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"t:{time_to_fall:.0f}fr", (x1, y1 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


def _draw_dashed_rect(frame, p1, p2, color, thickness=2, dash=12):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_lost_result(frame, bbox, risk_score, person_id, missing_frames):
    """目标已丢失: 灰色虚线框 + 最后已知状态, 与实时检测结果在视觉上区分开。"""
    x1, y1, x2, y2 = map(int, bbox)
    color = (128, 128, 128)
    _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"ID:{person_id} FALL (lost {missing_frames}f)", (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"risk:{risk_score:.3f} (last seen)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def _create_pipeline():
    pose_ext = PoseExtractor(model_path=POSE_MODEL)
    tracker = PersonTracker(max_missing=MISSING_TOLERANCE)
    buffer = SequenceBufferV3(seq_len=WINDOW_SIZE)
    classifier = LSTMClassifierV2(
        model_path=LSTM_MODEL,
        norm_path=NORM_PARAMS,
        temperature_path=TEMP_PARAMS,
    )
    risk_engine = RiskEngineV2()

    # 按 ID 保存最近特征和报警状态，供短暂漏检时继续维护时序。
    last_feature = {}
    display = {}
    return pose_ext, tracker, buffer, classifier, risk_engine, last_feature, display


def _open_video(video_path):
    if video_path is None:
        cap = cv2.VideoCapture(0)
        source_name = "Camera"
    else:
        cap = cv2.VideoCapture(video_path)
        source_name = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open {source_name}")
        return None, source_name
    return cap, source_name


def _draw_frame_status(frame, tracks, display, has_live_target, no_person_frames):
    # 丢失目标仍有跌倒报警时，优先显示报警而不是普通的无人提示。
    lost_danger = [track for track in tracks
                   if track["missing"] > 0
                   and display.get(track["id"], {}).get("state") == "DANGER"]

    if lost_danger:
        cv2.putText(frame, "FALL SUSPECTED (target lost)", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif not has_live_target:
        if no_person_frames >= MISSING_TOLERANCE:
            cv2.putText(frame, "Target lost", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        else:
            cv2.putText(frame, "No person", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)


def _process_live_track(frame, track, buffer, classifier, risk_engine,
                        last_feature, display):
    person_id = track["id"]
    bbox = track["bbox"]
    kpts = track["keypoints"]
    kpts_conf = track["kpts_conf"]
    bbox_score = float(track.get("score", 0.0))

    # 按权重版本匹配 score 输入：启用、训练时固定关闭，或旧版权重。
    if getattr(classifier, "uses_score", False):
        feature_kwargs = {"score": bbox_score, "has_score": 1.0}
    elif getattr(classifier, "score_dims_pinned", False):
        feature_kwargs = {"score": 0.0, "has_score": 0.0}
    else:
        feature_kwargs = {}

    nk, nc = normalize_keypoints(kpts, kpts_conf)
    if nk is not None:
        feature = FeatureBuilderV2.build(nk, nc, **feature_kwargs)
    else:
        feature = FeatureBuilderV2.build(kpts, kpts_conf, **feature_kwargs)
    last_feature[person_id] = feature
    buffer.update(person_id, feature)

    # 每个 ID 独立累积窗口；窗口就绪后才进行推理。
    if not buffer.is_ready(person_id):
        return frame

    seq_192 = build_192_feature(buffer, person_id)
    if seq_192 is not None and seq_192.shape == (WINDOW_SIZE, 192):
        result = classifier.predict(seq_192)
        rr = risk_engine.update(
            person_id,
            class_id=result["class_id"],
            model_risk=result["risk"],
        )

        time_val = result.get("time_to_fall", 0)
        display[person_id] = {
            "state": rr["state"],
            "risk": rr["risk_score"],
        }
        frame = draw_result(frame, bbox, rr["state"], rr["risk_score"],
                            person_id, time_val)
    return frame


def _process_track(frame, track, has_live_target, buffer, classifier,
                   risk_engine, last_feature, display):
    person_id = track["id"]
    bbox = track["bbox"]

    if track["missing"] > 0:
        # 用上一帧特征填充，避免漏检造成虚假的速度突变。
        if person_id in last_feature:
            buffer.update(person_id, last_feature[person_id])
        info = display.get(person_id)
        if info is None or info["state"] != "DANGER":
            return frame
        # 只有当前没有实时目标时才画历史框，避免被误认成实时检测框。
        if not has_live_target:
            frame = draw_lost_result(
                frame, bbox, info["risk"], person_id, track["missing"])
        return frame

    return _process_live_track(
        frame, track, buffer, classifier, risk_engine, last_feature, display)


def _process_frame(frame, pose_ext, tracker, buffer, classifier, risk_engine,
                   last_feature, display, no_person_frames):
    persons = pose_ext.extract(frame)
    tracks, removed_ids = tracker.update(persons)
    has_live_target = len(persons) > 0

    # 跟踪器永久移除 ID 后，清理该目标关联的全部状态。
    for person_id in removed_ids:
        buffer.remove(person_id)
        risk_engine.remove(person_id)
        last_feature.pop(person_id, None)
        display.pop(person_id, None)

    if has_live_target:
        no_person_frames = 0
    else:
        no_person_frames += 1
    _draw_frame_status(
        frame, tracks, display, has_live_target, no_person_frames)

    for track in tracks:
        frame = _process_track(
            frame, track, has_live_target, buffer, classifier, risk_engine,
            last_feature, display)
    return frame, no_person_frames


def _release_video(cap):
    cap.release()
    cv2.destroyAllWindows()


def main(video_path=None):
    (pose_ext, tracker, buffer, classifier, risk_engine,
     last_feature, display) = _create_pipeline()
    cap, source_name = _open_video(video_path)
    if cap is None:
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    print(f"Source: {source_name}  ({fps:.1f}fps)")
    print(f"Model: {os.path.basename(LSTM_MODEL)}")
    print(f"Window: {WINDOW_SIZE} frames (~{WINDOW_SIZE/fps:.1f}s)")
    print("Press 'q' to quit\n")

    frame_count = 0
    no_person_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            frame, no_person_frames = _process_frame(
                frame, pose_ext, tracker, buffer, classifier, risk_engine,
                last_feature, display, no_person_frames)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Fall Detection", frame)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
    finally:
        _release_video(cap)

    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else None
    main(video)
