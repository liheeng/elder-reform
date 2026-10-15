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


def main(video_path=None):
    pose_ext = PoseExtractor(model_path=POSE_MODEL)
    tracker = PersonTracker(max_missing=MISSING_TOLERANCE)
    buffer = SequenceBufferV3(seq_len=WINDOW_SIZE)
    classifier = LSTMClassifierV2(
        model_path=LSTM_MODEL,
        norm_path=NORM_PARAMS,
        temperature_path=TEMP_PARAMS,
    )
    risk_engine = RiskEngineV2()

    # 每个 ID 一份: 最近一帧可用特征(丢失时用于填充)/最近一次推理结果(丢失后继续显示)
    last_feature = {}
    display = {}

    if video_path is None:
        cap = cv2.VideoCapture(0)
        source_name = "Camera"
    else:
        cap = cv2.VideoCapture(video_path)
        source_name = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open {source_name}")
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        persons = pose_ext.extract(frame)
        tracks, removed_ids = tracker.update(persons)
        has_live_target = len(persons) > 0

        # ---- 跟踪器判定该目标已消失: 清理其全部状态 ----
        for person_id in removed_ids:
            buffer.remove(person_id)
            risk_engine.remove(person_id)
            last_feature.pop(person_id, None)
            display.pop(person_id, None)

        # ---- 状态栏: 丢失中的跌倒目标优先于"无人"提示 ----
        lost_danger = [t for t in tracks
                       if t["missing"] > 0
                       and display.get(t["id"], {}).get("state") == "DANGER"]

        if not has_live_target:
            no_person_frames += 1
        else:
            no_person_frames = 0

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

        for track in tracks:
            person_id = track["id"]
            bbox = track["bbox"]

            # ---- 本帧未检出该目标 ----
            if track["missing"] > 0:
                # 用上一帧特征填充, 防止 velocity 通道出现人为跳变
                if person_id in last_feature:
                    buffer.update(person_id, last_feature[person_id])
                info = display.get(person_id)
                if info is None or info["state"] != "DANGER":
                    continue
                # 画面里还有已检出的活体目标时只留报警文字, 不画历史框,
                # 否则一个停在旧位置的框会被误当成实时检测结果
                if not has_live_target:
                    draw_lost_result(frame, bbox, info["risk"], person_id,
                                     track["missing"])
                continue

            kpts = track["keypoints"]
            kpts_conf = track["kpts_conf"]

            # 使用 FeatureBuilderV2 构建 64 维特征
            nk, nc = normalize_keypoints(kpts, kpts_conf)
            if nk is not None:
                feature = FeatureBuilderV2.build(nk, nc)
            else:
                feature = FeatureBuilderV2.build(kpts, kpts_conf)
            last_feature[person_id] = feature

            buffer.update(person_id, feature)

            # 每人各自缓存满 WINDOW_SIZE 帧才推理
            if not buffer.is_ready(person_id):
                continue

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

        cv2.putText(frame, f"Frame: {frame_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Fall Detection", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else None
    main(video)
