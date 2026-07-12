"""
realtime_detect.py [V4 版]

流程: 视频帧 → YOLO-Pose → FeatureBuilderV2(64维) → SequenceBuffer(30帧) 
      → 拼接(30,192) → RiskClassifierV2(4-class+risk) → RiskEngineV2 → 状态
"""

import os
import cv2
import numpy as np
from pose.pose_extractor import PoseExtractor
from sequence.sequence_buffer import SequenceBufferV3
from features.feature_builder_v2 import FeatureBuilderV2
from features.normalize import normalize_keypoints
from classifier.lstm_classifier_v2 import LSTMClassifierV2
from pipeline.risk_engine_v2 import RiskEngineV2

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
POSE_MODEL = os.path.join(MODEL_DIR, "yolo11n-pose.pt")
LSTM_MODEL = os.path.join(MODEL_DIR, "lstm_v2.pt")
NORM_PARAMS = os.path.join(MODEL_DIR, "norm_v2.npz")
TEMP_PARAMS = os.path.join(MODEL_DIR, "temperature_v2.npz")

WINDOW_SIZE = 30          # 30帧窗口 (原生帧率, ~1.2秒)
FEATURE_DIM = 64           # 每帧64维 (FeatureBuilderV2)


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


def main(video_path=None):
    pose_ext = PoseExtractor(model_path=POSE_MODEL)
    buffer = SequenceBufferV3(seq_len=WINDOW_SIZE)
    classifier = LSTMClassifierV2(
        model_path=LSTM_MODEL,
        norm_path=NORM_PARAMS,
        temperature_path=TEMP_PARAMS,
    )
    risk_engine = RiskEngineV2()

    no_person_frames = 0
    last_state = "SAFE"
    last_bbox = np.array([0, 0, 100, 100])
    last_risk_score = 0.0
    danger_confirmed = False

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

    print(f"Source: {source_name}  ({fps:.1f}fps)")
    print(f"Model: {os.path.basename(LSTM_MODEL)}")
    print(f"Window: {WINDOW_SIZE} frames (~{WINDOW_SIZE/fps:.1f}s)")
    print("Press 'q' to quit\n")

    frame_count = 0
    last_good_feature = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        persons = pose_ext.extract(frame)

        # ---- 无人检测时处理 ----
        if len(persons) == 0:
            # 无人: 重用上一帧特征填补buffer, 防止 velocity 跳跃
            if last_good_feature is not None:
                buffer.update(0, last_good_feature)
            no_person_frames += 1
            if danger_confirmed and no_person_frames < 60:
                display_state = "DANGER"
                display_risk = last_risk_score
                display_bbox = last_bbox
                cv2.putText(frame, "FALL SUSPECTED (target lost)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                frame = draw_result(frame, display_bbox, display_state, display_risk, 0)
            elif no_person_frames >= 60:
                cv2.putText(frame, "Target lost", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
            else:
                cv2.putText(frame, "No person", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
        else:
            no_person_frames = 0
            for person in persons:
                kpts = person["keypoints"]
                bbox = person["bbox"]
                kpts_conf = person.get("kpts_conf", None)

                # 使用 FeatureBuilderV2 构建 64 维特征
                nk, nc = normalize_keypoints(kpts, kpts_conf)
                if nk is not None:
                    feature = FeatureBuilderV2.build(nk, nc)
                else:
                    feature = FeatureBuilderV2.build(kpts, kpts_conf)
                last_good_feature = feature

                person_id = 0
                buffer.update(person_id, feature)

                # 缓存满 WINDOW_SIZE 帧才推理
                if buffer.is_ready(person_id):
                    seq_192 = build_192_feature(buffer, person_id)
                    if seq_192 is not None and seq_192.shape == (WINDOW_SIZE, 192):
                        result = classifier.predict(seq_192)
                        rr = risk_engine.update(
                            person_id,
                            class_id=result["class_id"],
                            model_risk=result["risk"],
                        )

                        last_state = rr["state"]
                        last_risk_score = rr["risk_score"]
                        last_bbox = bbox
                        danger_confirmed = rr.get("danger_confirmed", False)
                        display_state = rr["state"]
                        display_risk = rr["risk_score"]

                        time_val = result.get("time_to_fall", 0)
                        frame = draw_result(frame, bbox, display_state, display_risk, person_id, time_val)

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
