"""
verify_issues.py [REFORM 版]

多场景验证 检查P1(物品误识为人) + P2(坐下误识为跌倒.
"""

import os, sys, cv2, json, pickle, random, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from ultralytics import YOLO
from sequence.sequence_buffer import SequenceBufferV3
from classifier.lstm_classifier import LSTMClassifier
from pipeline.risk_engine import RiskEngine
from utils.device import resolve_device

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
YOLO_MODEL = os.path.join(PROJECT_ROOT, "models", "yolo26n-pose.pt")
LSTM_MODEL = os.path.join(PROJECT_ROOT, "models", "lstm_4class.pt")
NORM_PARAMS = os.path.join(PROJECT_ROOT, "models", "norm_4class.npz")
if not os.path.exists(LSTM_MODEL):
    LSTM_MODEL = os.path.join(PROJECT_ROOT, "models", "lstm_multitask.pt")
    NORM_PARAMS = os.path.join(PROJECT_ROOT, "models", "norm_params.npz")

ARCHIVE = r"C:\Users\y'y'x\Downloads\archive"
WINDOW = 30
CACHE_DIR = os.path.join(PROJECT_ROOT, "verify_issues_cache")

# 运行时解析（CUDA -> MPS -> CPU）
YOLO_DEVICE = resolve_device("auto")
os.makedirs(CACHE_DIR, exist_ok=True)

SAMPLE_PER_SCENE = 8
SEED = 20260710

SCENES = {
    "Coffee_room_01": (os.path.join(ARCHIVE, "Coffee_room_01", "Coffee_room_01", "Videos"),
                       os.path.join(ARCHIVE, "Coffee_room_01", "Coffee_room_01", "Annotation_files")),
    "Coffee_room_02": (os.path.join(ARCHIVE, "Coffee_room_02", "Coffee_room_02", "Videos"),
                       os.path.join(ARCHIVE, "Coffee_room_02", "Coffee_room_02", "Annotations_files")),
    "Home_01": (os.path.join(ARCHIVE, "Home_01", "Home_01", "Videos"),
                os.path.join(ARCHIVE, "Home_01", "Home_01", "Annotation_files")),
    "Home_02": (os.path.join(ARCHIVE, "Home_02", "Home_02", "Videos"),
                os.path.join(ARCHIVE, "Home_02", "Home_02", "Annotation_files")),
    "Lecture_room": (os.path.join(ARCHIVE, "Lecture_room", "Lecture room"), None),
    "Office": (os.path.join(ARCHIVE, "Office", "Office"), None),
}


def parse_annotation(anno_path):
    if anno_path is None or not os.path.exists(anno_path):
        return None, None, {}
    with open(anno_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    fall_start, fall_end = None, None
    frame_dict = {}
    pure_ints = []
    for ln in lines:
        parts = ln.split(",")
        if len(parts) == 1:
            try: pure_ints.append(int(parts[0]))
            except: pass
        elif len(parts) == 6:
            try:
                fidx = int(parts[0]); label = int(parts[1])
                x1, y1, x2, y2 = map(float, parts[2:])
                frame_dict[fidx] = (label, x1, y1, x2, y2)
            except: pass
    if len(pure_ints) >= 2:
        fall_start, fall_end = pure_ints[0], pure_ints[1]
    if fall_start == 0 and fall_end == 0:
        fall_start, fall_end = None, None
    return fall_start, fall_end, frame_dict


def list_videos():
    out = []
    for scene, (vdir, adir) in SCENES.items():
        if not os.path.isdir(vdir):
            continue
        for fn in sorted(os.listdir(vdir)):
            if not fn.endswith(".avi"):
                continue
            vid = fn.replace("video (", "").replace(").avi", "")
            vpath = os.path.join(vdir, fn)
            apath = None
            if adir:
                cand = os.path.join(adir, "video (%s).txt" % vid)
                if os.path.exists(cand):
                    apath = cand
            out.append((scene, vid, vpath, apath))
    return out


def sample_videos(all_videos, n_per_scene, seed):
    rng = random.Random(seed)
    by_scene = {}
    for v in all_videos:
        by_scene.setdefault(v[0], []).append(v)
    picked = []
    for scene in sorted(by_scene.keys()):
        lst = by_scene[scene]
        rng.shuffle(lst)
        picked.extend(lst[:n_per_scene])
    for v in all_videos:
        if v[0] == "Coffee_room_02" and v[1] == "50" and v not in picked:
            picked.append(v)
            break
    return picked


def extract_perframe(video_path, conf=0.05, iou=0.5):
    key = os.path.basename(video_path).replace(" ", "_").replace("(", "").replace(")", "")
    cache_path = os.path.join(CACHE_DIR, "%s_c%d_i%d.pkl" % (key, int(conf * 100), int(iou * 100)))
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(video_path)
    frames_data = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        res = model(frame, verbose=False, conf=conf, iou=iou, device=YOLO_DEVICE)
        persons = []
        if res[0].boxes is not None and len(res[0].boxes) > 0:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            scores = res[0].boxes.conf.cpu().numpy()
            kpts_xy = res[0].keypoints.xy.cpu().numpy()
            kpts_conf = None
            if res[0].keypoints.conf is not None:
                kpts_conf = res[0].keypoints.conf.cpu().numpy()
            for i in range(len(boxes)):
                kc = kpts_conf[i] if kpts_conf is not None else None
                persons.append({"bbox": boxes[i].tolist(), "score": float(scores[i]),
                                "kpts": kpts_xy[i].tolist(), "kpts_conf": kc.tolist() if kc is not None else None})
        frames_data.append({"h": h, "w": w, "persons": persons})
    cap.release()
    with open(cache_path, "wb") as f:
        pickle.dump(frames_data, f)
    return frames_data


def run_pipeline(frames_data, merge_iou=0.3, apply_quality_filter=True):
    clf = LSTMClassifier(model_path=LSTM_MODEL, norm_path=NORM_PARAMS)
    buffer = SequenceBufferV3(seq_len=WINDOW)
    engine = RiskEngine()
    from pose.pose_extractor import _merge_overlapping_detections, _is_valid_person

    states, n_persons_list, best_bboxes = [], [], []
    danger_confirmed = False
    last_state = "SAFE"
    last_bbox = None
    last_risk = 0.0
    no_person = 0

    for fd in frames_data:
        h = fd["h"]
        persons = []
        for p in fd["persons"]:
            kpts = np.array(p["kpts"], dtype=np.float32)
            bbox = np.array(p["bbox"], dtype=np.float32)
            kc = p.get("kpts_conf")
            kc_arr = np.array(kc, dtype=np.float32) if kc is not None else None
            if apply_quality_filter and not _is_valid_person(kpts, kc_arr, bbox, score=p["score"]):
                continue
            persons.append({"bbox": bbox, "keypoints": kpts, "score": p["score"]})
        persons = _merge_overlapping_detections(persons, iou_threshold=merge_iou)

        if len(persons) == 0:
            no_person += 1
            if no_person < 60 and danger_confirmed:
                st, rk, bb = "DANGER", last_risk, last_bbox
            elif no_person >= 60:
                st, rk, bb = "SAFE", 0.0, None
                last_state = "SAFE"
            else:
                st, rk, bb = "SAFE", 0.0, None
                last_state = "SAFE"
            states.append(st); n_persons_list.append(0); best_bboxes.append(bb)
            continue

        no_person = 0
        persons.sort(key=lambda p: p["score"], reverse=True)
        best = persons[0]
        kpts = best["keypoints"]
        bbox = best["bbox"]
        buffer.update(0, kpts.reshape(-1))
        st, rk = "SAFE", 0.0
        if buffer.is_ready(0):
            raw = buffer.get_sequence(0)
            vel = buffer.get_velocity(0)
            acc = buffer.get_acceleration(0)
            seq102 = np.concatenate([raw, vel, acc], axis=1)
            res = clf.predict(seq102)
            rr = engine.update(
                0,
                class_probs=res.get("probs"),
                class_id=res.get("class_id"),
                danger_prob=res.get("risk"),
            )
            last_state = rr["state"]
            last_risk = rr["risk_score"]
            last_bbox = bbox
            danger_confirmed = rr.get("danger_confirmed", False)
            st, rk = rr["state"], rr["risk_score"]
        states.append(st)
        n_persons_list.append(len(persons))
        best_bboxes.append(bbox)

    return {"states": states, "n_persons": n_persons_list, "bboxes": best_bboxes}


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def analyze(scene, vid, pipe, fall_start, fall_end, frame_dict):
    states = pipe["states"]
    n_persons = pipe["n_persons"]
    bboxes = pipe["bboxes"]
    T = len(states)

    p1_multi = p1_low_iou = p1_matched = 0
    for t in range(T):
        fidx = t + 1
        ann = frame_dict.get(fidx)
        n_det = n_persons[t]
        if n_det > 1:
            p1_multi += 1
        if ann is not None:
            _, ax1, ay1, ax2, ay2 = ann
            ann_bb = [ax1, ay1, ax2, ay2]
            if (ax2 - ax1) > 2 and (ay2 - ay1) > 2:
                if n_det > 0 and bboxes[t] is not None:
                    iv = iou(ann_bb, bboxes[t])
                    if iv < 0.1:
                        p1_low_iou += 1
                    else:
                        p1_matched += 1

    p2_prefall = p2_nonfall = p2_p1_induced = p2_true = p2_total = 0
    for t in range(T):
        if states[t] != "DANGER":
            continue
        p2_total += 1
        fidx = t + 1
        ann = frame_dict.get(fidx)
        is_fall_vid = fall_start is not None and fall_end is not None
        if is_fall_vid:
            if fidx < fall_start - 30:
                p2_prefall += 1
        else:
            p2_nonfall += 1
            if ann is not None and bboxes[t] is not None:
                _, ax1, ay1, ax2, ay2 = ann
                if (ax2 - ax1) > 2 and (ay2 - ay1) > 2:
                    if iou([ax1, ay1, ax2, ay2], bboxes[t]) < 0.1:
                        p2_p1_induced += 1
                    else:
                        p2_true += 1
                else:
                    p2_true += 1
            else:
                p2_true += 1

    return {"scene": scene, "vid": vid, "T": T, "fall_start": fall_start, "fall_end": fall_end,
            "p1_multi": p1_multi, "p1_low_iou": p1_low_iou, "p1_matched": p1_matched,
            "p2_prefall": p2_prefall, "p2_nonfall": p2_nonfall, "p2_p1_induced": p2_p1_induced,
            "p2_true": p2_true, "p2_total": p2_total}


def main():
    print("=" * 80)
    print("P1 + P2 验证 [REFORM]")
    print("=" * 80)
    print(f"Model: {os.path.basename(LSTM_MODEL)}")

    all_vids = list_videos()
    from collections import Counter
    cnt = Counter(v[0] for v in all_vids)
    print("total: %d (%s)" % (len(all_vids), ", ".join("%s=%d" % (k, v) for k, v in sorted(cnt.items()))))
    sampled = sample_videos(all_vids, SAMPLE_PER_SCENE, SEED)
    print(f"\nsample (seed={SEED}, {SAMPLE_PER_SCENE} per scene):")
    for scene, vid, vp, ap in sampled:
        print(f"  {scene} video ({vid})")

    results = []
    for i, (scene, vid, vp, ap) in enumerate(sampled):
        print(f"\n[{i+1}/{len(sampled)}] {scene} video ({vid}) ...", flush=True)
        fall_start, fall_end, frame_dict = parse_annotation(ap)
        frames_data = extract_perframe(vp)
        pipe = run_pipeline(frames_data)
        r = analyze(scene, vid, pipe, fall_start, fall_end, frame_dict)
        results.append(r)
        stc = Counter(pipe["states"])
        n_det = sum(1 for n in pipe["n_persons"] if n > 0)
        multi = sum(1 for n in pipe["n_persons"] if n > 1)
        print(f"  T={r['T']} fall=[{r['fall_start']},{r['fall_end']}]")
        print(f"  states: {dict(stc)}")
        print(f"  detect={n_det}/{r['T']} multi={multi}")
        print(f"  P1: multi={r['p1_multi']} lowIoU={r['p1_low_iou']} matched={r['p1_matched']}")
        print(f"  P2: DANGER={r['p2_total']} prefall={r['p2_prefall']} nonfall={r['p2_nonfall']}(p1ind={r['p2_p1_induced']},true={r['p2_true']})")

    print(f"\n{'='*80}")
    print("summary")
    print("="*80)
    print(f"{'scene':16} {'vid':5} {'fall':12} {'T':5} {'multi':6} {'lowIoU':6} "
          f"{'DANGER':7} {'prefall':8} {'nonfall':8} {'p1ind':6} {'trueP2':5} {'verdict':6}")
    p1p = p2p = 0
    for r in results:
        fall_str = f"[{r['fall_start']},{r['fall_end']}]" if r['fall_start'] else "--"
        p1_hit = (r['p1_multi'] > max(5, 0.1 * r['T']) or r['p1_low_iou'] > max(5, 0.1 * r['T']))
        p2_hit = (r['p2_prefall'] > 0 or r['p2_true'] > 0)
        if p1_hit: p1p += 1
        if p2_hit: p2p += 1
        tag = "P1+P2" if (p1_hit and p2_hit) else ("P1" if p1_hit else ("P2" if p2_hit else "OK"))
        print(f"{r['scene']:16} {r['vid']:5} {fall_str:12} {r['T']:5} {r['p1_multi']:6} "
              f"{r['p1_low_iou']:6} {r['p2_total']:7} {r['p2_prefall']:8} {r['p2_nonfall']:8} "
              f"{r['p2_p1_induced']:6} {r['p2_true']:5} {tag:6}")

    print(f"\nP1 (objects as person): {p1p}/{len(results)}")
    print(f"P2 (true sit-as-fall): {p2p}/{len(results)}")

    out_json = os.path.join(PROJECT_ROOT, "verify_issues_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nresult saved: {out_json}")
    print(f"Model: {os.path.basename(LSTM_MODEL)}")


if __name__ == "__main__":
    main()

