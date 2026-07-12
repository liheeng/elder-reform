import os, sys, json, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from features.feature_builder_v2 import FeatureBuilderV2, interpolate_missing_frames
from features.normalize import normalize_keypoints

PREVFALL_DIR = r"D:\Download\Pre-VFall\Pre-VFall"
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

WINDOW_SIZE = 30
STRIDE = 3
SCENE_ID = 10

SCENES = [
    "confusion_delirium",
    "confusion_nph",
    "dizziness_fall_forward",
    "dizziness_fall_side",
    "weakness_fall_forward",
    "weakness_fall_side",
]

CLASS_SAFE = 0
CLASS_FALLEN = 1

RISK_NORMAL = 0.0
RISK_ABNORMAL = 0.5
RISK_FALL = 1.0
NORMAL_TIME = 999

OPENPOSE_TO_COCO = {
    0: 0,    # Nose ‚Ü?Nose
    16: 1,   # LEye ‚Ü?LEye
    15: 2,   # REye ‚Ü?REye
    18: 3,   # LEar ‚Ü?LEar
    17: 4,   # REar ‚Ü?REar
    5: 5,    # LShoulder ‚Ü?LShoulder
    2: 6,    # RShoulder ‚Ü?RShoulder
    6: 7,    # LElbow ‚Ü?LElbow
    3: 8,    # RElbow ‚Ü?RElbow
    7: 9,    # LWrist ‚Ü?LWrist
    4: 10,   # RWrist ‚Ü?RWrist
    12: 11,  # LHip ‚Ü?LHip
    9: 12,   # RHip ‚Ü?RHip
    13: 13,  # LKnee ‚Ü?LKnee
    10: 14,  # RKnee ‚Ü?RKnee
    14: 15,  # LAnkle ‚Ü?LAnkle
    11: 16,  # RAnkle ‚Ü?RAnkle
}

def convert_openpose_to_coco(pose_keypoints_2d):
    kpts = np.zeros((17, 2), dtype=np.float32)
    confs = np.zeros(17, dtype=np.float32)
    for op_idx, coco_idx in OPENPOSE_TO_COCO.items():
        base = op_idx * 3
        x, y, c = pose_keypoints_2d[base], pose_keypoints_2d[base+1], pose_keypoints_2d[base+2]
        kpts[coco_idx] = [x, y]
        confs[coco_idx] = c
    return kpts, confs

def load_segment_jsons(segment_dir):
    files = []
    for fn in os.listdir(segment_dir):
        if fn.endswith(".json"):
            parts = fn.replace(".json", "").split("_")
            try:
                frame_num = int(parts[-2])
            except (ValueError, IndexError):
                continue
            files.append((frame_num, fn))
    files.sort(key=lambda x: x[0])

    kpts_list = []
    confs_list = []
    for frame_num, fn in files:
        with open(os.path.join(segment_dir, fn)) as f:
            data = json.load(f)
        if len(data["people"]) == 0:
            kpts_list.append(np.zeros((17, 2), dtype=np.float32))
            confs_list.append(np.zeros(17, dtype=np.float32))
            continue
        pose = data["people"][0]["pose_keypoints_2d"]
        kpts, confs = convert_openpose_to_coco(pose)
        kpts_list.append(kpts)
        confs_list.append(confs)

    return np.array(kpts_list, dtype=np.float32), np.array(confs_list, dtype=np.float32)

def build_segment_features(kpts, confs):
    kpts, confs = interpolate_missing_frames(kpts, confs)

    pos_list = []
    for i in range(len(kpts)):
        nk, nc = normalize_keypoints(kpts[i], confs[i])
        if nk is None:
            nk, nc = kpts[i], confs[i]
        feat = FeatureBuilderV2.build(nk, nc)
        pos_list.append(feat)
    pos = np.array(pos_list, dtype=np.float32)

    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([pos, vel, acc], axis=1).astype(np.float32)

def main():
    all_x = []
    all_classes = []
    all_risks = []
    all_times = []
    all_ids = []
    subject_counter = 0

    for scene_name in SCENES:
        scene_path = os.path.join(PREVFALL_DIR, scene_name)
        if not os.path.isdir(scene_path):
            print(f"[Ë∑≥Ëøá] {scene_path} ‰∏çÂ≠òÂú?)
            continue

        subjects = sorted([d for d in os.listdir(scene_path)
                          if os.path.isdir(os.path.join(scene_path, d))
                          and not d.endswith("-json")])

        for subject in subjects:
            subject_counter += 1
            subj_path = os.path.join(scene_path, subject)

            for segment_name, class_label, risk_label, time_label in [
                ("Normal",   CLASS_SAFE,   RISK_NORMAL,   999.0),
                ("Abnormal", CLASS_SAFE,   RISK_ABNORMAL,  999.0),
                ("Fall",     CLASS_FALLEN, RISK_FALL,       0.0),
            ]:
                seg_dir = os.path.join(subj_path, f"{subject}-{segment_name}-json")
                if not os.path.isdir(seg_dir):
                    continue

                kpts, confs = load_segment_jsons(seg_dir)
                if len(kpts) < WINDOW_SIZE:
                    continue

                features = build_segment_features(kpts, confs)
                T = features.shape[0]

                for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
                    end = start + WINDOW_SIZE
                    x_win = features[start:end]
                    all_x.append(x_win)
                    all_classes.append(class_label)
                    all_risks.append(risk_label)
                    all_times.append(time_label)
                    all_ids.append([SCENE_ID, subject_counter])

    if len(all_x) == 0:
        print("ÈîôËØØÔºöÊ≤°ÊúâÁîüÊàê‰ªª‰Ω?prevfall Ê†∑Êú¨Ôº?)
        sys.exit(1)

    xs = np.stack(all_x).astype(np.float32)
    classes = np.array(all_classes, dtype=np.int64)
    risks = np.array(all_risks, dtype=np.float32)
    times = np.array(all_times, dtype=np.float32)
    ids = np.array(all_ids, dtype=np.int64)

    np.save(os.path.join(PROCESSED_DIR, "prevfall_x_v2.npy"), xs)
    np.save(os.path.join(PROCESSED_DIR, "prevfall_class_v2.npy"), classes)
    np.save(os.path.join(PROCESSED_DIR, "prevfall_risk_v2.npy"), risks)
    np.save(os.path.join(PROCESSED_DIR, "prevfall_time_v2.npy"), times)
    np.save(os.path.join(PROCESSED_DIR, "prevfall_ids_v2.npy"), ids)

    print(f"Prevfall samples: {len(xs)}")
    print(f"  SAFE(0): {int((classes==0).sum())}")
    print(f"  FALLEN(1): {int((classes==1).sum())}")
    print(f"  risk=0.0: {int((risks==0.0).sum())}")
    print(f"  risk=0.4: {int((risks==0.4).sum())}")
    print(f"  risk=1.0: {int((risks==1.0).sum())}")

if __name__ == "__main__":
    main()

