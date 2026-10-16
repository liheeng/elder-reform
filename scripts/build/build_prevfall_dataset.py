import os, sys, json, argparse, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from features.feature_builder_v2 import FeatureBuilderV2, interpolate_missing_frames
from features.normalize import normalize_keypoints

# Pre-VFall 数据根目录（场景目录直接位于其下），可用 --data-dir 覆盖
PREVFALL_DIR = r"/Volumes/media-backup/AI-data/dataset/falldown/Pre-VFall"
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
    0: 0,    # Nose →Nose
    16: 1,   # LEye →LEye
    15: 2,   # REye →REye
    18: 3,   # LEar →LEar
    17: 4,   # REar →REar
    5: 5,    # LShoulder →LShoulder
    2: 6,    # RShoulder →RShoulder
    6: 7,    # LElbow →LElbow
    3: 8,    # RElbow →RElbow
    7: 9,    # LWrist →LWrist
    4: 10,   # RWrist →RWrist
    12: 11,  # LHip →LHip
    9: 12,   # RHip →RHip
    13: 13,  # LKnee →LKnee
    10: 14,  # RKnee →RKnee
    14: 15,  # LAnkle →LAnkle
    11: 16,  # RAnkle →RAnkle
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
        # PV 来自 OpenPose JSON，根本没有检测框 score。显式传 has_score=0
        # 让模型知道这一路的 0 是"缺信号"，不是"框分极低"——否则这 20% 数据
        # 会被 score 通道推着往 NON_PERSON 靠，反而污染风险头那 23.6% 的监督。
        feat = FeatureBuilderV2.build(nk, nc, score=0.0, has_score=0.0)
        pos_list.append(feat)
    pos = np.array(pos_list, dtype=np.float32)

    vel = np.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]
    acc = np.zeros_like(vel)
    acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([pos, vel, acc], axis=1).astype(np.float32)

SEGMENT_TYPES = [
    ("Normal",   CLASS_SAFE,   RISK_NORMAL,   999.0),
    ("Abnormal", CLASS_SAFE,   RISK_ABNORMAL,  999.0),
    ("Fall",     CLASS_FALLEN, RISK_FALL,       0.0),
]


def find_segment_dir(subj_path, subject, segment_name):
    """定位段落目录，兼容两种命名分隔符。

    多数场景为 `<subject>-<Segment>-json`（连字符），但 confusion_delirium
    整个场景用的是 `<subject>_<Segment>-json`（下划线），需同时支持。

    数据本身存在一处错位：confusion_delirium/CLFM2/ 下混有 CLFM1_Abnormal-json。
    这里只接受前缀与受试者名一致的目录，遇到前缀不符的打印警告并跳过，
    避免把它错记成 CLFM2 的样本（同时 CLFM1 自己仍有该段落目录）。
    """
    dirs = os.listdir(subj_path) if os.path.isdir(subj_path) else []
    for sep in ("-", "_"):
        expected = f"{subject}{sep}{segment_name}-json"
        if expected in dirs:
            return os.path.join(subj_path, expected)
    # 回退：容错模糊匹配（应对个别命名差异），但拒绝前缀不符者
    for d in sorted(dirs):
        if not (d.endswith(f"{segment_name}-json") and os.path.isdir(os.path.join(subj_path, d))):
            continue
        prefix = d[:-(len(segment_name) + 6)].rstrip("-_")
        if prefix != subject:
            print(f"  [警告] 前缀与受试者不符，跳过: {subj_path}/{d}")
            continue
        return os.path.join(subj_path, d)
    return None


def main():
    parser = argparse.ArgumentParser(description="Pre-VFall (OpenPose JSON) -> V2 训练数据转换")
    parser.add_argument("--data-dir", default=PREVFALL_DIR,
                        help="Pre-VFall 数据根目录，默认 " + PREVFALL_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir

    if not os.path.isdir(data_dir):
        print(f"[错误] Pre-VFall 数据目录不存在: {data_dir}")
        print("       请用 --data-dir 指定（应包含 confusion_delirium 等场景目录）")
        sys.exit(1)

    all_x = []
    all_classes = []
    all_risks = []
    all_times = []
    all_ids = []
    subject_counter = 0
    skipped = []

    for scene_name in SCENES:
        scene_path = os.path.join(data_dir, scene_name)
        if not os.path.isdir(scene_path):
            print(f"[跳过] {scene_path} 不存在")
            continue

        subjects = sorted([d for d in os.listdir(scene_path)
                          if os.path.isdir(os.path.join(scene_path, d))
                          and not d.endswith("-json")])

        for subject in subjects:
            subject_counter += 1
            subj_path = os.path.join(scene_path, subject)

            for segment_name, class_label, risk_label, time_label in SEGMENT_TYPES:
                seg_dir = find_segment_dir(subj_path, subject, segment_name)
                if seg_dir is None:
                    skipped.append((scene_name, subject, segment_name))
                    continue

                kpts, confs = load_segment_jsons(seg_dir)
                if len(kpts) < WINDOW_SIZE:
                    skipped.append((scene_name, subject, segment_name, f"仅 {len(kpts)} 帧"))
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
        print("错误：没有生成任何prevfall 样本？")
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
    print(f"  risk=0.0 (Normal):   {int((risks==0.0).sum())}   (RISK_NORMAL={RISK_NORMAL})")
    print(f"  risk=0.5 (Abnormal): {int((risks==RISK_ABNORMAL).sum())}   (RISK_ABNORMAL={RISK_ABNORMAL})")
    print(f"  risk=1.0 (Fall):     {int((risks==1.0).sum())}   (RISK_FALL={RISK_FALL})")
    if skipped:
        print(f"\n跳过 {len(skipped)} 个段落:")
        for item in skipped[:20]:
            print("  " + " / ".join(str(x) for x in item))
        if len(skipped) > 20:
            print(f"  ... 其余 {len(skipped) - 20} 项省略")

if __name__ == "__main__":
    main()

