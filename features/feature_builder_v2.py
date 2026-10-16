import numpy as np

# 单帧特征 64 维布局：[0,51) = 17 x {x, y, conf}；[51,57) = 原始几何；[57,64) = 增强几何。
#
# 51 与 57 是"恒定废维"：归一化以 hip_center 为原点，故 cg_y 恒为 0（51），
# cg_y_norm = cg_y / body_height 亦恒为 0（57）。（只有 normalize_keypoints 返回
# None 的回退帧会把原始像素塞进来，约占 0.03%，正是旧数据集这两维 std≈14 的来源。）
# 首版遗留的这两维全仓无人读取（见 docs/architecture.md §8.1.2），故原地复用来
# 承载检测框 score 与其有效性掩码，从而在不改变 192 维输入形状、不移动任何既有
# 下标的前提下新增一路信号。
#
# ⚠️ 不要复用 52/53/54/61/62：build_dataset_v2.is_sitting_posture_v2() 按下标读它们。
SCORE_DIM = 51       # 原 cg_y 恒定废维 -> 检测框 score
HAS_SCORE_DIM = 57   # 原 cg_y_norm 恒定废维 -> score 有效标志（1=可信，0=无此路信号）

PER_FRAME_DIM = 64   # 单帧特征维度；窗口特征 = pos/vel/acc 三段拼起来


def score_channel_window_dims(num_segments=3):
    """score 通道在窗口特征 `(T, num_segments * PER_FRAME_DIM)` 里占据的全部下标。

    pos / vel / acc 三段各含 `SCORE_DIM` 与 `HAS_SCORE_DIM` 两份，共 6 个下标。
    要屏蔽这一路信号时必须六维一起处理：只清 pos 那两份的话，vel / acc 里仍留着
    score 的差分与二阶差分，受控对照就漏气了。
    """
    return tuple(seg * PER_FRAME_DIM + d
                 for seg in range(num_segments)
                 for d in (SCORE_DIM, HAS_SCORE_DIM))


class FeatureBuilderV2:
    NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
    LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
    LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16

    @staticmethod
    def _angle(a, b, c):
        """三点夹角 ∠abc（度数）。"""
        ba, bc = a - b, c - b
        nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
        if nba < 1e-6 or nbc < 1e-6:
            return 0.0
        cos = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos)))

    @staticmethod
    def _angle_from_vertical(a, b):
        """线段 ab 与垂直方向(向下)的夹角（度数）。"""
        vec = b - a
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return 0.0
        cos_vert = np.clip(vec[1] / norm, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_vert)))

    @staticmethod
    def build(kpts, confs=None, score=None, has_score=None):
        """
        Parameters
        ----------
        kpts : ndarray (17, 2)  COCO 关键点 x,y
        confs : ndarray (17,) or None  关键点置信度
        score : float or None  检测框 score；None 则保持该维原常量值（0.0）
        has_score : float or None  score 是否可信；None 则保持原常量值（0.0）。
            仅由 PV(OpenPose) 数据显式传 0.0 —— 它没有框分，必须让模型知道
            这一路的 0 是"缺信号"而不是"框分极低"，否则会被推着判成 NON_PERSON。

        Returns
        -------
        ndarray (64,)
        """
        if confs is None:
            confs = np.ones(17, dtype=np.float32)
        # 51 dim: 17 x {x, y, conf}
        base = np.concatenate([kpts, confs.reshape(-1, 1)], axis=1).reshape(-1)

        lh, rh = kpts[11], kpts[12]
        ls, rs = kpts[5],  kpts[6]
        lk, rk = kpts[13], kpts[14]
        la, ra = kpts[15], kpts[16]
        hip_center = (lh + rh) / 2
        shoulder_center = (ls + rs) / 2

        # === 原始几何 (6 维) ===
        cg_y = float(hip_center[1])
        dx, dy = float(shoulder_center[0] - hip_center[0]), float(shoulder_center[1] - hip_center[1])
        torso_angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
        left_knee_angle = FeatureBuilderV2._angle(lh, lk, la)
        right_knee_angle = FeatureBuilderV2._angle(rh, rk, ra)
        foot_distance = float(np.linalg.norm(la - ra))
        body_height = float(np.linalg.norm(kpts[0] - (la + ra) / 2))

        geo_original = np.array(
            [cg_y, torso_angle, left_knee_angle, right_knee_angle,
             foot_distance, body_height], dtype=np.float32
        )

        # === 增强几何 (7 维) ===
        bh = max(body_height, 1.0)
        cg_y_norm = cg_y / bh
        foot_norm = foot_distance / bh
        com_x = float((lh[0] + rh[0] + ls[0] + rs[0]) / 4.0)
        com_y = float((lh[1] + rh[1] + ls[1] + rs[1]) / 4.0)
        left_hip_angle = FeatureBuilderV2._angle(shoulder_center, lh, lk)
        right_hip_angle = FeatureBuilderV2._angle(shoulder_center, rh, rk)

        geo_new = np.array(
            [cg_y_norm, foot_norm, com_x, com_y,
             left_hip_angle, right_hip_angle, torso_angle], dtype=np.float32
        )

        feat = np.concatenate([base, geo_original, geo_new]).astype(np.float32)
        if score is not None:
            feat[SCORE_DIM] = score
        if has_score is not None:
            feat[HAS_SCORE_DIM] = has_score
        return feat


def _fill_neighbors(arr, valid_mask, tail_scale=1.0):
    """把 arr 中 valid_mask 为假的帧按最近有效邻居填充。

    两侧都有邻居时线性插值；只有一侧时复制该侧值再乘 tail_scale
    （关键点用 1.0，置信度类用 0.3 —— 单侧填充属外推，需要自降可信度）。
    """
    out = np.array(arr)
    T = out.shape[0]
    for t in range(T):
        if valid_mask[t]:
            continue
        prev_t = t - 1
        while prev_t >= 0 and not valid_mask[prev_t]:
            prev_t -= 1
        next_t = t + 1
        while next_t < T and not valid_mask[next_t]:
            next_t += 1

        if prev_t >= 0 and next_t < T:
            alpha = (t - prev_t) / max(next_t - prev_t, 1)
            out[t] = out[prev_t] * (1 - alpha) + out[next_t] * alpha
        elif prev_t >= 0:
            out[t] = out[prev_t] * tail_scale
        elif next_t < T:
            out[t] = out[next_t] * tail_scale
    return out


def interpolate_missing_frames(keypoints, confs, scores=None):
    """
    修复 YOLO 漏检导致的零填充帧。

    keypoints: (T, 17, 2)
    confs:     (T, 17)
    scores:    (T,) or None  检测框 score

    score 与 confs 同样按邻居插值（而不是留 0）：漏检帧的几何量已被插值伪造，
    若 score 在此处骤降为 0，会引入一串"没人却有高分、突然掉到 0"的人造尖峰，
    模型会去拟合这个采样瑕疵。真正的"该帧是漏检"由 confs 已下压的数值表达。

    Returns: 未传 scores 时 (kp_out, cf_out)，传入时 (kp_out, cf_out, sc_out)。
    """
    T = keypoints.shape[0]
    valid_mask = keypoints.reshape(T, -1).sum(axis=1) > 1.0

    kp_out = _fill_neighbors(keypoints, valid_mask, tail_scale=1.0)
    cf_out = _fill_neighbors(confs, valid_mask, tail_scale=0.3)
    if scores is None:
        return kp_out, cf_out
    return kp_out, cf_out, _fill_neighbors(scores, valid_mask, tail_scale=0.3)
