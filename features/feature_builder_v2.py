import numpy as np

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
    def build(kpts, confs=None):
        """
        Parameters
        ----------
        kpts : ndarray (17, 2)  COCO 关键点 x,y
        confs : ndarray (17,) or None  关键点置信度

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

        return np.concatenate([base, geo_original, geo_new]).astype(np.float32)


def interpolate_missing_frames(keypoints, confs):
    """
    修复 YOLO 漏检导致的零填充帧。

    keypoints: (T, 17, 2)
    confs:     (T, 17)

    Returns: (kp_out, cf_out) 修复后的数据。
    """
    T = keypoints.shape[0]
    kp_out = keypoints.copy()
    cf_out = confs.copy()
    valid_mask = keypoints.reshape(T, -1).sum(axis=1) > 1.0

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
            kp_out[t] = kp_out[prev_t] * (1 - alpha) + kp_out[next_t] * alpha
            cf_out[t] = cf_out[prev_t] * (1 - alpha) + cf_out[next_t] * alpha
        elif prev_t >= 0:
            kp_out[t] = kp_out[prev_t]
            cf_out[t] = cf_out[prev_t] * 0.3
        elif next_t < T:
            kp_out[t] = kp_out[next_t]
            cf_out[t] = cf_out[next_t] * 0.3

    return kp_out, cf_out
