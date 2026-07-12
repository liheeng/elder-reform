from collections import deque
import numpy as np


class SequenceBufferV3:
    """
    工业级跌倒趋势缓存器（v3）

    输出：
        raw: (T, 34)
        vel: (T, 34)
        acc: (T, 34)
    """

    def __init__(self, seq_len=30, feat_dim=34):
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.buffers = {}

    # ==================================================
    # 更新
    # ==================================================
    def update(self, person_id, feature):
        """
        feature: (feat_dim,) — 支持任意特征维度
        """

        feature = np.asarray(feature, dtype=np.float32)

        # ❗异常过滤（工业必须）
        if feature.ndim != 1:
            return
        if np.isnan(feature).any():
            return

        if person_id not in self.buffers:
            self.buffers[person_id] = deque(maxlen=self.seq_len)

        self.buffers[person_id].append(feature)

    # ==================================================
    # 是否准备好
    # ==================================================
    def is_ready(self, person_id):
        return (
            person_id in self.buffers and
            len(self.buffers[person_id]) == self.seq_len
        )

    # ==================================================
    # 原始序列
    # ==================================================
    def get_sequence(self, person_id):
        if not self.is_ready(person_id):
            return None

        return np.asarray(self.buffers[person_id], dtype=np.float32)

    # ==================================================
    # 速度（核心趋势信号）
    # ==================================================
    def get_velocity(self, person_id):
        seq = self.get_sequence(person_id)
        if seq is None:
            return None

        vel = np.zeros_like(seq)
        vel[1:] = seq[1:] - seq[:-1]

        return vel

    # ==================================================
    # 加速度（趋势加速变化）
    # ==================================================
    def get_acceleration(self, person_id):
        vel = self.get_velocity(person_id)
        if vel is None:
            return None

        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        return acc

    # ==================================================
    # 删除人
    # ==================================================
    def remove(self, person_id):
        if person_id in self.buffers:
            del self.buffers[person_id]

    def clear(self):
        self.buffers.clear()

    # ==================================================
    # 调试
    # ==================================================
    def __len__(self):
        return len(self.buffers)

    def __repr__(self):
        return f"SequenceBufferV3(persons={len(self.buffers)}, seq_len={self.seq_len})"