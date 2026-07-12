import numpy as np


class LabelGeneratorV3:
    """
    生成：
        risk_score
        time_to_fall
    """

    def __init__(self, fps=10, max_time=3.0):
        self.fps = fps
        self.max_time = max_time

    # ==================================================
    # 核心函数
    # ==================================================
    def generate(self, seq_len, fall_frame=None):
        """
        Parameters
        ----------
        seq_len : int
            序列长度

        fall_frame : int or None
            跌倒发生帧
            None → SAFE样本
        """

        risk = np.zeros(seq_len, dtype=np.float32)
        time_to_fall = np.full(seq_len, -1, dtype=np.float32)

        # ==========================================
        # SAFE样本
        # ==========================================
        if fall_frame is None:
            return risk, time_to_fall

        # ==========================================
        # FALL样本
        # ==========================================
        for t in range(seq_len):

            delta = fall_frame - t
            time_sec = delta / self.fps

            time_to_fall[t] = max(time_sec, 0.0)

            # risk函数（工业推荐）
            if time_sec <= 0:
                risk[t] = 1.0
            else:
                risk[t] = np.clip(
                    1.0 - time_sec / self.max_time,
                    0.0,
                    1.0
                )

        return risk, time_to_fall