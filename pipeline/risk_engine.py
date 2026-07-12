"""
risk_engine.py  [REFORM 版]

从 495 行精简到 ~80 行。

移除内容:
  - fix5: posture_confirm_frames, torso_angle, hip_ground_confirm, _compute_posture
  - fix6: static_recovery_frames, static_count, _has_recent_motion
  - fix7: sit_recovery_frames, sit_hip_ground, sit_count
  - fix8: WARNING/PENDING → SAFE 坐姿/家具抑制
  - fix9: real_fall, risk_trail, init_risk_low, _has_significant_displacement
  - PENDING 状态: 由姿态门引入, 分类模型直接输出时不需要

保留内容:
  - EMA 平滑 (alpha=0.4)
  - 连续计数 (trigger_frames=10)
  - 状态机 SAFE → WARNING → DANGER
  - 锁定保护 (lock_frames=15)
  - danger_confirmed 标志

核心变化:
  - 模型直接输出 class → 映射为 state
  - NON_PERSON class → 直接忽略 (SAFE)
  - 无需任何姿态/运动后处理

兼容性:
  - 接收 class_probs/class_id (新模型) 或 danger_prob (旧模型)
  - 所有额外参数 (**kwargs) 被忽略, 不报错
"""

from collections import deque
import numpy as np


class RiskEngine:

    CLASS_SAFE = 0
    CLASS_WARNING = 1
    CLASS_FALLEN = 2
    CLASS_NON_PERSON = 3
    CLASS_SITTING = 4

    def __init__(
        self,
        smooth_alpha=0.4,
        warning_thres=0.35,
        trigger_frames=10,
        lock_frames=15,
    ):
        self.alpha = smooth_alpha
        self.warning_thres = warning_thres
        self.trigger_frames = trigger_frames
        self.lock_frames = lock_frames

        self.smooth_score = {}
        self.counter = {}
        self.danger_confirmed = {}
        self.lock_count = {}
        self.probs_history = {}

    def _init_if_needed(self, person_id):
        if person_id not in self.smooth_score:
            self.smooth_score[person_id] = 0.0
            self.counter[person_id] = 0
            self.danger_confirmed[person_id] = False
            self.lock_count[person_id] = 0
            self.probs_history[person_id] = deque(maxlen=30)

    def _compute_danger_prob(self, class_probs, class_id):
        """从 class 概率计算危险度 (0~1)。
        
        规则:
          - SITTING (4):   危险度 = 0.0 (坐姿完全安全)
          - NON_PERSON (3): 危险度 = 0.0 (忽略)
          - SAFE (0):      危险度 = 0.0
          - WARNING (1):   危险度 = 0.5~1.0
          - FALLEN (2):    危险度 = 1.0
        """
        if class_probs is not None:
            p_safe = class_probs.get("SAFE", 0.0)
            p_warn = class_probs.get("WARNING", 0.0)
            p_fallen = class_probs.get("FALLEN", 0.0)
            p_np = class_probs.get("NON_PERSON", 0.0)
            p_sit = class_probs.get("SITTING", 0.0)

            # SITTING → 完全安全
            if class_id == self.CLASS_SITTING or p_sit > 0.5:
                return 0.0
            # NON_PERSON → 忽略
            if class_id == self.CLASS_NON_PERSON or p_np > 0.5:
                return 0.0
            # FALLEN → 最高危险
            if class_id == self.CLASS_FALLEN or p_fallen > 0.5:
                return 1.0
            # WARNING → 中等 (p_warn / (p_safe + p_sit + p_warn) 归一化)
            if class_id == self.CLASS_WARNING or p_warn > 0.3:
                denom = max(p_safe + p_sit + p_warn, 1e-8)
                base = 0.5 + 0.5 * min(p_warn / denom, 1.0)
                # P4a: 低置信度 WARNING (p_warn < 0.5) → 大幅降权
                if p_warn < 0.5:
                    base *= 0.3
                return base
            # SAFE → 安全
            return 0.0
        return 0.0

    def update(self, person_id, class_probs=None, class_id=None, danger_prob=None, **kwargs):
        """
        Parameters
        ----------
        person_id : int
        class_probs : dict or None    (新模型) {"SAFE":..., "WARNING":..., ...}
        class_id : int or None        (新模型) 0-3
        danger_prob : float or None   (旧模型) risk 值 0~1
        **kwargs : ignored            向后兼容 (keypoints, frame_h, bbox, detect_rate 等)
        """
        self._init_if_needed(person_id)

        # P4b: 隔离 WARNING 过滤 — 前两帧都是 SAFE 时，单帧 WARNING 降级为 SAFE
        if class_id == self.CLASS_WARNING:
            hist = self.probs_history[person_id]
            if len(hist) >= 2:
                prev1 = hist[-1] if len(hist) >= 1 else {}
                prev2 = hist[-2] if len(hist) >= 2 else {}
                prev1_max = max(prev1, key=prev1.get) if prev1 else "SAFE"
                prev2_max = max(prev2, key=prev2.get) if prev2 else "SAFE"
                if prev1_max == "SAFE" and prev2_max == "SAFE":
                    class_id = self.CLASS_SAFE
                    if class_probs is not None:
                        class_probs = dict(class_probs)
                        class_probs["WARNING"] = min(class_probs.get("WARNING", 0.0), 0.2)

        # ---- 1. 确定危险度 ----
        if class_probs is not None and class_id is not None:
            danger_prob = self._compute_danger_prob(class_probs, class_id)
        elif danger_prob is None:
            danger_prob = 0.0
        danger_prob = float(danger_prob)

        # ---- 2. EMA 平滑 ----
        prev = self.smooth_score[person_id]
        smooth = self.alpha * danger_prob + (1.0 - self.alpha) * prev
        self.smooth_score[person_id] = smooth

        # 记录概率历史
        if class_probs is not None:
            self.probs_history[person_id].append(class_probs.copy())

        # ---- 3. 连续计数 ----
        if smooth >= self.warning_thres:
            self.counter[person_id] += 1
        else:
            self.counter[person_id] = 0

        # ---- 4. 状态机 ----
        pid = person_id
        if self.danger_confirmed.get(pid, False):
            # 已确认 DANGER → 只有模型连续输出 SAFE 才能解除
            if class_id == self.CLASS_SAFE:
                self.lock_count[pid] = self.lock_count.get(pid, 0) + 1
                if self.lock_count[pid] >= self.lock_frames:
                    self.danger_confirmed[pid] = False
                    self.lock_count[pid] = 0
                    self.counter[pid] = 0
                    state = "WARNING"
                else:
                    state = "DANGER"
            else:
                self.lock_count[pid] = 0
                state = "DANGER"

        elif self.counter[pid] >= self.trigger_frames:
            # 连续 WARNING 达到阈值 → DANGER
            self.danger_confirmed[pid] = True
            state = "DANGER"

        elif smooth >= self.warning_thres:
            # risk 超阈值但未达连续计数 → WARNING
            state = "WARNING"
            self.lock_count[pid] = self.lock_frames

        elif class_id == self.CLASS_NON_PERSON:
            # 非人目标 → 完全忽略
            state = "SAFE"

        elif class_id == self.CLASS_SITTING:
            # 坐姿 → 完全安全
            state = "SAFE"

        else:
            state = "SAFE"

        # ---- 5. 锁定保护 ----
        # WARNING 触发后 lock_frames 帧内不回退 SAFE
        if state == "SAFE" and self.lock_count.get(pid, 0) > 0:
            state = "WARNING"
            self.lock_count[pid] -= 1

        return {
            "risk_score": float(smooth),
            "state": state,
            "danger_confirmed": self.danger_confirmed.get(pid, False),
        }
