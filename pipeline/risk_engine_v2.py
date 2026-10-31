"""
risk_engine_v2.py

V2 RiskEngine with dual-head support:
  - Classification head: SAFE / FALLEN / SITTING / NON_PERSON
  - Regression head: continuous risk value [0, 1]
  - RiskEngine fusion: class-based safety gates + EMA smoothed risk
"""

from collections import deque


class RiskEngineV2:

    CLASS_SAFE = 0
    CLASS_FALLEN = 1
    CLASS_SITTING = 2
    CLASS_NON_PERSON = 3

    def __init__(self, smooth_alpha=0.4, warning_thres=0.40,
                 trigger_frames=10, lock_frames=15):
        # warning_thres 0.35 -> 0.40 (2026-10): conf 降到 0.02 后，边缘处低分
        # 静态框(score 0.02-0.06, 关键点均值 0.39-0.47)的风险值稳定落在
        # 0.37-0.40，恰好在 0.35 之上，使正常视频出现持续的 WARNING。
        # 详见 docs/tuning-log.md
        self.alpha = smooth_alpha
        self.warning_thres = warning_thres
        self.trigger_frames = trigger_frames
        self.lock_frames = lock_frames

        self.smooth_score = {}
        self.counter = {}
        self.danger_confirmed = {}
        self.lock_count = {}

    def _init_if_needed(self, person_id):
        if person_id not in self.smooth_score:
            self.smooth_score[person_id] = 0.0
            self.counter[person_id] = 0
            self.danger_confirmed[person_id] = False
            self.lock_count[person_id] = 0

    def remove(self, person_id):
        """清除某人的状态（跟踪器判定该目标已消失时调用）。"""
        self.smooth_score.pop(person_id, None)
        self.counter.pop(person_id, None)
        self.danger_confirmed.pop(person_id, None)
        self.lock_count.pop(person_id, None)

    def _compute_danger_prob(self, class_id, model_risk):
        """融合分类结果和回归风险值。"""
        # 安全关卡：坐姿和非人 → 强制零风险
        if class_id in (self.CLASS_SITTING, self.CLASS_NON_PERSON):
            return 0.0

        # 致命关卡：已倒地 → 至少 0.7
        if class_id == self.CLASS_FALLEN:
            return max(float(model_risk), 0.7)

        # 正常状态：回归头直达
        if class_id == self.CLASS_SAFE:
            return max(0.0, min(1.0, float(model_risk)))

        return 0.0

    def update(self, person_id, class_id, model_risk, **kwargs):
        self._init_if_needed(person_id)

        danger_prob = self._compute_danger_prob(class_id, model_risk)
        smooth = self._smooth_risk(person_id, danger_prob)
        self._count_high_risk_streak(person_id, smooth)

        state = self._resolve_state(person_id, class_id, danger_prob, smooth)
        state = self._hold_warning_before_safe(person_id, state)

        return {
            "risk_score": float(smooth),
            "state": state,
            "danger_confirmed": self._is_danger_latched(person_id),
        }

    def _smooth_risk(self, person_id, danger_prob):
        """EMA：本帧风险向历史平滑值靠拢。"""
        prev = self.smooth_score[person_id]
        smooth = self.alpha * danger_prob + (1.0 - self.alpha) * prev
        self.smooth_score[person_id] = smooth
        return smooth

    def _count_high_risk_streak(self, person_id, smooth):
        """累计连续超阈值帧数，中断即清零。"""
        if smooth >= self.warning_thres:
            self.counter[person_id] += 1
        else:
            self.counter[person_id] = 0

    def _resolve_state(self, person_id, class_id, danger_prob, smooth):
        if self._is_danger_latched(person_id):
            return self._on_danger_state_while_latched(
                person_id, class_id, danger_prob
            )
        return self._on_normal_state(person_id, smooth)

    def _on_danger_state_while_latched(self, person_id, class_id, danger_prob):
        """前置条件：此人处于 DANGER 锁存状态。

        只认连续 lock_frames 帧的稳定恢复，其余一律维持 DANGER。
        """
        if not self._is_recovery_frame(class_id, danger_prob):
            self.lock_count[person_id] = 0
            return "DANGER"

        self.lock_count[person_id] += 1
        if self.lock_count[person_id] < self.lock_frames:
            return "DANGER"

        self._clear_danger_latch(person_id)
        return "WARNING"

    def _on_normal_state(self, person_id, smooth):
        """前置条件：此人未处于 DANGER 锁存状态。

        超阈值帧数够则升级 DANGER，否则按平滑风险给出 WARNING / SAFE。
        """
        if self.counter[person_id] >= self.trigger_frames:
            self._latch_danger(person_id)
            return "DANGER"
        if smooth >= self.warning_thres:
            self._start_warning_hold(person_id)
            return "WARNING"
        return "SAFE"

    def _is_recovery_frame(self, class_id, danger_prob):
        """本帧模型判为 SAFE，且瞬时风险已低于阈值。"""
        return class_id == self.CLASS_SAFE and danger_prob < self.warning_thres

    def _start_warning_hold(self, person_id):
        """发出 WARNING 后开启保护期，期间 SAFE 候选一律被压回 WARNING。"""
        self.lock_count[person_id] = self.lock_frames

    def _has_warning_hold_remaining(self, person_id):
        return self.lock_count.get(person_id, 0) > 0

    def _consume_warning_hold_frame(self, person_id):
        self.lock_count[person_id] -= 1

    def _hold_warning_before_safe(self, person_id, state):
        """保护期未满时，不给 WARNING 直接退回 SAFE。"""
        if state != "SAFE" or not self._has_warning_hold_remaining(person_id):
            return state
        self._consume_warning_hold_frame(person_id)
        return "WARNING"

    def _is_danger_latched(self, person_id):
        return self.danger_confirmed.get(person_id, False)

    def _latch_danger(self, person_id):
        self.danger_confirmed[person_id] = True

    def _clear_danger_latch(self, person_id):
        self.danger_confirmed[person_id] = False
        self.lock_count[person_id] = 0
        self.counter[person_id] = 0
