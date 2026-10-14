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

        # EMA 平滑
        prev = self.smooth_score[person_id]
        smooth = self.alpha * danger_prob + (1.0 - self.alpha) * prev
        self.smooth_score[person_id] = smooth

        # 连续计数
        if smooth >= self.warning_thres:
            self.counter[person_id] += 1
        else:
            self.counter[person_id] = 0

        pid = person_id

        # 状态机
        if self.danger_confirmed.get(pid, False):
            if class_id == self.CLASS_SAFE and danger_prob < self.warning_thres:
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
            self.danger_confirmed[pid] = True
            state = "DANGER"

        elif smooth >= self.warning_thres:
            state = "WARNING"
            self.lock_count[pid] = self.lock_frames

        else:
            state = "SAFE"

        # 锁定保护
        if state == "SAFE" and self.lock_count.get(pid, 0) > 0:
            state = "WARNING"
            self.lock_count[pid] -= 1

        return {
            "risk_score": float(smooth),
            "state": state,
            "danger_confirmed": self.danger_confirmed.get(pid, False),
        }
