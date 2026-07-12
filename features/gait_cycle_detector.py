"""
步态周期检测器 (Gait Cycle Detector) — v3

基于脚踝关键点速度的步态事件检测，专门针对低分辨率视频(320×240)优化。

算法基础:
    Zeni JA, Richards JG, Higginson JS. "Two simple methods for determining
    gait events during treadmill and overground walking using kinematic data."
    Gait & Posture, 2008, 27(4):710-714.

v3 设计原则:
    - 使用欧氏距离速度(敏感, v1风格) + 交替联合后处理(准确, v2风格)
    - 左右脚联合检测: 确保 L-R-L-R 交替模式
    - 自适应阈值: 基于每段视频的速度分布自动计算
    - 专门适配320×240分辨率YOLO-Pose关键点的抖动特性

COCO关键点索引:
    15=左脚踝, 16=右脚踝, 11=左髋, 12=右髋
"""

import numpy as np
from typing import Dict, List, Tuple


class GaitCycleDetector:
    """
    步态周期检测器 v3: 灵敏度+精度的平衡。

    核心改进(vs v1/v2):
    - 候选检测: 阈值过零(不过度平滑，保持对低分视频的敏感度)
    - 后处理: 左右脚联合交替验证(去除同帧重复、确保步态节律)
    - 自适应: 针对每段视频自动调参
    """

    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(
        self,
        fps: float = 25.0,
        min_step_time: float = 0.30,
        max_step_time: float = 1.5,
        median_kernel: int = 5,
        vel_percentile_low: float = 12.0,
        n_std_threshold: float = 2.8,
    ):
        self.fps = fps
        self.min_step_frames = int(min_step_time * fps)
        self.max_step_frames = int(max_step_time * fps)
        self.median_kernel = median_kernel
        self.vel_percentile_low = vel_percentile_low
        self.n_std_threshold = n_std_threshold

    # ================================================================
    # 主入口
    # ================================================================

    def detect(self, keypoints_seq: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parameters
        ----------
        keypoints_seq : ndarray (T, 17, 2)

        Returns
        -------
        dict {'left': ndarray[frame_idx], 'right': ndarray[frame_idx]}
        """
        T = len(keypoints_seq)
        if T < self.min_step_frames * 2:
            return {'left': np.array([], dtype=int), 'right': np.array([], dtype=int)}

        # ── Step 1: 对每只脚独立检测候选 ──
        candidates = {}
        velocity_signals = {}  # 保存用于调试

        for foot_name, ankle_idx in [('left', self.LEFT_ANKLE), ('right', self.RIGHT_ANKLE)]:
            cands, vel_signal = self._detect_candidates_per_foot(keypoints_seq, ankle_idx)
            candidates[foot_name] = cands
            velocity_signals[foot_name] = vel_signal

        # ── Step 2: 联合后处理 ──
        heel_strikes = self._joint_postprocess(candidates, T)

        return heel_strikes

    # ================================================================
    # Step 1: 单脚候选检测
    # ================================================================

    def _detect_candidates_per_foot(
        self, keypoints_seq: np.ndarray, ankle_idx: int
    ) -> Tuple[List[int], dict]:
        """
        单脚候选事件检测。

        使用帧间欧氏距离作为速度(适合低分辨率),
        中值滤波 + 自适应阈值 + 过零检测。
        """
        T = len(keypoints_seq)

        # ---- 1a. 提取脚踝轨迹，计算帧间速度 ----
        ankle_traj = keypoints_seq[:, ankle_idx, :]  # (T, 2)
        disp = np.diff(ankle_traj, axis=0)            # (T-1, 2)
        velocity = np.linalg.norm(disp, axis=1)       # (T-1,)
        velocity = np.append(velocity, velocity[-1])   # (T,) 补齐

        # ---- 1b. 中值滤波 ----
        vel_filt = self._median_filter(velocity, self.median_kernel)

        # ---- 1c. 自适应阈值 ----
        threshold = self._compute_threshold(vel_filt)

        # ---- 1d. 过零检测 ----
        # 脚跟着地: 速度从 >threshold 降至 <=threshold
        raw_strikes = []
        for t in range(1, T):
            if vel_filt[t - 1] > threshold and vel_filt[t] <= threshold:
                raw_strikes.append(t)

        # ---- 1e. 合并过近事件 ----
        merged = self._merge_close(raw_strikes, velocity)

        vel_signal = {
            'raw': velocity,
            'filtered': vel_filt,
            'threshold': threshold,
        }
        return merged, vel_signal

    # ================================================================
    # Step 2: 联合后处理
    # ================================================================

    def _joint_postprocess(
        self, candidates: Dict[str, List[int]], total_frames: int
    ) -> Dict[str, np.ndarray]:
        """
        左右脚联合后处理。

        规则(按优先级):
        1. 禁止同帧检测: 同一帧左右都触发 → 保留速度更低的那只脚
        2. 禁止过密: 间隔 < min_step_frames*0.7 → 合并
        3. 强制交替: 连续两次同侧 → 插入对侧(从中点估计)
        4. 验证间隔: 间隔 > max_step_frames*2 → 可能属于不同行走段
        """
        left_raw = candidates.get('left', [])
        right_raw = candidates.get('right', [])

        # ── 构建统一时间线 ──
        all_events = []
        for t in left_raw:
            all_events.append({'frame': t, 'foot': 'left'})
        for t in right_raw:
            all_events.append({'frame': t, 'foot': 'right'})
        all_events.sort(key=lambda x: x['frame'])

        if len(all_events) < 2:
            return {
                'left': np.array(left_raw, dtype=int),
                'right': np.array(right_raw, dtype=int),
            }

        # ── Pass 1: 解析同帧冲突 ──
        resolved = []
        i = 0
        while i < len(all_events):
            # 收集同一帧的事件组
            group = [all_events[i]]
            j = i + 1
            while j < len(all_events) and all_events[j]['frame'] == group[0]['frame']:
                group.append(all_events[j])
                j += 1

            if len(group) == 1:
                resolved.append(group[0])
            else:
                # 同帧多事件 → 保留与上一个事件不同脚的那个
                if resolved and any(g['foot'] != resolved[-1]['foot'] for g in group):
                    for g in group:
                        if g['foot'] != resolved[-1]['foot']:
                            resolved.append(g)
                            break
                else:
                    # 无法判断 → 保留第一个
                    resolved.append(group[0])
            i = j

        # ── Pass 2: 去除过密事件 ──
        min_gap = max(3, int(self.min_step_frames * 0.6))
        dedup = [resolved[0]]
        for evt in resolved[1:]:
            gap = evt['frame'] - dedup[-1]['frame']
            if gap < min_gap:
                # 太近 → 替换上一个 iff 不同脚（有利于交替）
                if evt['foot'] != dedup[-1]['foot'] and gap >= min_gap // 2:
                    dedup[-1] = evt
                # 否则丢弃
            else:
                dedup.append(evt)

        # ── Pass 3: 强制交替模式 ──
        alternated = [dedup[0]]
        for evt in dedup[1:]:
            if evt['foot'] == alternated[-1]['foot']:
                # 同脚连续 → 这个位置应该是对侧脚事件
                # 尝试在中间插入对侧事件(取两帧中点)
                mid_frame = (alternated[-1]['frame'] + evt['frame']) // 2
                missing_foot = 'right' if evt['foot'] == 'left' else 'left'
                alternated.append({'frame': mid_frame, 'foot': missing_foot, 'interpolated': True})
                alternated.append(evt)
            else:
                alternated.append(evt)

        # ── Pass 4: 移除插入的虚拟事件中不合理的 ──
        validated = []
        for evt in alternated:
            if evt.get('interpolated') and validated:
                gap = evt['frame'] - validated[-1]['frame']
                if gap < min_gap:
                    continue  # 太近，丢弃插值
            validated.append(evt)

        # ── 分左右输出 ──
        left_out = sorted([e['frame'] for e in validated if e['foot'] == 'left'])
        right_out = sorted([e['frame'] for e in validated if e['foot'] == 'right'])

        return {
            'left': np.array(left_out, dtype=int),
            'right': np.array(right_out, dtype=int),
        }

    # ================================================================
    # 辅助方法
    # ================================================================

    def _median_filter(self, signal: np.ndarray, kernel: int) -> np.ndarray:
        """滑动窗口中值滤波"""
        if kernel % 2 == 0:
            kernel += 1
        n = len(signal)
        filtered = np.zeros_like(signal)
        half = kernel // 2
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            filtered[i] = np.median(signal[lo:hi])
        return filtered

    def _compute_threshold(self, velocity: np.ndarray) -> float:
        """
        自适应速度阈值。

        阈值 = 低百分位基线 + n_std × 低值区域标准差
        """
        low_baseline = np.percentile(velocity, self.vel_percentile_low)
        median_vel = np.median(velocity)
        low_samples = velocity[velocity < median_vel]
        if len(low_samples) < 5:
            low_samples = velocity
        noise_std = np.std(low_samples)
        threshold = low_baseline + self.n_std_threshold * noise_std
        threshold = max(threshold, np.mean(np.abs(velocity)) * 0.10)
        return threshold

    def _merge_close(self, strikes: List[int], velocity: np.ndarray) -> List[int]:
        """合并间隔过近的候选事件, 保留速度最低的那个"""
        if len(strikes) <= 1:
            return strikes
        merged = []
        i = 0
        while i < len(strikes):
            group = [strikes[i]]
            j = i + 1
            while j < len(strikes) and (strikes[j] - group[-1]) < self.min_step_frames:
                group.append(strikes[j])
                j += 1
            best = min(group, key=lambda f: velocity[f] if f < len(velocity) else float('inf'))
            merged.append(best)
            i = j
        return merged

    # ================================================================
    # 统计 & 验证接口
    # ================================================================

    def get_gait_cycles(self, heel_strikes: Dict[str, np.ndarray]) -> List[Dict]:
        """将着地事件组织为步态周期"""
        all_events = []
        for foot in ['left', 'right']:
            for frame in heel_strikes.get(foot, []):
                all_events.append({'frame': int(frame), 'foot': foot})
        all_events.sort(key=lambda x: x['frame'])

        cycles = []
        for i in range(len(all_events) - 1):
            cycles.append({
                'start_frame': all_events[i]['frame'],
                'end_frame': all_events[i + 1]['frame'],
                'start_foot': all_events[i]['foot'],
                'end_foot': all_events[i + 1]['foot'],
                'duration_frames': all_events[i + 1]['frame'] - all_events[i]['frame'],
                'duration_sec': (all_events[i + 1]['frame'] - all_events[i]['frame']) / self.fps,
            })
        return cycles

    def get_statistics(self, heel_strikes: Dict[str, np.ndarray]) -> Dict:
        """步态统计"""
        ln, rn = len(heel_strikes.get('left', [])), len(heel_strikes.get('right', []))
        stats = {'left_strikes': ln, 'right_strikes': rn,
                 'cadence_est': 0.0, 'mean_step_time': 0.0, 'l_r_ratio': 0.0}
        total = ln + rn
        if total >= 2:
            all_frames = sorted(list(heel_strikes.get('left', [])) +
                               list(heel_strikes.get('right', [])))
            intervals = np.diff(all_frames) / self.fps
            if len(intervals) > 0:
                stats['mean_step_time'] = float(np.mean(intervals))
                stats['cadence_est'] = float(60.0 / stats['mean_step_time'])
            if ln > 0 and rn > 0:
                stats['l_r_ratio'] = round(ln / rn, 2)
        return stats

    def validate(self, heel_strikes: Dict[str, np.ndarray]) -> Dict:
        """检测质量评估"""
        issues = []
        stats = self.get_statistics(heel_strikes)
        total = stats['left_strikes'] + stats['right_strikes']

        if stats['left_strikes'] > 0 and stats['right_strikes'] > 0:
            ratio = stats['l_r_ratio']
            if ratio > 1.5 or ratio < 0.67:
                issues.append(f"L/R imbalance: {ratio:.2f}")

        cadence = stats['cadence_est']
        if cadence > 0:
            if cadence < 60:
                issues.append(f"Cadence low ({cadence:.0f} spm) — may be under-detect")
            elif cadence > 180:
                issues.append(f"Cadence high ({cadence:.0f} spm) — may be over-detect")

        if total < 4:
            issues.append(f"Few events ({total}) — may be insufficient walking")

        quality = 'good' if len(issues) == 0 else ('acceptable' if len(issues) == 1 else 'poor')
        return {'quality': quality, 'issues': issues, 'stats': stats}


class GaitCycleVisualizer:
    """可视化工具"""

    @staticmethod
    def plot_detection(
        vel_left_raw, vel_right_raw,
        vel_left_filt, vel_right_filt,
        thr_left, thr_right,
        heel_strikes,
        title="Gait Detection",
        save_path=None,
    ):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        T = len(vel_left_raw)
        frames = np.arange(T)
        colors = {'left': '#E74C3C', 'right': '#3498DB'}

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(title, fontsize=13, fontweight='bold')

        for idx, (foot, raw, filt, thr) in enumerate([
            ('left', vel_left_raw, vel_left_filt, thr_left),
            ('right', vel_right_raw, vel_right_filt, thr_right),
        ]):
            ax = axes[idx]
            ax.plot(frames, raw, alpha=0.18, color='gray', linewidth=0.6, label='Raw')
            ax.plot(frames, filt, color=colors[foot], linewidth=2.0,
                    label=f'{foot.capitalize()} Foot')
            ax.axhline(y=thr, color='red', ls='--', lw=1.0, alpha=0.6)
            ax.text(T - 5, thr, f' thr={thr:.1f}', fontsize=8, va='bottom', ha='right', color='red')

            strikes = heel_strikes.get(foot, [])
            if len(strikes) > 0:
                sv = filt[strikes]
                ax.scatter(strikes, sv, c=colors[foot], marker='v', s=100,
                          zorder=10, edgecolors='black', linewidths=0.8,
                          label=f'HS x{len(strikes)}')
                for s in strikes:
                    ax.axvline(x=s, color=colors[foot], alpha=0.10, linewidth=1.5)

            ax.set_ylabel('Velocity (px/frame)')
            ax.legend(loc='upper right', fontsize=7.5, framealpha=0.75)
            ax.grid(True, alpha=0.25)

        axes[1].set_xlabel('Frame')
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
        return fig
