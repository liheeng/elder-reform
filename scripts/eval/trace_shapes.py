#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trace_shapes.py

形状与参数量追踪工具。把下面这条链路里每一环的**实际张量形状**打印出来:

    单帧 64 维  →  SequenceBufferV3 30 帧  →  (30, 64)
               →  build_192_feature 拼接   →  (30, 192)
               →  2 层 LSTM               →  (1, 30, 128)
               →  双池化                   →  (1, 128)
               →  三个并行预测头           →  (1,4) / (1,) / (1,)

设计约束:
    * **不需要任何权重文件** —— models/ 为空也能运行（模型只做前向，不加载 state_dict）
    * **不需要 ultralytics / opencv** —— 模型侧只依赖 torch，特征侧只依赖 numpy
    * torch 不可用时自动降级为"仅特征侧"模式

用法:
    python scripts/eval/trace_shapes.py
    python scripts/eval/trace_shapes.py --samples 2000
    python scripts/eval/trace_shapes.py --features-only
"""

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features.feature_builder_v2 import FeatureBuilderV2  # noqa: E402
from features.normalize import normalize_keypoints  # noqa: E402
from sequence.sequence_buffer import SequenceBufferV3  # noqa: E402

# ---------------------------------------------------------------- 常量

SEQ_LEN = 30
FEAT_DIM = 64
INPUT_DIM = 192
HIDDEN_DIM = 128
NUM_CLASSES = 4

KEYPOINT_NAMES = [
    "nose", "leye", "reye", "lear", "rear",
    "lsho", "rsho", "lelb", "relb", "lwri", "rwri",
    "lhip", "rhip", "lkne", "rkne", "lank", "rank",
]

# FeatureBuilderV2.build 的 64 维语义布局
GEO_ORIGINAL_NAMES = [
    "cg_y", "torso_angle", "left_knee_angle",
    "right_knee_angle", "foot_distance", "body_height",
]
GEO_ENHANCED_NAMES = [
    "cg_y_norm", "foot_norm", "com_x", "com_y",
    "left_hip_angle", "right_hip_angle", "torso_angle",
]


def dim_names():
    """返回 64 个维度的可读名称。

    51 / 57 原本是几何量 cg_y / cg_y_norm（归一化后恒为 0），现原地复用为
    检测框 score 与其有效性掩码，见 FeatureBuilderV2.SCORE_DIM。
    """
    names = []
    for kp in KEYPOINT_NAMES:
        names += [f"kpt.{kp}.x", f"kpt.{kp}.y", f"kpt.{kp}.conf"]
    names += [f"geo.{n}" for n in GEO_ORIGINAL_NAMES]
    names += [f"geo2.{n}" for n in GEO_ENHANCED_NAMES]
    names[FeatureBuilderV2.SCORE_DIM] = "bbox.score"
    names[FeatureBuilderV2.HAS_SCORE_DIM] = "bbox.has_score"
    return names


DIM_NAMES = dim_names()

# 期望被 hook 的模块: (模块名, 说明)
HOOK_TARGETS = [
    ("lstm", "2 层 LSTM  → 每时间步 128 维"),
    ("class_proj.0", "class 头中间层 Linear 128→64"),
    ("class_proj.2", "class 头 LayerNorm 64"),
    ("classifier", "class 头输出 Linear 64→4"),
    ("risk_proj.0", "risk 头中间层 Linear 128→64"),
    ("risk_proj.2", "risk 头 LayerNorm 64"),
    ("risk_head.0", "risk 头输出 Linear 64→1"),
    ("risk_head.1", "risk 头 Sigmoid 挤压到 [0,1]"),
    ("time_proj.0", "time 头中间层 Linear 128→64"),
    ("time_proj.2", "time 头 LayerNorm 64"),
    ("time_head.0", "time 头输出 Linear 64→1"),
]

PARAM_GROUPS = [
    "lstm", "class_proj", "classifier",
    "risk_proj", "risk_head",
    "time_proj", "time_head",
]

# ---------------------------------------------------------------- 打印工具

WIDTH = 78


def title(text):
    print()
    print("=" * WIDTH)
    print(text)
    print("=" * WIDTH)


def row(label, shape, note=""):
    shape_s = str(tuple(shape)) if shape is not None else "-"
    print(f"  {label:<28} {shape_s:<18} {note}")


# ---------------------------------------------------------------- 合成骨架

# 以髋中心为原点、y 轴向下的标准站姿骨架（单位: 像素）
CANON_SKELETON = np.array([
    [0, -60],      # 0  nose
    [4, -64],      # 1  leye
    [-4, -64],     # 2  reye
    [8, -62],      # 3  lear
    [-8, -62],     # 4  rear
    [12, -45],     # 5  lsho
    [-12, -45],    # 6  rsho
    [18, -20],     # 7  lelb
    [-18, -20],    # 8  relb
    [20, 3],       # 9  lwri
    [-20, 3],      # 10 rwri
    [8, 0],        # 11 lhip
    [-8, 0],       # 12 rhip
    [9, 50],       # 13 lkne
    [-9, 50],      # 14 rkne
    [10, 100],     # 15 lank
    [-10, 100],    # 16 rank
], dtype=np.float32)


def make_skeleton(rng, jitter=3.0):
    """生成一个带随机扰动、随机缩放与随机平移的骨架。"""
    kp = CANON_SKELETON + rng.normal(0.0, jitter, CANON_SKELETON.shape)
    kp *= float(rng.uniform(0.5, 2.0))            # 随机尺度
    kp += rng.uniform(0.0, 640.0, (1, 2))         # 随机平移
    return kp.astype(np.float32)


def make_confs(rng):
    """生成仿真的 YOLO 关键点置信度.

    必须让置信度变化: 若一律传 1.0, 则 17 个 conf 维度会退化为常量,
    在冗余分析里产生**假阳性** (那是造数据的假象, 不是特征本身的性质).
    这里额外模拟若干被遮挡的低置信度关键点.
    """
    confs = rng.uniform(0.3, 1.0, 17)
    n_occluded = int(rng.integers(0, 5))
    if n_occluded:
        idx = rng.choice(17, size=n_occluded, replace=False)
        confs[idx] = rng.uniform(0.05, 0.3, n_occluded)
    return confs.astype(np.float32)


def single_feature(kp, confs):
    """走真实特征链: normalize_keypoints → FeatureBuilderV2.build"""
    nk, nc = normalize_keypoints(kp, confs)
    if nk is None:
        return None
    return FeatureBuilderV2.build(nk, nc)


# ---------------------------------------------------------------- 192 拼接

def _build_192_local(buffer, person_id):
    """build_192_feature 的本地等价实现（导入 realtime_detect 失败时兜底）。"""
    raw = buffer.get_sequence(person_id)
    if raw is None:
        return None
    if len(raw) < 2:
        vel = np.zeros_like(raw)
        acc = np.zeros_like(raw)
    else:
        vel = np.zeros_like(raw)
        vel[1:] = raw[1:] - raw[:-1]
        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([raw, vel, acc], axis=1).astype(np.float32)


def load_build_192():
    """优先复用 realtime_detect 的真实实现，失败则降级。"""
    try:
        from realtime_detect import build_192_feature
        return build_192_feature, "复用 realtime_detect.build_192_feature"
    except Exception as exc:
        return _build_192_local, f"本地等价实现（导入 realtime_detect 失败: {type(exc).__name__}）"


# ---------------------------------------------------------------- 各段分析

def section_feature():
    title("[1/5] 特征侧 —— 单帧 64 维的构成")

    rng = np.random.default_rng(0)
    feat = single_feature(make_skeleton(rng), make_confs(rng))
    if feat is None:
        print("  骨架构造失败（body_height < 5），无法继续")
        return None, None

    row("单帧特征", feat.shape, "FeatureBuilderV2.build 输出")

    print()
    print(f"  {'区块':<14} {'索引':<12} {'维度':<6} 内容")
    print(f"  {'-' * 70}")
    print(f"  {'关键点':<14} {'0..50':<12} {'51':<6} 17 × (x, y, conf) —— 交错布局")
    print(f"  {'原始几何':<14} {'51..56':<12} {'6':<6} " + ", ".join(GEO_ORIGINAL_NAMES))
    print(f"  {'增强几何':<14} {'57..63':<12} {'7':<6} " + ", ".join(GEO_ENHANCED_NAMES))

    return feat, rng


def section_sequence(feat):
    title("[2/5] 序列侧 —— 30 帧窗口与 192 维拼接")

    buf = SequenceBufferV3(seq_len=SEQ_LEN, feat_dim=FEAT_DIM)
    for _ in range(SEQ_LEN):
        buf.update(0, feat)

    raw = buf.get_sequence(0)
    if raw is None:
        print("  缓冲区未就绪，无法继续")
        return None

    build_192, src = load_build_192()
    seq192 = build_192(buf, 0)
    if seq192 is None:
        print("  拼接失败（缓冲区未就绪），无法继续")
        return None

    print(f"  拼接实现: {src}")
    print()
    row("buffer 内部", raw.shape, "SequenceBufferV3.get_sequence")
    row("模型输入", seq192.shape, "build_192_feature 输出")
    print()
    row("  ├─ 位置 pos", raw.shape, "索引 0..63")
    row("  ├─ 速度 vel", raw.shape, "索引 64..127")
    row("  └─ 加速度 acc", raw.shape, "索引 128..191")

    # 交叉验证: buffer 自带的 vel/acc 与 build_192 的拼接是否一致
    # （两者签名上都可能返回 None，故显式收窄）
    vel_buf = buf.get_velocity(0)
    acc_buf = buf.get_acceleration(0)
    if vel_buf is not None and acc_buf is not None:
        same_vel = bool(np.allclose(vel_buf, seq192[:, FEAT_DIM:FEAT_DIM * 2]))
        same_acc = bool(np.allclose(acc_buf, seq192[:, FEAT_DIM * 2:]))
    else:
        same_vel = same_acc = False
    print()
    print(f"  一致性检查: buffer.get_velocity == seq192[:, 64:128]  → {same_vel}")
    print(f"              buffer.get_acceleration == seq192[:, 128:] → {same_acc}")
    print("              注: 两处实现完全等价, 属重复代码 (见 architecture.md §6.4)")

    return seq192


def section_model(seq192):
    title("[3/5] 模型侧 —— LSTM 与三个头部的逐层形状")

    try:
        import torch
    except Exception as exc:
        print(f"  torch 不可用（{type(exc).__name__}: {exc}），跳过模型侧追踪")
        return None

    from classifier.risk_classifier_v2 import RiskClassifierV2

    model = RiskClassifierV2()
    model.eval()

    x = torch.tensor(seq192, dtype=torch.float32).unsqueeze(0)
    print(f"  输入: seq192 {tuple(seq192.shape)} → unsqueeze(0) → {tuple(x.shape)}")
    print("  （predict 内部做的 batch 维扩展）")
    print()

    captured = {}
    mods = dict(model.named_modules())
    handles = []
    for name, _desc in HOOK_TARGETS:
        mod = mods.get(name)
        if mod is None:
            continue

        def make_hook(n):
            def hook(_module, _inputs, output):
                captured[n] = output
            return hook

        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        class_logits, risk, time_pred = model(x)

    for h in handles:
        h.remove()

    print(f"  {'层 / 输出':<32} {'形状':<16} 说明")
    print(f"  {'-' * 74}")
    row("输入", tuple(x.shape), "batch × 30 帧 × 192 维")

    lstm_out = captured.get("lstm")
    if lstm_out is not None:
        out = lstm_out[0]
        row("LSTM 输出", tuple(out.shape), "每时间步 128 维")
        pooled = 0.5 * out[:, -1, :] + 0.5 * out.mean(dim=1)
        row("双池化 pooled", tuple(pooled.shape), "非模块输出，由 LSTM 输出计算")

    for name, desc in HOOK_TARGETS:
        if name == "lstm" or name not in captured:
            continue
        val = captured[name]
        row(name, tuple(val.shape), desc)

    print()
    print("  ← 以上三个头**并行**消费同一个 pooled 向量，输出不拼接")
    print()
    row("返回 class_logits", tuple(class_logits.shape), "4 类 logits")
    row("返回 risk", tuple(risk.shape), "已 squeeze(-1)")
    row("返回 time_pred", tuple(time_pred.shape), "已 squeeze(-1)")

    return model


def section_params(model):
    title("[4/5] 参数量分布")

    if model is None:
        print("  torch 不可用或模型未构建，跳过")
        return

    groups = {g: 0 for g in PARAM_GROUPS}
    groups["其他"] = 0
    for pname, p in model.named_parameters():
        top = pname.split(".")[0]
        groups[top if top in groups else "其他"] += p.numel()

    total = sum(groups.values())
    print(f"  {'组件':<24} {'参数量':>12} {'占比':>9}")
    print(f"  {'-' * 48}")
    for name in PARAM_GROUPS + ["其他"]:
        n = groups[name]
        if n == 0:
            continue
        print(f"  {name:<24} {n:>12,} {n / total * 100:>8.2f}%")
    print(f"  {'-' * 48}")
    print(f"  {'合计':<24} {total:>12,} {100.0:>8.2f}%")
    print()
    print("  注: 参数绝大部分在 LSTM 上，三个头合计占比很低。")
    print("      想显著改变容量需调整 hidden_dim / num_layers，改头宽度收益很小。")


def section_redundancy(samples, seed, atol=1e-3, corr_tol=1e-6):
    title("[5/5] 相邻冗余维度实测 —— 验证 architecture.md §8.1")

    rng = np.random.default_rng(seed)
    feats = []
    for _ in range(samples):
        f = single_feature(make_skeleton(rng), make_confs(rng))
        if f is not None:
            feats.append(f)

    if len(feats) < 10:
        print("  有效样本不足，跳过")
        return

    X: np.ndarray = np.asarray(feats, dtype=np.float64)
    print(f"  样本数: {len(feats)}   特征维度: {X.shape[1]}")
    print(f"  常量判据: 绝对标准差 < {atol:g}（该特征空间下 < 1e-3 的波动无实际信息量）")
    print()

    std = X.std(axis=0)

    const_mask = std < atol
    const_idx = [i for i in range(FEAT_DIM) if const_mask[i]]

    print(f"  ── 常量维度: {len(const_idx)} 个")
    if const_idx:
        for i in const_idx:
            print(f"     [{i:>2}] {DIM_NAMES[i]:<24} 均值={X[:, i].mean():>10.4f}  std={std[i]:.3e}")
    else:
        print("     （无）")
    print()

    keep_idx = [i for i in range(FEAT_DIM) if not const_mask[i]]
    Xk = X[:, keep_idx]
    C: np.ndarray = np.corrcoef(Xk, rowvar=False)

    pairs = []
    for a in range(len(keep_idx)):
        for b in range(a + 1, len(keep_idx)):
            r = C[a, b]
            if np.isfinite(r) and abs(r) > 1.0 - corr_tol:
                pairs.append((keep_idx[a], keep_idx[b], r))

    print(f"  ── 完全相关维度对（|r| > {1.0 - corr_tol:g}）: {len(pairs)} 对")
    if pairs:
        for i, j, r in pairs:
            print(f"     [{i:>2}] {DIM_NAMES[i]:<24} ↔ [{j:>2}] {DIM_NAMES[j]:<24}  r={r:.10f}")
    else:
        print("     （无）")
    print()

    removable = len(const_idx) + len(pairs)
    print(f"  ── 结论: 可剔除维度 {removable} 个 →  {FEAT_DIM} - {removable} = {FEAT_DIM - removable}")
    print()
    print("  成因分三类:")
    print("    (a) 归一化锁定常量 —— hip_center 被平移到原点、体高缩放到 100,")
    print("        使 geo.cg_y 与 geo2.cg_y_norm 恒为 0、geo.body_height 恒为 100")
    print("    (b) 重复计算 —— geo.torso_angle 与 geo2.torso_angle 是同一个表达式;")
    print("        geo2.foot_norm 只是 geo.foot_distance 除以常量 100")
    print("    (c) 髋中心的结构性镜像 —— 归一化后 lhip 与 rhip 关于原点对称,")
    print("        即 kpt.rhip = -kpt.lhip 恒成立, 与输入无关")
    print()
    print("  副产物: 绝对尺度信息在 64 维中完全丢失 —— 远景与近景的人无法区分,")
    print("          同时也无法利用『髋部离地面多高』这一直接的跌倒线索。")


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(description="追踪跌倒检测链路的张量形状与参数量")
    parser.add_argument("--samples", type=int, default=800,
                        help="冗余维度实测的样本数（默认 800）")
    parser.add_argument("--seed", type=int, default=20260926,
                        help="冗余维度实测的随机种子（默认 20260926）")
    parser.add_argument("--features-only", action="store_true",
                        help="只跑特征侧与序列侧，跳过 torch 相关部分")
    args = parser.parse_args()

    print()
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.version.split()[0]}   numpy: {np.__version__}")

    feat, _rng = section_feature()
    if feat is None:
        return 1

    seq192 = section_sequence(feat)
    if seq192 is None:
        return 1

    model = None
    if not args.features_only:
        model = section_model(seq192)
        section_params(model)

    section_redundancy(args.samples, args.seed)

    title("形状流转总览")
    print("  单帧 64 维")
    print("      ↓  SequenceBufferV3 deque(maxlen=30)")
    print("  (30, 64)           ← 缓冲区中间态，不是模型输入")
    print("      ↓  build_192_feature: 沿 axis=1 拼接 pos | vel | acc")
    print("  (30, 192)          ← 真正的模型输入")
    print("      ↓  unsqueeze(0)")
    print("  (1, 30, 192)")
    print("      ↓  2 层 LSTM(hidden=128)")
    print("  (1, 30, 128)")
    print("      ↓  双池化 0.5×末帧 + 0.5×均值")
    print("  (1, 128)           ← 三个头共用的表示")
    print("      ├→ class 头  128→64→4   →  (1, 4)")
    print("      ├→ risk  头  128→64→1   →  (1,)   Sigmoid")
    print("      └→ time  头  128→64→1   →  (1,)   无激活")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
