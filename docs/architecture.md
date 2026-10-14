# 系统架构与学习路径

> 本文档基于对当前代码的实际通读整理，**以代码为准**，与 `README.md` 不一致之处已在第五节列出。

---

## 目录

- [一、项目定位](#一项目定位)
- [二、架构图](#二架构图)
- [三、数据契约](#三数据契约)
- [四、模块速查](#四模块速查)
- [五、关键设计决策](#五关键设计决策)
- [六、仓库现状与偏差](#六仓库现状与偏差)
- [七、学习路径](#七学习路径)
- [八、待验证清单](#八待验证清单)
- [九、后续优化方向](#九后续优化方向)

---

## 一、项目定位

**姿态估计 + 时序深度学习**的老年人跌倒检测流水线。

```
摄像头帧 → YOLO26n-Pose 关键点 → 64 维生物力学特征 → 30 帧滑窗
         → 192 维时序张量 → 2 层 LSTM 多任务模型 → 风险状态机 → SAFE/WARNING/DANGER
```

**当前主线是 V2 版本**（4 类分类 + 风险回归 + 时间回归），由 `realtime_detect.py` 驱动。
`README.md` 描述的是 V4/5 类版本，部分内容与代码不符（见第六节）。

### 4 类定义

| 类 ID | 名称 | 含义 |
|---|---|---|
| 0 | SAFE | 正常行走 / 站立 |
| 1 | FALLEN | 已跌倒 / 躺地 |
| 2 | SITTING | 坐下 |
| 3 | NON_PERSON | 家具等非人误检 |

> **没有 WARNING 类**。WARNING 由回归头输出的连续风险值配合状态机表达，见第五节。
> 详细标签规则见 [`docs/classification-logic.md`](classification-logic.md)。

---

## 二、架构图

### 2.1 离线训练 + 在线推理

```mermaid
flowchart TD
    subgraph OFFLINE["离线训练流程 scripts/"]
        V["原始视频<br/>Le2i / Office / Pre-VFall / Home_02"]
        ANNO["逐帧标注<br/>fall_start / fall_end / status"]
        Y0["PoseExtractor 批量抽点"]
        KPC["yolo_keypoints/<br/>预提取关键点缓存"]
        BLD["build_dataset_v2.py<br/>滑窗切分 + 6 条标签规则"]
        NPY["processed/*.npy<br/>x_v2 / class_labels_v2<br/>risk_labels_v2 / time_labels_v2"]
        TRN["train_v2.py<br/>MultiTaskLoss + 3-fold LOSO"]
        W["models/lstm_v2.pt<br/>norm_v2.npz<br/>temperature_v2.npz"]
        V --> ANNO --> BLD
        V --> Y0 --> KPC --> BLD
        BLD --> NPY --> TRN --> W
    end

    subgraph ONLINE["在线推理流水线 realtime_detect.py"]
        CAM["视频源<br/>摄像头 / 视频文件"]
        PE["PoseExtractor<br/>YOLO26n-Pose + 质量门控<br/>+ 重叠检测合并"]
        TRK["PersonTracker<br/>质心跟踪分配 ID"]
        NRM["normalize_keypoints<br/>髋中心平移 + 身高缩放"]
        FB["FeatureBuilderV2<br/>64 维生物力学特征"]
        SB["SequenceBufferV3<br/>30 帧 deque"]
        CAT["build_192_feature<br/>位置 64 + 速度 64 + 加速度 64"]
        CLS["LSTMClassifierV2<br/>mean/std 归一化 + 温度缩放"]
        NET["RiskClassifierV2<br/>2 层 LSTM + 3 个预测头"]
        ENG["RiskEngineV2<br/>安全门 + EMA + 状态机"]
        OUT["可视化<br/>bbox + risk + 状态颜色"]
        CAM --> PE --> TRK --> NRM --> FB --> SB --> CAT --> CLS --> NET --> ENG --> OUT
    end

    W -.加载权重.-> CLS
    style TRK stroke-dasharray: 5 5
```

> `PersonTracker` 以虚线标记：它在 `tracker/` 中存在，但**并未被 `realtime_detect.py` 调用**
> （`person_id` 被硬编码为 `0`）。当前实现实为**单目标**，详见第六节。

### 2.2 状态机 `RiskEngineV2`

```mermaid
stateDiagram-v2
    [*] --> SAFE
    SAFE --> WARNING: 平滑风险 >= 0.35
    WARNING --> SAFE: 风险回落，但被锁定保护 15 帧
    WARNING --> DANGER: 连续 10 帧超阈值
    DANGER --> WARNING: 连续 15 帧 class=SAFE 且风险低
    DANGER --> DANGER: 其他情况保持锁存
```

三个计数器语义：

| 变量 | 作用 |
|---|---|
| `counter` | 连续超阈值帧数，达 `trigger_frames=10` 触发 DANGER |
| `danger_confirmed` | DANGER 锁存标志，防止状态抖动 |
| `lock_count` | 双重职责：DANGER 解锁倒计时 + WARNING 退出的锁定保护 |

### 2.3 多任务网络 `RiskClassifierV2`

```mermaid
flowchart LR
    IN["输入<br/>30 x 192"] --> LSTM["2 层 LSTM<br/>hidden=128<br/>dropout=0.3"]
    LSTM --> POOL["双池化<br/>0.5 x 末帧 + 0.5 x 均值"]
    POOL --> H1["class 头<br/>128-64-4<br/>无激活"]
    POOL --> H2["risk 头<br/>128-64-1<br/>Sigmoid"]
    POOL --> H3["time 头<br/>128-64-1<br/>无激活"]
    H1 --> L1["CrossEntropy"]
    H2 --> L2["MSELoss"]
    H3 --> L3["SmoothL1Loss<br/>仅有效样本"]
```

#### 头部形状流转（实测）

三个头**并行**消费同一个 `pooled` 向量，不存在 “3×64” 这样的拼接结构。
`64` 是每个头**内部的中间层宽度**（超参），`4 / 1 / 1` 才是各头的**输出**维度。

| 层 / 输出 | 形状 | 说明 |
|---|---|---|
| 输入 | `(1, 30, 192)` | batch × 30 帧 × 192 维 |
| LSTM 输出 | `(1, 30, 128)` | 每个时间步 128 维 |
| 双池化 `pooled` | `(1, 128)` | 非模块输出，由 LSTM 输出计算 |
| class 头中间层 | `(1, 64)` | `Linear(128, 64)` |
| class 头输出 | `(1, 4)` | `Linear(64, 4)`，无激活 → logits |
| risk 头中间层 | `(1, 64)` | `Linear(128, 64)` |
| risk 头输出 | `(1, 1)` → `(1,)` | `Linear(64, 1)` + `Sigmoid`，再 `squeeze(-1)` |
| time 头中间层 | `(1, 64)` | `Linear(128, 64)` |
| time 头输出 | `(1, 1)` → `(1,)` | `Linear(64, 1)`，无激活，再 `squeeze(-1)` |

> 名为 `64` 但含义不同的量共有三处，彼此无关：
> 单帧特征维度（51+6+7）、192 中每个区块的宽度、每个头的中间层宽度。

#### 参数量分布（实测）

| 组件 | 参数量 | 占比 |
|---|---|---|
| `lstm` | 296,960 | 92.08% |
| `class_proj` | 8,384 | 2.60% |
| `risk_proj` | 8,384 | 2.60% |
| `time_proj` | 8,384 | 2.60% |
| `classifier` | 260 | 0.08% |
| `risk_head` | 65 | 0.02% |
| `time_head` | 65 | 0.02% |
| **合计** | **322,502** | 100% |

参数绝大部分在 LSTM 上，三个头合计仅约 8%。想显著改变模型容量应调整
`hidden_dim` / `num_layers`，改头宽度收益很小。

> 以上数据由 `scripts/eval/trace_shapes.py` 实测输出（无需权重文件即可运行）。

---

## 三、数据契约

理解本项目的钥匙 —— 张量形状的流转链路：

| 阶段 | 形状 | 产出位置 |
|---|---|---|
| 视频帧 | `(H, W, 3)` BGR uint8 | `cv2.VideoCapture` |
| YOLO 检测 | `bbox(4,)` + `kpts(17,2)` + `conf(17,)` | `PoseExtractor.extract` |
| 关键点归一化 | `(17,2)` | `features/normalize.py` |
| 单帧特征 | `(64,)` | `FeatureBuilderV2.build` |
| 序列缓冲 | `(30, 64)` | `SequenceBufferV3` |
| 模型输入 | `(30, 192)` | `build_192_feature` |
| 模型输出 | `probs(4)` + `risk∈[0,1]` + `time_to_fall` | `LSTMClassifierV2.predict` |
| 决策输出 | `{risk_score, state, danger_confirmed}` | `RiskEngineV2.update` |

### 3.1 单帧 64 维的生成路径

全部发生在 `realtime_detect.py` 的这两行：

```python
nk, nc = normalize_keypoints(kpts, kpts_conf)   # Step 1 + 2
feature = FeatureBuilderV2.build(nk, nc)        # Step 3
```

**Step 1 — YOLO 输出**

| 变量 | 形状 | 含义 |
|---|---|---|
| `keypoints` | `(17, 2)` | 像素坐标 |
| `kpts_conf` | `(17,)` | 逐关键点置信度 |

COCO 顺序：`0 nose, 1 leye, 2 reye, 3 lear, 4 rear, 5 lsho, 6 rsho, 7 lelb,
8 relb, 9 lwri, 10 rwri, 11 lhip, 12 rhip, 13 lkne, 14 rkne, 15 lank, 16 rank`

**Step 2 — 归一化：平移 + 缩放**

```python
bh = |nose - (lank + rank)/2|          # 体高
if bh < 5.0: 归一化失败
hip_center = (lhip + rhip)/2
kpts_norm = (kpts - hip_center) * (100.0 / bh)
```

把**髀中心移到原点、体高缩放到 100 px**。置信度原样透传，不参与归一化。

**Step 3 — 组装 64 维**

| 区块 | 索引 | 维度 | 内容 |
|---|---|---|---|
| 关键点 | `0..50` | 51 | 第 k 个关键点的 `x, y, conf`，**交错布局**（`3k, 3k+1, 3k+2`） |
| 原始几何 | `51..56` | 6 | `cg_y` / `torso_angle` / `left_knee_angle` / `right_knee_angle` / `foot_distance` / `body_height` |
| 增强几何 | `57..63` | 7 | `cg_y_norm` / `foot_norm` / `com_x` / `com_y` / `left_hip_angle` / `right_hip_angle` / `torso_angle` |

几何量的公式：

```python
# 躯干倾角：肩中点相对髀中心的偏移
dx, dy = shoulder_center - hip_center
torso_angle = degrees(arctan2(|dx|, |dy|))       # 0° = 直立, 90° = 平躺

# 关节角：_angle(a, b, c) 即 ∠abc，b 为顶点
left_knee_angle = _angle(lhip, lkne, lank)
left_hip_angle  = _angle(shoulder_center, lhip, lkne)

# 质心：四点平均
com = (lhip + rhip + lsho + rsho) / 4
```

`_angle` 内部为 `arccos(ba · bc / (|ba| |bc|))`，两点重合时返回 0。

**实测样例**（放在画面右下角的站姿骨架，`bh = 160px`、`scale = 0.625`）：

```
[  0.    -37.5     1.      2.5   -40.      0.988  -2.5   -40.      0.975]   <- [0:9]

[51] cg_y               =       0.0000   <- 恒为 0
[52] torso_angle        =       0.0000   <- 站直
[53] left_knee_angle    =     180.0000   <- 站直，膝角 180°
[54] right_knee_angle   =     180.0000
[55] foot_distance      =      12.5000
[56] body_height        =     100.0000   <- 恒为 100
[57] cg_y_norm          =       0.0000
[58] foot_norm          =       0.1250   <- foot_distance / 100
[59] com_x              =       0.0000
[60] com_y              =     -14.0625   <- 质心在髀中心上方
[61] left_hip_angle     =     171.0652
[62] right_hip_angle    =     171.0652
[63] torso_angle        =       0.0000   <- 与 [52] 完全相同
```

### 3.2 使用前必须先标准化

原始 64 维各维**尺度差异极大**：`conf` 在 `0~1`、坐标在 `±100`、角度在 `0~180`。
因此 `LSTMClassifierV2.normalize()` 会先用训练集统计的 `mean/std` 逐维标准化，再送进 LSTM。

> 这也是 `models/norm_v2.npz` 不可或缺的原因 —— 缺了它模型无法正确推理。

### 3.3 四个易踩的坑

| # | 位置 | 问题 |
|---|---|---|
| 1 | `FeatureBuilderV2.build` | `confs=None` 时内部用 `np.ones(17)` 兜底，17 个 `conf` 维退化为常量 1.0，等于白占维度 |
| 2 | `normalize_keypoints` | 归一化锁死 `cg_y = 0`、`body_height = 100`，**绝对尺度信息完全丢失**（详见 §8.1） |
| 3 | `torso_angle` | 公式取 `arctan2(|dx|, |dy|)`，输出恒在 `[0°, 90°]` —— **丢失方向**，前倾/后仰、左倾/右倾无法区分 |
| 4 | `realtime_detect.py` | 归一化失败时兜底走 `build(kpts, kpts_conf)`，即用**未归一化的原始像素坐标**建特征，与训练分布不符（见 §6.4） |

### 3.4 192 维的构成

`[位置(64) | 速度(64) | 加速度(64)]`，其中速度/加速度由**一阶/二阶差分**得到：

```
vel[0] = 0,  vel[t] = pos[t] - pos[t-1]
acc[0] = 0,  acc[t] = vel[t] - vel[t-1]  =  pos[t] - 2·pos[t-1] + pos[t-2]
```

三点性质：

1. **不是新采集的数据**。`vel` / `acc` 都是 `pos` 的确定性函数，逐元素差分**不改变形状**，
   所以 `(30,64) → (30,192)` 是同一份数据的三种视角。
   —— **192 = 64 维信息 × 3 种视角，不是 3 倍信息**。其价值在于把"变化"显式化，
   免去线性层自己学差分。
2. **有两处虚假边界值**。`vel[0] = 0` 是硬编码的（无 `t-1` 可减），且会污染 `acc[1]`：
   实测某窗口 `acc[1]` 真实值 `0.6898` 被算成 `0.1437`。详见 §9.1。
3. **26.6% 的维度与置信度相关**（51/192），且其中 34 维价值可疑。
   `pos` 里的 17 个 `conf` 维提供“可信度权重”（合理），
   但 `vel` / `acc` 里各 17 个是**置信度的差分** —— 那是差分“盲操作”的副作用，
   不是设计选择。实测其 SNR 仅 `0.002`（坐标维 `0.593`）。详见 §9.4。

---

## 四、模块速查

| 路径 | 职责 | 关键点 |
|---|---|---|
| `realtime_detect.py` | 推理主循环 | 192 维拼接、无人帧填充、单目标 |
| `pose/pose_extractor.py` | YOLO26n-Pose 封装 | `conf=0.05`、三重质量门、重叠合并 |
| `tracker/person_tracker.py` | 质心跟踪 | **未被调用** |
| `features/normalize.py` | 关键点归一化 | 髋中心平移 + 身高缩放到 100px |
| `features/feature_builder_v2.py` | 64 维特征 | 51 + 6 + 7 |
| `sequence/sequence_buffer.py` | 30 帧滑窗 | per-person deque，NaN 丢弃 |
| `classifier/risk_classifier_v2.py` | 网络定义 | 双池化、三头、激活差异 |
| `classifier/lstm_classifier_v2.py` | 推理封装 | mean/std、温度缩放、time 还原 |
| `pipeline/risk_engine_v2.py` | 状态机 | 安全门、EMA、三计数器 |
| `scripts/build/build_dataset_v2.py` | 数据构建 | 6 条标签规则、滑窗 STRIDE=3 |
| `scripts/build/extract_keypoints.py` | 关键点缓存生成 | 逐帧 PoseExtractor、多人取 conf 和最大者 |
| `scripts/train/train_v2.py` | 训练 | MultiTaskLoss、3-fold LOSO、温度标定 |
| `scripts/eval/trace_shapes.py` | 形状 / 参数量 / 冗余自查 | 无需权重文件即可运行 |

### 核心常量一览

| 模块 | 常量 | 值 |
|---|---|---|
| `build_dataset_v2.py` | `WINDOW_SIZE` / `STRIDE` | 30 / 3 |
| `build_dataset_v2.py` | `TAU` / `PRE_FALL` / `NORMAL_TIME` | 50 / 120 / 999 |
| `train_v2.py` | `EPOCHS` / `BATCH_SIZE` / `HIDDEN_SIZE` | 50 / 256 / 128 |
| `train_v2.py` | `LAMBDA_RISK` / `LAMBDA_TIME` | 0.30 / 0.10 |
| `RiskEngineV2` | `alpha` / `warning_thres` | 0.4 / 0.35 |
| `RiskEngineV2` | `trigger_frames` / `lock_frames` | 10 / 15 |

---

## 五、关键设计决策

### 5.1 不设 WARNING 类，改用回归头表达连续风险

规避"WARNING 类需要足够多标注数据"的难题，同时得到比离散类更细的风险梯度。
训练标签为指数衰减：`risk = exp(-time_frames / TAU)`，`TAU=50`：

| 距跌倒完成帧数 | 风险值 |
|---|---|
| 0 | 1.000 |
| 10 | 0.819 |
| 30 | 0.549 |
| 50 | 0.368 |
| 120（`PRE_FALL` 边界） | 0.091 |

**风险回归头的三档实际来源**（A/B 实测，见 §8.4）：

| 真实风险档 | 来源 | 样本数 |
|---|---|---|
| 0.0 | Le2i 正常 / Pre-VFall Normal | 15,924 |
| 0.5 | **仅** Pre-VFall Abnormal（异常步态未跌倒） | 1,806 |
| 1.0 | Le2i 跌倒 / Pre-VFall Fall | 3,184 |
| 0–1 连续 | Le2i 跌倒前指数衰减（依据上表） | 3,392 |

即**预警档 0.5 完全依赖 Pre-VFall 的 Abnormal 片段**；仅用 Le2i 时该档样本数为 0。

### 5.2 双池化

```python
pooled = 0.5 * out[:, -1, :] + 0.5 * out.mean(dim=1)
```

兼顾"末帧即时状态"与"整窗平均趋势"，对滑窗相位不敏感。

### 5.3 两道安全门（`_compute_danger_prob`）

把分类头的**语义硬约束**灌进回归头，防止回归值在语义矛盾时误导状态机：

| 分类结果 | 处理 |
|---|---|
| `SITTING` / `NON_PERSON` | 风险强制归零 |
| `FALLEN` | 风险抬到 ≥ 0.7 |
| `SAFE` | 直接采用回归值，裁剪到 `[0,1]` |

### 5.4 `conf=0.05` 的低阈值 + 质量门控

人跌倒后 YOLO-Pose 检测置信度大幅下降（检测率 5% → 55%），故把 `conf` 从 0.25 降到 0.05；
但低阈值会引入家具误检，用 `_is_valid_person` 的三重门补回来：

- 有效关键点（`conf>0.3`）数量 ≥ 4
- 有效关键点空间跨度 ≥ 8px
- 平均关键点置信度 ≥ 0.3
- 附加条件：`bbox score < 0.15` **且** 跨度 < 35px → 判为物品

### 5.5 重叠检测合并使用 IoU + containment 双条件

跌倒后 YOLO 常把一个人检测成多个框（头一个 + 身体一个）。
单靠 IoU 无效（小框在大框内时 IoU 可能 < 0.3），故补充
`containment = intersection / min(areaA, areaB)`，阈值 0.6。

### 5.6 温度缩放只作用于分类头

`RiskClassifierV2` 分类头输出除以温度 `T` 后再 softmax，用于校准置信度；
风险头是 Sigmoid 回归，不做温度缩放。

---

## 六、仓库现状与偏差

### 6.1 缺失的运行时产物

仓库为**纯代码**，缺失全部权重与训练数据：

| 缺失项 | 影响 |
|---|---|
| `models/` 整个目录 | 无 `yolo26n-pose.pt`、`lstm_v2.pt`、`norm_v2.npz`、`temperature_v2.npz` |
| `yolo_keypoints/` 目录 | 无预提取关键点缓存，需用 `scripts/build/extract_keypoints.py` 重新抽点 |
| `processed/*.npy` | 仅有 3 个 json meta，无 `x_v2.npy` 等训练数据 |
| Le2i 原始视频与标注 | 不含于仓库，需从 [Kaggle: falldataset-imvia](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) 获取（190 个 `.avi` + 130 份标注，约 16 GB），并按 `build_dataset_v2.py` 的 `ARCHIVE` 结构摆放 |
| Pre-VFall 原始数据 | 不含于仓库，需从 [Figshare v3（Pre-VFall.zip）](https://figshare.com/articles/dataset/Pre-VFall_Vision_Sensor_Simulated_Early_Signs_of_Fall_Dataset/26488216/3?file=48148606) 获取（约 19.6 GB，CC BY 4.0，MD5 `45dc0cc5134ea3a9120b81271dd6ae5e`）；解压后已含逐帧 OpenPose JSON，用 `build_prevfall_dataset.py` 转换即可（6 场景 / 108 受试者 / 287 段目录 / 22,559 个 JSON → 4888 个窗口） |

**结论：clone 后无法直接推理**，必须先重建数据集并训练，或从别处获取权重。

依赖清单已补入仓库根目录的 `requirements.txt`（原本连依赖声明都没有）。

### 6.2 硬编码路径

`scripts/build/build_dataset_v2.py` 中：

```python
ARCHIVE = r"C:\Users\y'y'x\Downloads\archive"
```

Windows 绝对路径，在 macOS/Linux 上需修改后方可重建数据集。

`scripts/build/build_prevfall_dataset.py` 中（Pre-VFall 数据根目录，默认值同样硬编码 Windows 路径，但**已支持 `--data-dir` 覆盖**）：

```python
PREVFALL_DIR = r"/Volumes/media-backup/AI-data/falldown/Pre-VFall"   # 可用 --data-dir 覆盖
```

此外 `extract_keypoints.py` 的 `ARCHIVE` 与 `KEYPOINTS_DIR` 也已硬编码为绝对路径。

### 6.3 代码与文档的偏差

| 项 | README 描述 | 代码实际 |
|---|---|---|
| 版本 | V4 / 5 类 | V2 / 4 类 |
| 跟踪 | 完整 `PersonTracker` 链路 | **未接入**，`person_id` 硬编码 0（详见 §9.1） |
| 权重 | `models/` 23 个文件 | 目录不存在 |
| 目录 | 含 `configs/`、`PROJECT_DOC.md` 等 | 不存在 |

### 6.4 其他已知问题

| 位置 | 问题 |
|---|---|
| `sequence_buffer.py` | 默认 `feat_dim=34` 与实际 64 不符（仅文档用途，`update` 不校验维度） |
| `sequence_buffer.py` | `get_velocity/get_acceleration` 与 `realtime_detect.build_192_feature` 逻辑重复 |
| `build_dataset_v2.py` | `subsample_to_10fps` 内部 `step=1`，实为空操作，函数名误导 |
| `train_v2.py` | `scaler` 定义在 `main()` 内，`train_fold` 通过全局引用（可运行但不健壮） |
| `realtime_detect.py` | 多目标场景下所有人共用一个 `person_id=0`，缓冲区互相污染（详见 §9.1） |
| `realtime_detect.py` | 归一化失败（`bh < 5`）时兜底用**未归一化的原始像素坐标**建特征，属分布外输入 |
| `feature_builder_v2.py` | `_angle_from_vertical` 已定义但全仓库无任何调用（死代码） |
| `feature_builder_v2.py` | `torso_angle` 取绝对值导致倾斜**方向信息丢失**（前倾/后仰不可区分） |
| `build_prevfall_dataset.py` | ~~统计打印 `risk=0.4` 与实际常量 `RISK_ABNORMAL=0.5` 不符，且漏印 `risk=1.0`~~ **已修复**：改为按常量打印三档并附跳过明细 |
| `build_prevfall_dataset.py` | ~~`PREVFALL_DIR` 硬编码 Windows 路径，无 CLI 参数；数据缺失时直接 `sys.exit(1)`~~ **已修复**：新增 `--data-dir`，并在校验段落目录时兼容下划线/连字符两种命名、跳过前缀不符的错位目录 |
| Pre-VFall 数据 | 42 个段落不足 `WINDOW_SIZE=30` 帧（多为 Fall 段，22–29 帧）而被跳过；`confusion_*` 两场景无 Fall 段，共 36 个受试者缺该类别 |

### 6.5 已验证的运行环境

仓库内 `.conda/` 是工作区本地环境（Python 3.14.7）。按 `requirements.txt` 安装后
**实测通过**的版本组合：

| 包 | 版本 |
|---|---|
| numpy | 2.5.3 |
| torch | 2.14.0 |
| torchvision | 0.29.0 |
| opencv-python | 5.0.0.93 |
| ultralytics | 8.4.163 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |

冒烟测试覆盖范围：

- 项目模块在 numpy 2.x 下正常导入（无 `np.float_` / `np.NaN` 等已移除别名的用法）
- `torch.load(state_dict)` 在 torch 默认 `weights_only=True` 下可用 —— 对应
  `LSTMClassifierV2.__init__` 的加载路径
- `torch.cuda.amp.autocast` 与 `torch.cuda.amp.GradScaler` **仍存在** ——
  `train_v2.py` 依赖这两个已弃用 API，暂不受影响
- `cv2` 的 `VideoCapture` / `rectangle` / `putText` / `FONT_HERSHEY_SIMPLEX` 等调用正常
- 全链路前向：骨架 → 64 维特征 → 30 帧序列 → `(30,192)` → 模型 → `(1,4)/(1,)/(1,)`

> ⚠️ 这套版本显著新于项目开发时所用（numpy 1.24 / opencv 4.x / Python 3.10）。
> 当前实测兼容，但 `numpy 2.x` 与 `opencv 5.0` 都是大版本跃迁。若后续出现难以定位的
> 行为差异，可回退到 `requirements.txt` 注释中记录的 3.10 组合。

> ℹ️ `.vscode/settings.json`（受 git 跟踪）中 `python-envs.defaultEnvManager` 被设为
> `conda`，Python Environments 扩展会据此在仓库根创建/使用 `.conda`。
> 换用其他解释器时需留意这一点。

---

## 七、学习路径

按**执行顺序**阅读，不要按目录顺序。每阶段都有可验证的产出。

### 阶段 0 · 环境跑通（0.5 天）

目标：让画面出现 bbox 和状态文字。

1. 补依赖：`ultralytics`、`torch`、`opencv-python`、`numpy`、`scikit-learn`、`scipy`
2. 获取 `models/yolo26n-pose.pt`（Ultralytics 官方可下载）
3. 跑 `python realtime_detect.py <video>`

**验收**：画面出现 `ID:0 SAFE risk:0.0xx` 叠加层

> 若只有 YOLO 权重而无 LSTM 权重，需先走阶段 5 重建数据集并训练。

### 阶段 1 · 姿态层 `pose/pose_extractor.py`（1 天）

- 通读全文，重点理解 `_is_valid_person` 的阈值取舍与 `_merge_overlapping_detections` 的双条件
- 练习：单张图跑 `extract`，打印每个候选的 `n_valid / spread / mean_conf`，观察过滤边界

**验收**：能解释"为什么 `conf=0.05` 反而比 0.25 好"

### 阶段 2 · 特征工程 `features/`（1–2 天）

- 读 `normalize.py` → `feature_builder_v2.py`，画出 51+6+7 的构成表
- **关键洞察**：因为**先归一化再建特征**，`hip_center` 被平移到原点、身高被缩放到 100，
  导致部分维度退化为常量或完全重复（实测结论见 §8.1）
- 跑 `python scripts/eval/trace_shapes.py` 亲手看一遍 64 维的方差与相关矩阵

**验收**：能说出至少 3 个冗余维度及其成因

**这是全项目性价比最高的改造层**：删冗余维（64 → 57）重训，看是否无损。

### 阶段 3 · 时序建模 `classifier/`（2–3 天）

- `risk_classifier_v2.py`：三个头的激活差异（分类无激活 / 风险 Sigmoid / 时间无激活）+ 双池化
- `lstm_classifier_v2.py`：训练集 mean/std 归一化、温度缩放只作用于分类头、`time` 输出乘 `PRE_FALL` 还原
- 练习：手造假 batch 过 `forward` 验证形状；把温度 1.0 → 2.0 观察 softmax 变化

**验收**：能解释三种损失函数各自的适配理由，以及 time 损失为何只在 `risk>0 & time<1.0` 样本上计算

### 阶段 4 · 决策层 `pipeline/risk_engine_v2.py`（1–2 天）

- 三个计数器的语义区分（见 2.2 节）
- 练习：写独立脚本直接喂人造 `(class_id, risk)` 序列给 `RiskEngineV2`，
  观察 SAFE → WARNING → DANGER → WARNING 的完整时序

**这一层不用重训模型就能做 A/B 实验**，是调参首选。

### 阶段 5 · 训练闭环 `scripts/`（3–5 天）

- `build_dataset_v2.py`：6 条标签规则的**优先级**（已倒地 → 标注倒地 → 末尾坐姿 → 全局坐姿 → 跌倒前预警 → 兜底）
- `train_v2.py`：`MultiTaskLoss` 权重权衡；3-fold LOSO（Leave-One-Scene-Out）为何比随机划分可信
- 前置：修改 `ARCHIVE` 硬编码路径；确保 Le2i 数据集（[Kaggle: falldataset-imvia](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)）与 `yolo_keypoints/` 缓存就位
- 练习：跑通训练，看 confusion matrix，调 `class_weights` 观察 FALLEN 召回敏感度

**验收**：能从 `x_v2.npy` 一路追到 `temperature_v2.npz` 的生成过程

### 阶段 6 · 系统集成改造（持续）

按性价比排序：

1. **接入 `PersonTracker`**（首选）—— 消除 `person_id` 硬编码，形成"检测 → 跟踪 → 特征 → 时序"完整多目标链路
2. 消除 `build_192_feature` 与 `SequenceBufferV3` 的重复实现
3. 修正 `feat_dim` 默认值
4. 抽出 `configs/` 集中管理超参
5. 更新 `README.md` 使其与 V2 代码对齐

---

## 八、待验证清单

以下条目按优先级排列。§8.1 已由脚本实测确认，其余仍为静态阅读的推断。

### 8.1 特征冗余（✅ 已实测确认）

由 `scripts/eval/trace_shapes.py` 用 800 组合成骨架（含随机尺度、平移、扰动与仿真置信度）
实测 64 维的方差与相关矩阵，确认 **7 个维度可剔除 → 有效维度约 57**。

**常量维度（3 个）**

| 索引 | 维度 | 实测 | 成因 |
|---|---|---|---|
| 51 | `geo.cg_y` | 均值 ≈ 0，std 7e-6 | `cg_y = hip_center[1]`，而 `hip_center` 已被归一化平移到原点 |
| 56 | `geo.body_height` | 均值 = 100，std 1e-5 | 归一化用的 `bh` 与 `body_height` 计算公式完全相同 |
| 57 | `geo2.cg_y_norm` | 均值 ≈ 0 | 即 `cg_y / 100` |

**完全相关维度对（4 对，每对可剔除 1 维）**

| 索引对 | 维度 | r | 成因 |
|---|---|---|---|
| 33 ↔ 36 | `kpt.lhip.x` ↔ `kpt.rhip.x` | -1.0 | 归一化后髋部关于原点对称，`rhip = -lhip` **恒成立** |
| 34 ↔ 37 | `kpt.lhip.y` ↔ `kpt.rhip.y` | -1.0 | 同上 |
| 52 ↔ 63 | `geo.torso_angle` ↔ `geo2.torso_angle` | +1.0 | 同一表达式被计算了两次 |
| 55 ↔ 58 | `geo.foot_distance` ↔ `geo2.foot_norm` | +1.0 | 仅差常数因子 100 |

其中 (33↔36) 与 (34↔37) 是**静态阅读时未预料到的**：`normalize_keypoints` 以
`hip_center = (lhip + rhip) / 2` 为原点，故归一化后必然有 `rhip = -lhip`，与输入无关。

**副产物：绝对尺度信息完全丢失**

由于 `cg_y` 与 `body_height` 被归一化锁定为常量，64 维中不含任何绝对尺度信息：

- 远景的人与近景的人无法区分 —— 尺度不变性，通常有利于泛化
- 无法利用「髋部离地面多高」这一直接的跌倒线索 —— 不利，但该量本身受摄像头高度与
  安装角度影响，直接使用会损害跨场景泛化

**复核方式**：`python scripts/eval/trace_shapes.py --samples 2000`

> ⚠️ 该脚本必须给关键点传入**变化的**置信度。若一律传 1.0，17 个 `.conf` 维度会退化为
> 常量，在冗余统计中产生假阳性 —— 这是造数据的假象，不是特征本身的性质。

> 💡 **删常量维还有一个附带好处**：这些维度在训练集上近乎恒定，
> 会让 `LSTMClassifierV2.normalize` 的 `std + 1e-8` 防护变得脆弱（详见 §9.5）。
> 从根上删掉它们，也就同时消除了这个隐患。

### 8.2 训练数据增强的维度对齐（中优先级）

`train_v2.py` 的 `_augment` 中两处增强使用了**不同的布局假设**：

- 噪声增强：`x[:, :34] += randn * 0.015`，按"先全部 x 再全部 y"的平面布局理解
- 遮挡增强：`x[fstart:fstart+nf, off + kidx*3 : off + kidx*3 + 2] = 0`，按 `(x,y,conf)` 交错布局理解

而 `FeatureBuilderV2.build` 实际产出的是**交错布局**（`x0,y0,c0,x1,y1,c1,...`）。
故噪声增强实际只覆盖了前 11 个关键点。

**验证方法**：打印 `np.flatnonzero(x.std(dim=0) > 0)` 确认噪声影响范围，或按交错布局改为 `x[:, :51]`。

### 8.3 其余可验证项

- `subsample_to_10fps` 在 `step=1` 下不改变帧率，"10fps" 命名与 `TARGET_FPS=25` 常量均具误导性
- 推理滑窗为 stride=1（每帧更新），训练滑窗为 `STRIDE=3`，存在轻微分布差异
- `RiskEngineV2` 的 `lock_count` 在 `elif smooth >= warning_thres` 分支被重置为 `lock_frames`，
  导致 WARNING 状态在风险回落后仍被强制保持 15 帧 —— 需确认这是预期行为

### 8.4 风险回归头的 0.5 预警档是否真的学到（✅ 已实测确认）

**背景**：`0.5` 档仅由 Pre-VFall 的 Abnormal 片段提供（见 §5.1）。在接入 Pre-VFall 之前，
全数据集该档样本数为 **0**，风险回归头实际只在 0.0 / 1.0 两个端点上训练，预警能力无从学起。

**方法**：同一份代码与超参，仅更换训练集做 A/B：

| | A 组（Le2i + Pre-VFall） | B 组（仅 Le2i） |
|---|---|---|
| 样本数 | 24,306 | 19,418 |
| 数据中 `risk=0.5` 标注数 | **1,806** | **0** |
| Risk MAE（脚本自报） | 0.0223 | 0.0237 |
| 阈值准确率（t=0.35） | 98.55% | 98.46% |

**按数据来源分档统计预测值**（`scripts/eval/analyze_risk_buckets.py`，A 组）：

| 来源 / 真实标签 | 样本数 | 预测均值 | 预测 std | 判为预警(≥0.35) |
|---|---|---|---|---|
| Pre-VFall Normal (0.0) | 2,908 | 0.0045 | 0.0017 | 0.0% |
| **Pre-VFall Abnormal (0.5)** | **1,806** | **0.5012** | **0.0093** | **99.9%** |
| Pre-VFall Fall (1.0) | 174 | 0.9980 | 0.0006 | 100.0% |
| Le2i risk=0.0 | 13,016 | 0.0157 | 0.0245 | 0.0% |
| Le2i 指数衰减 (0<r<1) | 3,392 | 0.2645 | 0.2482 | 33.0% |
| Le2i risk=1.0 | 3,010 | 0.9971 | 0.0076 | 100.0% |

**结论**：预警能力从"不可学"变为"可学且已学到" —— A 组在 0.5 档的预测均值 0.5012、
标准差仅 0.0093，几乎精准命中目标；B 组因不存在该档标注，只能学到两个端点。
A 组同时未牺牲原有能力（Le2i 的 0.0 与 1.0 档预测依然准确）。

> **统计陷阱**：若按"预测值落在 0.45–0.55"分档，B 组会出现 312 条"0.5 档"样本，
> 但它们是 Le2i 的**指数衰减**标签恰好落在该区间，并非真异常标注。
> 因此分档必须结合**数据来源**（`scene_id == 10`）而非仅看数值。

---

## 九、后续优化方向

记录已识别、但**当前暂不处理**的改进点，按优先级排列。

### 9.1 多目标支持：当前实为单目标处理（优先级：高）

#### 现状

`realtime_detect.py:142` 把 `person_id` 硬编码为 `0`：

```python
for person in persons:        # 确实遍历了所有人
    ...
    person_id = 0             # 但 ID 恒为 0
    buffer.update(person_id, feature)
```

画面里 N 个人全部写进同一个 `buffers[0]`，并共用同一个 `RiskEngineV2` 状态。

四层对照 —— **底层本就是按多人设计的，只有驱动层没把真实 ID 传下去**：

| 层 | 是否支持多人 | 证据 |
|---|---|---|
| `SequenceBufferV3` | ✅ | `self.buffers = {}` → `{person_id: deque}` |
| `RiskEngineV2` | ✅ | `smooth_score` / `counter` / `danger_confirmed` / `lock_count` 均为 dict |
| `draw_result` | ✅ | 已渲染 `ID:{person_id}` |
| `PersonTracker` | ✅ | `tracker/person_tracker.py` 已实现，含 `next_id` / `max_missing` |
| **`realtime_detect.py` 主循环** | ❌ | `person_id = 0` |

#### 问题 1：30 帧窗口被按人数平分

`deque` 的 `maxlen = 30` 限制的是**条目数而非帧数**：

| 人数 | 每帧写入条目 | 窗口跨越帧数 | 每人分到的帧 |
|---|---|---|---|
| 1 | 1 | 30 | 30 |
| 2 | 2 | 15 | 15 |
| 3 | 3 | 10 | 10 |

LSTM 是在“30 个连续帧”上训练的，混写后实际时间尺度按人数成比例缩短。

#### 问题 2：不同姿态交替写入 → 幽灵振荡

先要澄清一个**反直觉**的点：归一化会把位置与尺度差异**完全抹平**，
所以两个人分别站在画面两端**不会**造成伪跳变：

| 对比 | 特征最大差异 |
|---|---|
| 同姿态 @ `(300,400)` vs @ `(900,200)` | `0.000000` |
| 同姿态 1× 身高 vs 2× 身高 | `0.000000` |

真正的污染来自**姿态差异**。以「A 站立 + B 倒地」为例，实测：

| 指标 | 单人站立 | 站立 + 倒地混写 | 放大 |
|---|---|---|---|
| `vel` 平均幅度 | 0.686 | 18.649 | **27.2×** |
| `acc` 平均幅度 | 1.162 | 36.044 | **31.0×** |

`buffer[0]` 内部逐格交替：

```
 格   torso_angle      来源
  0           0.5    A 站立
  1          89.3    B 倒地
  2           0.1    A 站立
  3          89.5    B 倒地
```

`torso_angle` 在 `0°` / `90°` 之间逐格跳变 —— 真人不可能每帧在站立与倒地之间来回切换。
这是喂给 LSTM 的**幽灵振荡**。

顺带一提：正因为归一化抹平了位置与尺度，**姿势相近的两个人特征几乎完全相同**，
使这个错误格外隐蔽 —— 画面上两个框看起来都在正常跟踪。

#### 问题 3：风险引擎计数被按人数放大

`risk_engine.update()` 位于 per-person 循环内，一帧被调用 N 次：

| 场景 | 触发 DANGER 的帧 |
|---|---|
| 单人 | 第 10 帧（`trigger_frames = 10`） |
| 双人 | 第 5 帧 |

EMA 平滑也在同一帧内被反复施加，等效时间常数随之改变。

#### 问题 4：视觉上会骗人

每个框用**自己的 bbox** 绘制，但状态与风险值来自**共享**引擎。后果是双向的：

- 一人跌倒 → 画面里**所有**框都变红显示 `FALL`
- 反之，站立者的特征也在稀释跌倒者的信号（弱化为 `1/N`）

#### 影响范围：只影响推理，不影响训练

| | 窗口来源 | 是否混人 |
|---|---|---|
| 训练 `build_dataset_v2.py` | 按视频加载一份 `(T,17,2)`，滑窗切 `features[start:end]` | ❌ 不混，每窗 30 帧来自同一视频 |
| 推理（现状） | 所有检测结果写进 `buffers[0]` | ✅ 混 |

所以准确的说法是**训练与推理分布不一致**。Le2i 的 `Coffee_room` / `Home_01` / `Home_02`
均为单人场景，训练窗口本身是干净的。

#### 提取脚本

`yolo_keypoints/` 缓存由 **`scripts/build/extract_keypoints.py`** 生成（190 个视频 × 2 = 380 个文件）：
逐帧调用 `PoseExtractor`（含物品误检质量门与分裂检测合并），多人场景取 `kpts_conf.sum()`
最大者，漏检帧写全 0 行。因此“取 score 最高者”确为当年的选人策略。

**验证结果**：按该策略重建缓存后，`Lecture_room` / `Office` / `Coffee_room_01` 相邻帧
髋部中点位移动幅度中位数仅 1–2px、最大 27px，**未出现 A/B 交替的锯齿**，逐帧选人稳定，
训练窗口未因选人跳变而混人。

#### 建议方案

改动集中在主循环：

1. 循环外创建一次 `tracker = PersonTracker(max_missing=30, distance_threshold=100)`
2. 每帧 `tracks, removed_ids = tracker.update(persons)`
3. 遍历 `tracks`（而非 `persons`），用 `track["id"]` 调用 `buffer` 与 `risk_engine`
4. 对 `removed_ids` 调用 `buffer.remove(tid)` 并清理引擎状态，避免 ID 累积导致内存增长
5. 绘制改用 `track["bbox"]`

#### 接入前必须解决的两个陷阱

**陷阱 A：`PersonTracker` 会丢掉关键点置信度**

`Track.to_dict()` 返回 `{id, bbox, keypoints, missing, age, hit_count}`，**没有 `kpts_conf`**；
`create_track` 与 `Track.update` 也都只接收 `bbox` / `keypoints`。

若直接使用，`person.get("kpts_conf", None)` 得到 `None` → `FeatureBuilderV2.build`
内部以 `np.ones(17)` 兜底 → **17 个 `conf` 维静默退化为常量 1.0**
（即 §3.3 的第 1 个坑）。

必须在 `track.py` 与 `person_tracker.py` 中让 conf 跟着走，
**否则模型静默降级且不报任何错**。

**陷阱 B：质心跟踪在跌倒瞬间最容易断**

匹配为**贪心最近质心**，阈值 `distance_threshold = 100` px，无 IoU 校验、无运动预测：

```python
dist = self.distance(person["bbox"], track.bbox)
if best_distance < self.distance_threshold:
    best_track.update(...)
else:
    new_track = self.create_track(person)    # 直接新建 ID
```

人跌倒时 bbox 由“竖长”突变为“横宽”，质心大幅下移，很容易超过 100px → 匹配失败 → 新建 ID：

- 新 ID 的 buffer 为空，需再攒 30 帧（25fps 下约 **1.2 秒**）才能推理
- 而跌倒恰恰发生在这段盲区内
- 旧 ID 停止更新，`max_missing = 30` 帧后被移除

此外贪心匹配无全局最优分配，两人靠近时也易**互换 ID**。

**可能的改进方向**（供后续取舍）：放宽 `distance_threshold`、改用 IoU + 质心组合度量、
引入运动预测（如匀速外推），或对已确认 DANGER 的 ID 加“锁定保护”，禁止其被新 ID 取代。

#### 顺带发现：另一处训练/推理不一致

| | vel/acc 计算范围 | 窗口第 0 帧的 vel/acc |
|---|---|---|
| 训练 `build_window_features` | 整段序列先算完再切窗口 | **非零**（与窗口外前一帧差分） |
| 推理 `build_192_feature` | 仅在窗口内计算 | `vel[0] = 0`、`acc[0] = 0` |

`vel[0]` 被硬编码为 0 后会**污染下一帧**：`acc[1] = vel[1] - vel[0]`，
使 `acc[1]` 也变成假值。实测（`start=10` 的窗口）：

| 区块 | 不一致时间步 | 不一致值数 |
|---|---|---|
| `pos` | 无 | 0 |
| `vel` | `t=0` | 64 |
| `acc` | `t=0`, `t=1` | 128 |
| **合计** | | **192 / 5760 = 3.33%** |

出现频率的差别更关键：

| | 带此缺陷的窗口比例 |
|---|---|
| 训练 | `1 / 31 ≈ 3.2%`（仅 `start=0` 的那个窗口） |
| 推理 | **100%**（每个窗口都在窗口内重算差分） |

即推理时**每一次**都带着这个伪影，而训练时模型几乎见不到它。
量不大，但与"两处拼接逻辑重复实现"（§6.4）同源，统一实现时一并解决。

#### 核心结论

> 模型是在“单个人 30 个连续帧”上训练的 → 推理就必须喂“单个人 30 个连续帧”。

这是训练方式决定的**硬约束**，不是可选优化。归一化本身没有问题
（它提供的位置/尺度不变性是必要设计），问题只在 `person_id = 0` 这一行。

### 9.2 `torso_angle` 的方向信息丢失（优先级：中）

#### 现状

`FeatureBuilderV2.build` 的躯干倾角公式：

```python
dx, dy = shoulder_center - hip_center
torso_angle = degrees(arctan2(|dx|, |dy|))
```

两个分量都取了绝对值，因此输出恒在 `[0°, 90°]` —— 只能表达“倾斜了多少度”，
**不能表达“往哪边倾”**。同时该值在索引 52 与 63 各出现一次（重复，见 §8.1）。

#### 为什么可能是问题

跌倒的**方向**是有判别力的信号：向前扑倒、向后仰倒、侧向倒地的姿态演化差异很大。
当前模型只能看到倾角的幅度，看不到符号。

#### 但关键坑在于：不能简单地“把绝对值去掉”

直觉上的修法是改成 `degrees(arctan2(dx, dy))`，但这样会引入**朝向混淆**：

> YOLO 的 `lsho` / `rsho` 是按**人体解剖学左右**标注的，不是图像左右。
> 当人背对镜头时，`lsho` 会出现在图像右侧，`dx` 的符号整体翻转。
>
> 结果：同一个“向左倒”的动作，人面朝镜头与背对镜头时会给出**相反**的 `torso_angle` 符号。

现在的绝对值写法很可能**本来就是刻意的**，用来消除这种朝向依赖。
所以真正的优化目标不是“恢复方向”，而是“在朝向无关的前提下拿到方向信息”。

#### 候选实现

| 方案 | 做法 | 是否朝向无关 |
|---|---|---|
| A. 竖向符号 | 增加 `dy / bh`（肩中点相对髀中心的归一化竖向偏移） | ✅ 是。头在上 / 头在下与朝向无关 |
| B. 倾斜向量 | 直接给出归一化的 `(dx/bh, dy/bh)` 两维，让模型自己学 | ⚠️ 否，与朝向耦合 |
| C. 交给时序 | 不改单帧特征，靠 `192` 维里的速度/加速度块捕捉躯干角的变化率 | ✅ 是，零改动 |

方案 C 值得先验证 —— `192` 维里的 `vel` / `acc` 已包含躯干角的一阶/二阶差分，
很可能已经隐含了方向信息，建议先做消融再决定要不要加维度。

#### 改动前必须知道的风险

1. **破坏维度契约**：`64 → 65+` 后 `models/norm_v2.npz` 的 `mean` / `std` 形状不匹配，
   必须重训模型并重新生成归一化参数
2. **静默错位**：`build_dataset_v2.py` 的 `is_sitting_posture_v2` 直接硬编码读取
   `feat[52] / feat[53] / feat[54] / feat[61] / feat[62]`。任何插维/删维都必须同步修改它，
   否则坐姿检测会静默拿到错误的量，**不会报错但结果全错**
3. **性价比**：当前误报的主因依我看在特征冗余（§8.1）与 SITTING / NON_PERSON 混淆，
   方向信息属于锦上添花，建议排在它们之后

#### 验证方法

- 先做消融：对真实视频统计跌倒前后 `torso_angle` 的分布，看“站→倒”的变化是否已足够可分
- 再用方案 C 对比：把 `vel` / `acc` 中的躯干角分量置零，看 FALLEN 召回率下降多少。
  若下降明显，说明方向信息已被隐含利用，无需再加维度

### 9.3 其他候选（已记录在别处）

| 优化点 | 详情 |
|---|---|
| 删冗余特征 7 维（`64 → 57`） | §8.1 |
| 消除 `build_192_feature` 与 `SequenceBufferV3` 的重复实现 | §6.4 / §9.1 |
| 修复 `_augment` 的维度错位 | §8.2 |
| 修复归一化失败时的兜底分支 | §3.3 / §6.4 |
| 修复窗口第 0 帧 vel/acc 与训练不一致 | §9.1 |
| conf 差分置零消融（`192 → 158`） | §9.4 |
| `normalize` 的 `std` 防护加固 | §9.5 |

### 9.4 conf 差分价值低：置零消融方案（优先级：中）

#### 问题：差分是“盲操作”

```python
vel[1:] = raw[1:] - raw[:-1]     # 它不知道哪一维是坐标、哪一维是 conf
```

`raw` 是 64 维整体，差分统一施加。所以 17 个 `conf` 维被差分**不是设计选择，而是副作用** ——
比例达 `51 / 192 = 26.6%`（`pos` 17 + `vel` 17 + `acc` 17）。

#### 为什么价值低：差分对信号与噪声的作用相反

| | 差分前 | 差分后 | 效果 |
|---|---|---|---|
| 平滑运动（坐标维） | std `8.95` | std `≈ 0`（恒速） | **信号被提取成小量** ✅ |
| iid 抖动（conf 维） | std `0.05` | std `0.071` | **噪声被放大 √2 倍** ❌ |

实测噪声放大 `1.4168` 倍（理论 `√2 = 1.4142`）；差分后的信噪比：

| 维度类型 | `vel` 的 SNR |
|---|---|
| 坐标维 | `0.593` |
| conf 维 | **`0.002`** |

`conf` 反映的是“YOLO 有多确信这个点可见”，没有“运动”可言。以一次真实遮挡为例
（`t=12` 进入遮挡、`t=17` 恢复）：

- 差分绝对值最大的时刻恰好是 `t=12` 与 `t=17` —— 遮挡的**开始与结束**
- 其余 `26 / 30 = 87%` 的帧是纯放大的噪声

#### 但也不是零信息，只是冗余

conf 差分的非零时刻确实标记了遮挡切换。但这份信息 `pos` 里的原始 `conf` 已经给了，
LSTM 有能力自己从 `conf` 序列学出“它刚掉下去了”。所以差分版是**冗余且以噪声为主**。

#### 建议的验证方式：置零消融

不删维度，只在差分后把 conf 位置清零：

```python
CONF_IDX = np.array([3 * k + 2 for k in range(17)])   # 2, 5, 8, ..., 50

vel[:, CONF_IDX] = 0
acc[:, CONF_IDX] = 0
```

两处都要改：`realtime_detect.build_192_feature` 与 `build_dataset_v2.build_window_features`。

**为什么用置零而不是删维度**：

| | 置零 | 真删维（`192 → 158`） |
|---|---|---|
| shape 不变 | ✅ 仍为 `192` | ❌ 需重新生成 `norm_v2.npz` |
| 影响 `is_sitting_posture_v2` 的硬编码索引 | ✅ 无（`pos` 索引未变） | ❌ 必须同步修改（§9.2） |
| 变量是否单一 | ✅ 只改信息量，容量不变，消融干净 | ❌ 同时改了容量，混淆因素 |
| NaN 风险 | ✅ 训练集上恒为 0 → `mean=0, std=0` → `(0-0)/1e-8 = 0` | — |

只需重训一次，对比 FALLEN 召回率与风险 MAE。**若指标不降，即确认这 34 维可删。**

#### 真删维的收益（供参考）

| 方案 | LSTM 第一层 `weight_ih` 参数量 | 相对现在 |
|---|---|---|
| 现在 `192` | 98,304 | — |
| 去掉 34 个 conf 差分 `158` | 80,896 | **−17.7%** |
| 再叠加 §8.1 的 7 个冗余维 `137` | 70,144 | **−28.6%** |

对全模型（322,502 参数）只省 5~9% —— 真正的收益在**降低过拟合风险**，不是省算力。

### 9.5 `normalize` 的 `std` 防护脆弱性（优先级：低）

#### 现状

```python
self.std = np.load(norm_path)["std"] + 1e-8
out = (x - mean) / std
```

`+1e-8` 是为防除零。但对**恒为常量的维度**（§8.1 列出的那些）防护不足：

| 训练集上的 `std` | 推理时的值 | 标准化后 |
|---|---|---|
| `0.0` | `1e-5` | **1000.0** ← 1000 倍离群输入 |
| `0.0` | `1e-6` | 100.0 |
| `1e-7` | `1e-5` | 90.9 |
| `1e-5` | `1e-5` | 1.0 ← 健康 |

#### 当前实测是安全的

```
[51] geo.cg_y        mean=0.0  std=7.698e-06  取值量级=4.554e-06   <- std 与取值同阶 ✓
[56] geo.body_height mean=100  std=1.193e-05  取值量级=1.000e+02   ✓
[57] geo2.cg_y_norm  mean=0.0  std=7.698e-08  取值量级=4.554e-08   ✓
```

`std` 与取值量级同阶 → 标准化后仍是 `O(1)`。但这依赖“训练与推理的浮点舍入噪声恰好一致”，
比较脆弱：若某次重构让训练侧恰好得到 `std = 0`，而推理侧仍有 `1e-5` 抖动，
该维度就会变成 1000 倍离群值。

#### 建议

1. **优先按 §8.1 删掉常量维** —— 从根上消除这些维度，也就不存在这个脆弱点
2. 若暂时保留，可把防护改成相对判据，对 `std` 过小的维度直接输出 0：
   ```python
   safe_std = np.where(raw_std < 1e-6, 1.0, raw_std)
   ```
   这样常量维度标准化后恒为 `0`，而非被放大
3. 注意改动会改变已有 `norm_v2.npz` 的语义，需重训或重新标定

---

## 附录：相关文档

- [`docs/classification-logic.md`](classification-logic.md) —— 4 类分类体系与标签生成规则详解
- `scripts/eval/trace_shapes.py` —— 形状、参数量与冗余维度自查工具
- `README.md` —— 项目总览（注意版本偏差）
