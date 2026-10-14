# 老年人跌倒检测系统 V4

> 基于姿态估计（YOLO26n-Pose）+ 时序深度学习（LSTM 多任务模型）的实时跌倒检测系统。  
> Real-time elderly fall detection using pose estimation + temporal deep learning.

[![Python](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/YOLO-26n--Pose-00BBFF)](https://www.ultralytics.com/)

---

## 目录

- [概述](#概述)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [流水线详解](#流水线详解)
- [环境配置](#环境配置)
- [数据集](#数据集)
- [模型训练](#模型训练)
- [推理检测](#推理检测)
- [验证测试](#验证测试)
- [模型清单](#模型清单)
- [关键设计决策](#关键设计决策)
- [性能指标](#性能指标)
- [已知限制](#已知限制)
- [项目演进](#项目演进)

---

## 概述

本系统通过摄像头捕获人体姿态关键点，利用 2 层 LSTM 多任务模型分析动作序列，实现**实时跌倒风险评估**，输出三种状态：

| 状态 | 含义 |
|-------|---------|
| **SAFE** | 正常行走 / 站立 / 坐下 |
| **WARNING** | 步态异常，可能跌倒（早期预警） |
| **DANGER** | 确认跌倒 |

模型同时预测：
- **类别**（4 类）：`SAFE(0)`、`FALLEN(1)`、`SITTING(2)`、`NON_PERSON(3)`
- **风险值** `[0, 1]`：连续跌倒概率（从跌倒时刻指数衰减）
- **距跌倒时间**：预估距离跌倒完成的帧数

---

## 系统架构

```
摄像头帧 (BGR)
  │
  ▼
┌─────────────────────────────────────┐
│ PoseExtractor                       │  YOLO26n-Pose → 17 个 COCO 关键点
│ pose/pose_extractor.py              │  质量门控 + 重叠检测合并
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ PersonTracker                       │  质心跟踪，为每人分配唯一 ID
│ tracker/person_tracker.py           │  距离阈值匹配（100px）
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ FeatureBuilderV2                    │  64 维生物力学特征：
│ features/feature_builder_v2.py      │  - 51 维：17 × (x, y, conf)
│                                     │  - 6 维：原始几何（角度、高度）
│                                     │  - 7 维：增强几何（归一化、质心）
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ SequenceBuffer                      │  30 帧滑动窗口（deque）
│ sequence/sequence_buffer.py         │  每人独立缓存，NaN 过滤
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 特征拼接                            │  [位置(64) | 速度(64) | 加速度(64)] = 192 维
│ realtime_detect.py:build_192_feature│
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ RiskClassifierV2 (多任务 LSTM)      │  三个并行预测头：
│ classifier/risk_classifier_v2.py    │  - 分类头：128→64→4（交叉熵损失）
│                                     │  - 风险头：128→64→1 + Sigmoid（MSE 损失）
│                                     │  - 时间头：128→64→1（SmoothL1 损失）
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ RiskEngineV2                        │  EMA 平滑 + 状态机：
│ pipeline/risk_engine_v2.py          │  SAFE →(≥0.35)→ WARNING →(连续10帧)→ DANGER
│                                     │  安全门：SITTING/NON_PERSON → 风险值强制归零
└──────────────┬──────────────────────┘
               │
               ▼
         可视化输出（边界框 + 风险值 + 状态颜色）
```

### 状态机

```
SAFE ──(平滑风险 ≥ 0.35)──► WARNING ──(连续计数 ≥ 10)──► DANGER
  ▲                                                          │
  └────────────────────(计数重置)─────────────────────────────┘
  ◄──(连续 SAFE ≥ 15 帧, 且 class=SAFE, 风险<0.35)────────────┘
```

---

## 目录结构

```
elder-master-reformv4/
│
├── build_dataset_v2.py          # V2 数据集构建（4 类 + 风险回归）
├── build_dataset_5class.py      # 5 类数据集构建（含 SITTING）
├── build_prevfall_dataset.py    # Pre-VFall 数据集转换
├── train_v2.py                  # V2 多任务 LSTM 训练（4 类 + 风险 + 时间）
├── train_5class.py              # 5 类 LSTM 训练
├── realtime_detect.py           # 实时检测演示（V4 流水线）
├── scan_sit_videos.py           # 坐下误报扫描工具
├── verify_issues.py             # P1/P2 回归验证
├── verify_real_videos.py        # 跌倒/正常视频回归测试
│
├── classifier/                  # 模型定义与推理封装
│   ├── risk_classifier.py       #   5 类 LSTM 模型定义
│   ├── risk_classifier_v2.py    #   V2 多任务模型（3 个预测头）
│   ├── lstm_classifier.py       #   5 类推理封装
│   └── lstm_classifier_v2.py    #   V2 推理封装
│
├── features/                    # 特征工程
│   ├── feature_builder_v2.py    #   64 维生物力学特征构建器
│   ├── normalize.py             #   关键点归一化（髋中心平移 + 缩放）
│   ├── label_generator_v3.py    #   风险/时间标签生成器
│   └── gait_cycle_detector.py   #   步态周期检测
│
├── pose/                        # 姿态估计
│   └── pose_extractor.py        #   YOLO26n-Pose 封装 + 质量门控
│
├── tracker/                     # 多目标跟踪
│   ├── track.py                 #   跟踪目标数据结构
│   └── person_tracker.py        #   质心跟踪器
│
├── sequence/                    # 序列缓冲
│   └── sequence_buffer.py       #   30 帧 deque，自动计算速度/加速度
│
├── pipeline/                    # 流水线调度与状态机
│   ├── risk_engine.py           #   REFORM 版本（4 类/5 类）
│   └── risk_engine_v2.py        #   V2 版本（双头融合）
│
├── models/                      # 已训练权重（23 个文件）
│   ├── yolo26n-pose.pt          #   YOLO 姿态模型
│   ├── lstm_v2.pt               #   V2 多任务模型（最终版）
│   ├── lstm_v2_fold{1-3}.pt     #   V2 交叉验证折叠模型
│   ├── lstm_5class.pt           #   5 类模型（最终版）
│   ├── lstm_4class.pt           #   4 类模型（REFORM 遗留）
│   ├── norm_v2.npz              #   V2 归一化参数
│   └── temperature_v2.npz       #   温度缩放值
│
├── processed/                   # 预处理后的训练数据
│   ├── x_v2.npy                 #   V2 特征 (M, 30, 192)
│   ├── class_labels_v2.npy      #   V2 类别标签
│   ├── risk_labels_v2.npy       #   V2 风险回归标签
│   └── x.npy                    #   5 类特征 (N, 30, 192)
│
├── yolo_keypoints/              # 预提取的 YOLO 关键点（190 视频 × 2 = 380 个文件）
│                                #   由 scripts/build/extract_keypoints.py 生成
│
├── configs/                     # 配置目录（待开发）
├── docs/                        # 文档
│   ├── annotation-guide.md      #   逐帧标注指南
│   └── refactoring-plan.md      #   未来重构计划
│
├── PROJECT_DOC.md               # 内部技术文档
├── REFORM.md                    # REFORM 重构说明
├── SOLUTION.md                  # Windows 用户名特殊字符解决方案
├── README.md                    # 本文件
└── .opencode/                   # OpenCode AI 配置
```

---

## 流水线详解

### 1. 姿态估计（`pose/pose_extractor.py`）

**模型**：YOLO26n-Pose（Ultralytics）  
**关键点**：17 个 COCO 关键点（鼻子、眼睛、肩膀、肘部、手腕、髋部、膝盖、脚踝）

**质量过滤**：
- 置信度阈值：`conf=0.05`（低阈值以捕捉跌倒后的人体）
- `_is_valid_person()`：≥4 个有效关键点（置信度>0.3）、空间跨度≥8px、平均置信度≥0.3
- `_merge_overlapping_detections()`：IoU>0.3 或包含度>0.6 时合并（修复跌倒后检测分裂）
- 平均关键点置信度 <0.3 → 丢弃（过滤家具误检）

### 2. 人员跟踪（`tracker/person_tracker.py`）

**算法**：质心跟踪（基于距离匹配）

| 参数 | 值 | 说明 |
|-----------|-------|-------------|
| `max_missing` | 30 帧 | 跟踪目标最大丢失帧数 |
| `distance_threshold` | 100 px | 质心最大匹配距离 |

### 3. 特征构建（`features/feature_builder_v2.py`）

输入归一化后的 (17, 2) 关键点，输出 **64 维**特征向量：

| 维度范围 | 内容 | 说明 |
|----------|---------|-------------|
| [0:51] | 17 × (x, y, conf) | 原始关键点位置 + 置信度 |
| [51:57] | 6 维原始几何 | 重心_y、躯干角、左/右膝角、脚距、身高 |
| [57:64] | 7 维增强几何 | 归一化重心_y、归一化脚距、质心 (x,y)、左/右髋角、躯干角 |

### 4. 序列缓冲（`sequence/sequence_buffer.py`）

- 每人独立 `deque(maxlen=30)` — 30 帧滑动窗口
- NaN 过滤与形状检查

### 5. 特征拼接

```
最终特征(30, 192) = [位置(30,64) | 速度(30,64) | 加速度(30,64)]
```

其中：
- `位置[t] = FeatureBuilderV2.build(关键点[t])`
- `速度[t] = 位置[t] - 位置[t-1]`
- `加速度[t] = 速度[t] - 速度[t-1]`

### 6. LSTM 多任务模型（`classifier/risk_classifier_v2.py`）

```
输入: (batch, 30, 192)
  ─► LSTM（2 层, 隐藏层=128, dropout=0.3）
  ─► 池化: 0.5 × 最后时刻输出 + 0.5 × 所有时刻均值

  ├──► 分类头: Linear(128→64)→ReLU→LayerNorm→Dropout→Linear(64→4) → 类别 logits
  ├──► 风险头: Linear(128→64)→ReLU→LayerNorm→Dropout→Linear(64→1)→Sigmoid → [0,1]
  └──► 时间头: Linear(128→64)→ReLU→LayerNorm→Dropout→Linear(64→1) → 时间值
```

### 7. 风险评估引擎（`pipeline/risk_engine_v2.py`）

**安全门**：
- `SITTING` 或 `NON_PERSON` → `danger_prob = 0.0`
- `FALLEN` → `danger_prob = max(model_risk, 0.7)`
- `SAFE` → `danger_prob = model_risk`

**处理流程**：
1. EMA 平滑：`平滑值 = α × 当前风险 + (1-α) × 上一帧`（α=0.4）
2. 连续计数：`平滑值 ≥ 0.35` → 计数器+1，否则重置
3. 状态机：计数器 ≥ 10 → DANGER（确认）；计数器开始 → WARNING
4. 锁定机制：DANGER 状态下连续 `lock_frames=15` 帧 SAFE 预测才解除

---

## 环境配置

### 安装

```bash
# 创建 conda 环境
conda create -n yolo python=3.9
conda activate yolo

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy scikit-learn scipy
```

核心依赖：

| 包 | 最低版本 | 用途 |
|---------|---------------|---------|
| Python | 3.9 | 运行环境 |
| PyTorch | 2.0+ | 深度学习框架（LSTM、混合精度训练） |
| Ultralytics | 8.0+ | YOLO26n-Pose 姿态估计 |
| OpenCV | 4.5+ | 视频输入输出、可视化 |
| NumPy | 1.21+ | 数值计算 |
| scikit-learn | 1.0+ | 评估指标、分类报告 |
| scipy | 1.7+ | 温度缩放优化 |

### Windows 注意事项

如果 Windows 用户名包含特殊字符（如 `y'y'x` 包含单引号），YOLO 的配置文件目录解析可能失败。详见 `SOLUTION.md` 中的解决方案（移动项目或设置 `YOLO_CONFIG_DIR` 环境变量）。

---

## 数据集

### 数据来源

| 数据集 | 场景 | 视频数 | 获取地址 | 说明 |
|---------|--------|--------|--------|-------------|
| **Le2i** | Coffee_room_01/02, Home_01/02 | ~130 | [Kaggle: falldataset-imvia](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) | 多场景跌倒视频，含逐帧标注 |
| **Le2i 正常** | Lecture_room, Office | ~28 | 同上（同一数据集内的未跌倒场景） | 确认无跌倒的正常视频 |
| **Pre-VFall** | — | 不定 | [Figshare v3（Pre-VFall.zip）](https://figshare.com/articles/dataset/Pre-VFall_Vision_Sensor_Simulated_Early_Signs_of_Fall_Dataset/26488216/3?file=48148606) | OpenPose 格式的预跌倒数据集 |

**Le2i 目录结构**（`build_dataset_v2.py` 的 `ARCHIVE` 需指向这一层）：

```
<ARCHIVE>/                          # 例: /Volumes/media-backup/AI-data/dataset/falldown/Le2i-v2
├── Coffee_room_01/Coffee_room_01/{Videos,Annotation_files}/
├── Coffee_room_02/Coffee_room_02/{Videos,Annotations_files}/   # 注意是 Annotations_files
├── Home_01/Home_01/{Videos,Annotation_files}/
├── Home_02/Home_02/{Videos,Annotation_files}/
├── Lecture_room/Lecture room/       # 无标注（正常场景）
└── Office/Office/                   # 无标注（正常场景）
```

实测该数据集：**190 个 `.avi`**（320×240, 25 FPS）、**130 份逐帧标注 `.txt`**（Coffee_room_01 48 / Coffee_room_02 22 / Home_01 30 / Home_02 30）、约 16 GB。视频文件名形如 `video (i).avi`，标注为 `video (i).txt`，第 1 行 `fall_start`、第 2 行 `fall_end`，其后每行 `帧号,status,...`。

**Pre-VFall 实况**：本项目使用 **v3**（`Pre-VFall.zip`，21,083,952,708 B，MD5 `45dc0cc5134ea3a9120b81271dd6ae5e`，约 19.6 GB，CC BY 4.0）：
- 下载页 <https://figshare.com/articles/dataset/Pre-VFall_Vision_Sensor_Simulated_Early_Signs_of_Fall_Dataset/26488216/3?file=48148606>
- 直链 <https://ndownloader.figshare.com/files/48148606>（支持 `curl -C -` 断点续传）
- DOI `10.6084/m9.figshare.26488216.v3`；最新版 v5 为 `Pre-VFall.rar`（21,096,978,752 B，MD5 `0308622d728063f8513367d823e07c2a`），二者内容同源，但 `.rar` 需额外解压工具

解压后**已含逐帧 OpenPose JSON**，无需自行生成。结构为 `<场景>/<受试者>/<受试者>-{Normal|Abnormal|Fall}-json/_<受试者>_<帧号>_keypoints.json`，实测 20 GB：

| 场景 | 受试者 | 段目录 | JSON 数 |
|---|---|---|---|
| confusion_delirium | 18 | 37 | 4,486 |
| confusion_nph | 18 | 36 | 4,262 |
| dizziness_fall_forward | 18 | 54 | 3,458 |
| dizziness_fall_side | 18 | 54 | 3,403 |
| weakness_fall_forward | 18 | 52 | 3,932 |
| weakness_fall_side | 18 | 54 | 3,018 |
| **合计** | **108** | **287** | **22,559** |

两点命名陷阱（脚本已兼容，勿用旧版 `f"{subject}-{segment}-json"` 硬拼）：
1. **`confusion_delirium` 整个场景用下划线**（`CLSM9_Normal-json`），其余 5 个场景用连字符；
2. `confusion_delirium/CLFM2/` 下混有一份 `CLFM1_Abnormal-json`（数据自身错位），脚本按"前缀须与受试者名一致"跳过它，避免错记；`CLFM1` 自身仍有该段落目录，不会丢数据。

`confusion_*` 两个场景无 Fall 段（36 个受试者），另有 42 个段落不足 30 帧，均会被跳过并打印明细。转换产物：**4888 个窗口**（risk=0.0 2908 / 0.5 1806 / 1.0 174），风险回归头的 `0.5` 档由此而来。

### 标签方案（V2）

| 类别 | ID | 风险值 | 时间值 | 条件 |
|-------|----|------|------|-----------|
| SAFE | 0 | 0.0 或指数衰减 | 999 或 time/PRE_FALL | 跌倒前或正常视频 |
| FALLEN | 1 | 1.0 | 0 | 跌倒结束后或跌倒过程中 |
| SITTING | 2 | 0.0 | 999 | 窗口内坐姿占比 ≥60% |
| NON_PERSON | 3 | 0.0 | 999 | 家具视频（Home_02 47, 57） |

**风险衰减**：对于跌倒前窗口（距跌倒 `PRE_FALL=120` 帧内）：
```
risk = exp(-剩余时间 / TAU),  TAU=50
```

### 构建数据集

```bash
# 0. 预提取关键点缓存（只需在首次、或更换视频/权重时执行）
python scripts/build/extract_keypoints.py

# V2 数据集（4 类 + 风险 + 时间）
python build_dataset_v2.py

# 5 类数据集（含 SITTING 类）
python build_dataset_5class.py

# Pre-VFall 转换（--data-dir 指向解压后的 Pre-VFall 根目录）
python scripts/build/build_prevfall_dataset.py --data-dir /path/to/Pre-VFall
```

---

## 模型训练

### V2 多任务训练

```bash
python train_v2.py
```

| 超参数 | 值 |
|---------------|-------|
| 训练轮数 | 50 |
| 批量大小 | 256 |
| 学习率 | 1e-3（AdamW） |
| 隐藏层大小 | 128 |
| LSTM 层数 | 2 |
| Dropout | 0.3 |
| 风险损失权重 λ\_risk | 0.30 |
| 时间损失权重 λ\_time | 0.10 |
| 早停耐心值 | 12 |
| 梯度裁剪 | 5.0 |

**损失函数**：
```
总损失 = 交叉熵(类别) + 0.30 × MSE(风险) + 0.10 × SmoothL1(时间)
```
- 时间损失**仅**在风险>0 且时间<1.0 的样本上计算。

**交叉验证**：3 折留一场景法（LOSO）：
- 第 1 折：Coffee_room_01/02 + PV 组 0
- 第 2 折：Home_01/02 + PV 组 1
- 第 3 折：Lecture_room/Office + PV 组 2

**数据增强**（仅训练时）：
- 坐标噪声：高斯噪声(0, 0.015) 施加于前 34 个坐标
- 时序丢弃：随机 1-3 帧做插值替换（30% 概率）
- 关键点遮挡：随机 1-3 个关键点遮挡 1-3 帧（20% 概率）

**训练后处理**：温度缩放（在区间 [0.1, 10.0] 内搜索最小化负对数似然的温度值）。

### 5 类训练

```bash
python train_5class.py
```

使用 Focal Loss（γ=2.0）+ 类别权重，代替多任务损失。交叉验证策略与 V2 相同。

---

## 推理检测

### 实时检测

```bash
# 摄像头模式
python realtime_detect.py

# 视频文件模式
python realtime_detect.py "路径/视频.avi"
```

脚本执行流程：
1. 打开视频文件或摄像头（默认加载 `lstm_v2.pt` + `norm_v2.npz` + `temperature_v2.npz`）
2. 逐帧运行完整流水线
3. 显示彩色状态边界框：
   - **绿色** → SAFE（安全）
   - **橙色** → WARNING（预警）
   - **红色** → DANGER（确认跌倒）

### 无人处理逻辑

- 复用上一帧特征填充缓冲区（防止速度突变）
- 若已确认 DANGER 且目标丢失 <60 帧 → 保持 DANGER 状态
- 若目标丢失 ≥60 帧 → 显示"Target lost"

---

## 验证测试

### 回归测试

```bash
# 跌倒视频必须触发 DANGER，正常视频必须保持 SAFE
python verify_real_videos.py

# P1（物品误识为人）+ P2（坐下误判为跌倒）分析
python verify_issues.py

# 坐下场景误报扫描
python scan_sit_videos.py
```

| 测试集 | 视频数 | 预期 | 状态 |
|----------|--------|----------|--------|
| 跌倒视频 | 6（Home_01×3, Coffee_01×1 等） | 跌倒时刻必须 DANGER | ✅ 通过 |
| 正常视频 | 4（Lecture, Office） | 全程 SAFE | ✅ 通过 |
| P1（家具） | 多场景随机采样 | 低报警率 | — |
| P2（坐下） | 坐下视频 | 无 DANGER | — |

### 交叉验证结果（V2，3 折 LOSO）

| 指标 | 值 |
|--------|-------|
| 分类准确率（校准后） | ~93% |
| 风险 MAE | ~0.07 |
| 风险阈值准确率（t=0.35） | ~95% |

---

## 模型清单

| 模型文件 | 类型 | 输入维度 | 输出 | 说明 |
|------------|------|-----------|--------|-------------|
| `lstm_v2.pt` | 多任务 | 192 | 类别(4)+风险+时间 | **V2 最终版**（全量数据训练） |
| `lstm_5class.pt` | 分类 | 192 | 类别(5) | 5 类模型（SAFE/WARNING/FALLEN/NON_PERSON/SITTING） |
| `lstm_4class.pt` | 分类 | 102 | 类别(4) | REFORM 遗留模型（SAFE/WARNING/FALLEN/NON_PERSON） |
| `lstm_multitask.pt` | 回归 | 102 | 风险+时间 | 原始基线模型 |
| `yolo26n-pose.pt` | 姿态（YOLO） | — | (17,2) 关键点 | Ultralytics nano 姿态模型 |

---

## 关键设计决策

### 为什么用 192 维特征而非原始 34 维关键点？

34 维（仅位置）无法区分"站着不动"和"正在跌倒但此刻姿态正常"。引入速度和加速度后可以捕捉运动特征——跌倒会产生突然的垂直速度尖峰和特征性加速度模式。

### 为什么用指数风险衰减（exp(-t/τ)）而非线性？

跌倒发生很快（25fps 下约 1-2 秒）。指数衰减赋予靠近跌倒时刻的帧更高权重，帮助模型更有效地区分"正常行走"和"即将跌倒"，效果优于线性衰减。

### 为什么用多任务（类别 + 风险 + 时间）？

- **分类头**：提供分类安全门（SITTING→风险=0，FALLEN→风险≥0.7）
- **风险头**：连续的跌倒概率，实现平滑、校准的风险评分
- **时间头**：预估紧迫程度，支持不同响应阈值

### 为什么用 4 类（SAFE/FALLEN/SITTING/NON_PERSON）而非 2 类？

早期 REFORM 版本使用 4 类（SAFE/WARNING/FALLEN/NON_PERSON）和 5 类（+SITTING）。V2 将 WARNING 简化为连续风险值，同时保留 SITTING 和 NON_PERSON 作为分类安全门。这样兼顾了分类的可解释性与回归的细腻度。

### 为什么需要温度缩放？

神经网络倾向于过度自信。温度缩放可以校准类别概率，不改变 argmax 决策结果，使概率输出对下游阈值判定更可靠。

---

## 性能指标

### V2 模型（Le2i + Pre-VFall）

| 指标 | 值 |
|--------|-------|
| 分类准确率（LOSO 交叉验证） | ~93% |
| 风险 MAE | ~0.07 |
| 风险阈值准确率（t=0.35） | ~95% |
| 跌倒视频检出率 | 100%（6/6） |
| 正常视频无报警率 | 100%（4/4） |

### 速度

| 组件 | 延迟 |
|-----------|---------|
| YOLO26n-Pose（GPU） | ~10-15ms |
| 特征提取 | <1ms |
| LSTM 推理（GPU） | ~2-5ms |
| **每帧总计** | **~15-25ms**（GPU 下 >40 FPS） |

---

## 已知限制

| 问题 | 当前状态 | 计划修复 |
|-------|--------|-------------|
| YOLO 对仰卧/倒地姿态检测失败 | 已缓解（conf=0.05） | 引入更多跌倒姿态训练数据 |
| 质心跟踪器 ID 漂移（人交叉时） | 待解决 | 集成 ByteTrack / DeepSORT |
| 镜子/反射面导致的家具误报 | 部分解决 | 增加 NON_PERSON 训练数据 |
| realtime_detect.py 单人假设 | 待解决 | 完整集成 PersonTracker |
| 无 API/部署（仅本地演示） | 待解决 | 封装为 FastAPI REST 服务 |
| 配置硬编码在 30+ 文件中 | 待解决 | 集中式 YAML 配置（见 `docs/refactoring-plan.md`） |

---

## 项目演进

| 阶段 | 时间 | 关键变更 |
|-------|------|-------------|
| **原始版（v1）** | — | YOLO-Pose + 质心跟踪 + 34 维特征 + LSTM 回归 |
| **Stage 0** | — | 阈值调优、姿态确认门、PENDING 状态、检测丢失修复 |
| **Stage 1** | — | 负样本扩充、Pre-VFall 集成、数据集构建重构 |
| **REFORM（v2.1）** | 2026-07-05 | 回归 → 4 分类；RiskEngine 495 行→80 行 |
| **5 类扩展** | 2026-07-06 | +SITTING 类，直接解决"坐下误判为跌倒" |
| **V2（当前）** | 2026-07 | 多任务：类别 + 风险 + 时间；192 维特征；LOSO 交叉验证；温度缩放 |

---

## 参考

### 数据集

- **Le2i Fall Detection Dataset**（IMVIA / 法国第戎大学 LE2I UMR6306）：
  下载 [Kaggle: tuyenldvn/falldataset-imvia](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)
  引用：
  > I. Charfi, J. Mitéran, J. Dubois, M. Atri, R. Tourki, "Optimised spatio-temporal descriptors for real-time fall detection: comparison of SVM and Adaboost based classification", *Journal of Electronic Imaging (JEI)*, Vol. 22, Issue 4, pp. 17, October 2013.

- **Pre-VFall: Vision Sensor Simulated Early Signs of Fall Dataset**（C. C. Olisah, X. Yang）：
  本项目使用 **v3**：下载页 <https://figshare.com/articles/dataset/Pre-VFall_Vision_Sensor_Simulated_Early_Signs_of_Fall_Dataset/26488216/3?file=48148606>，DOI `10.6084/m9.figshare.26488216.v3`，
  文件 `Pre-VFall.zip`（21,083,952,708 B，MD5 `45dc0cc5134ea3a9120b81271dd6ae5e`），许可 CC BY 4.0。
  > 引用时请写明实际使用的版本；最新版 v5 为 `Pre-VFall.rar`（21,096,978,752 B，MD5 `0308622d728063f8513367d823e07c2a`），内容同源。

### 代码

- **YOLO26**：Ultralytics YOLO26 - [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **PyTorch LSTM**：[https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

---

## 许可

内部研究项目。
