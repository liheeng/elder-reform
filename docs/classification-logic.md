# 分类逻辑详解

> 本文档详细说明 V2 多任务模型当前使用的 4 类分类逻辑。

---

## 目录

- [一、分类体系](#一分類體系)
- [二、标签生成规则](#二標籤生成規則)
- [三、分类模型架构](#三分類模型架構)

---

## 一、分类体系

### 1.1 4 类定义

| 类 ID | 名称 | 含义 | 训练数据来源 |
|-------|------|------|-------------|
| 0 | SAFE | 正常行走 / 站立 | Le2i label=1 + Office/Lecture 正常视频 + 跌倒前超出 PRE_FALL 范围的帧 |
| 1 | FALLEN | 已跌倒 / 躺地 | Le2i label=3/4/7/8 + fall_end 之后 |
| 2 | SITTING | 坐下 | 自动姿态检测（关节角度规则） + Le2i 标注辅助 |
| 3 | NON_PERSON | 椅子/沙发等非人误检 | Home_02 家具视频（YOLO 将家具误检为人） |

### 1.2 WARNING 的表达方式

当前方案**没有 WARNING 独立类**。WARNING 级别的预警由回归头输出的**连续风险值**表达，配合 RiskEngine 状态机实现渐进式预警：

```
SAFE(类0) ─── risk 值连续上升 ───→ DANGER 状态
                ↑                        ↑
        回归头输出 risk∈[0,1]    RiskEngine 状态机判定
```

这种设计避免了"WARNING 类需要足够多标注数据"的难题，同时提供了比离散类更细腻的风险评估。

---

## 二、标签生成规则

### 2.1 数据来源

**Le2i 数据集**（[Kaggle: falldataset-imvia](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia)）标注格式：

```
第1行: fall_start   (跌倒开始帧号)
第2行: fall_end     (跌倒结束帧号)
第3行起: frame_id, status, bbox_x, bbox_y, bbox_w, bbox_h
```

Le2i 逐帧标注 status 的含义：
- `1` = 正常行走
- `2` = 跌倒开始（正在坠落）
- `3, 4, 7, 8` = 倒地 / 躺地

**其他来源**：
- Office/Lecture_room 确认无跌倒的视频 → 全部 SAFE（含坐姿自动检测生成 SITTING）
- Home_02 视频 47, 57（YOLO 持续误检为人的家具视频）→ NON_PERSON
- Pre-VFall 数据集（[Figshare v3](https://figshare.com/articles/dataset/Pre-VFall_Vision_Sensor_Simulated_Early_Signs_of_Fall_Dataset/26488216/3?file=48148606)，DOI `10.6084/m9.figshare.26488216.v3`，CC BY 4.0）合并提供 Fall / Abnormal / Normal 片段

### 2.2 标签生成入口 `build_dataset_v2.py:get_window_class_and_risk`

每条训练样本为一个 30 帧滑动窗口，标签生成按优先级依次判定：

```
输入: (start, end, fall_start, fall_end, frame_labels, raw_keypoints, subsample_indices)
输出: (class_label, risk_label, time_label)
```

以下规则按**优先级从高到低**依次检查，匹配即返回。

### 规则 1：已倒地

```python
if fall_start is not None and fall_end is not None:
    if last_orig_idx >= fall_end:
        return CLASS_FALLEN, RISK_FALL, 0.0
```

**条件**：窗口最后一帧对应的原始帧索引 ≥ fall_end  
**含义**：人已经完全倒地  
**class=FALLEN(1), risk=1.0, time=0**

### 规则 2：跌倒过程中被标注为倒地

```python
if fall_start is not None and fall_end is not None:
    if fall_start <= last_orig_idx < fall_end:
        ol = frame_labels.get(last_orig_idx)
        if ol is not None and ol in (3, 4, 7, 8):
            return CLASS_FALLEN, RISK_FALL, 0.0
```

**条件**：窗口末帧在 [fall_start, fall_end) 区间内，且该帧的 Le2i 逐帧标注为 3/4/7/8（倒地状态）  
**含义**：人在坠落过程中已被标注为"倒地"，视为已跌倒  
**class=FALLEN(1), risk=1.0, time=0**

### 规则 3：窗口末尾坐姿（后 10 帧检测）

```python
window_len = end - start
check_start = max(0, window_len - 10)
sitting_count = 0
check_frames = 0
for t_s in range(start + check_start, end):
    orig_idx = subsample_indices[t_s]
    if orig_idx < len(raw_keypoints) and is_sitting_posture_v2(raw_keypoints[orig_idx]):
        sitting_count += 1
    check_frames += 1
if check_frames > 0 and sitting_count >= check_frames * 0.6:
    return CLASS_SITTING, RISK_SAFE, NORMAL_TIME
```

**条件**：窗口最后 10 帧中 ≥60% 被 `is_sitting_posture_v2` 判定为坐姿  
**含义**：人在窗口末尾坐下  
**class=SITTING(2), risk=0.0, time=999**

### 规则 4：窗口内逐帧标注辅助判断

```python
if frame_labels:
    sitting_count = 0
    for t_s in range(start, end):
        orig_idx = subsample_indices[t_s]
        ol = frame_labels.get(orig_idx)
        if ol is not None and ol in (3, 4, 7, 8):
            if orig_idx < len(raw_keypoints) and is_sitting_posture_v2(raw_keypoints[orig_idx]):
                sitting_count += 1
    if sitting_count >= (end - start) * 0.5:
        return CLASS_SITTING, RISK_SAFE, NORMAL_TIME
```

**条件**：Le2i 标注 + 姿态检测联合确认坐姿帧占比 ≥50%  
**用途**：规则 3 的补充，覆盖坐姿分布在整个窗口而非仅末尾的情况  
**class=SITTING(2), risk=0.0, time=999**

### 规则 5：跌倒前预警窗口（指数风险衰减）

```python
if fall_start is not None and fall_end is not None:
    if last_orig_idx >= fall_end:
        return CLASS_FALLEN, RISK_FALL, 0.0
    elif last_orig_idx >= fall_start - PRE_FALL:
        time_frames = max(0, fall_end - 1 - last_orig_idx)
        risk = float(np.exp(-time_frames / TAU))
        return CLASS_SAFE, risk, float(time_frames) / PRE_FALL
    else:
        return CLASS_SAFE, RISK_SAFE, NORMAL_TIME
```

**条件**：窗口末帧在 [fall_start - PRE_FALL, fall_end - 1] 区间内  
**含义**：距离跌倒还有 PRE_FALL=120 帧以内，风险逐渐升高  
**class=SAFE(0), risk=exp(-time_frames/50)**

```
risk = exp(-time_frames / TAU),  TAU=50
```

| 距跌倒完成帧数 | 风险值 |
|---|---|
| 0（刚跌倒） | 1.000 |
| 10 帧前 | 0.819 |
| 30 帧前 | 0.549 |
| 50 帧前 | 0.368 |
| 120 帧前（PRE_FALL 边界） | 0.091 |

### 规则 6：正常状态（兜底）

```python
return CLASS_SAFE, RISK_SAFE, NORMAL_TIME
```

**条件**：以上规则均不匹配  
**class=SAFE(0), risk=0.0, time=999**

### 2.3 Pre-VFall 数据集处理

Pre-VFall 数据集（[Figshare v3](https://figshare.com/articles/dataset/Pre-VFall_Vision_Sensor_Simulated_Early_Signs_of_Fall_Dataset/26488216/3?file=48148606)，DOI `10.6084/m9.figshare.26488216.v3`，CC BY 4.0）**解压后已含逐帧 OpenPose JSON**，无需自行生成。每个片段以独立目录存放，需要通过格式转换和分段标注才能合并到 V2 训练集。

> **命名陷阱**：段落目录有两种分隔符 —— `confusion_delirium` 用下划线（`CLSM9_Normal-json`），其余 5 个场景用连字符（`FWSFM1-Fall-json`）。另有 `confusion_delirium/CLFM2/CLFM1_Abnormal-json` 一处数据自身错位，需按前缀一致性跳过。

**OpenPose → COCO 关键点转换**：

```python
OPENPOSE_TO_COCO = {
    0: 0,    # Nose → Nose
    16: 1,   # LEye → LEye
    15: 2,   # REye → REye
    18: 3,   # LEar → LEar
    17: 4,   # REar → REar
    5: 5,    # LShoulder → LShoulder
    2: 6,    # RShoulder → RShoulder
    6: 7,    # LElbow → LElbow
    3: 8,    # RElbow → RElbow
    7: 9,    # LWrist → LWrist
    4: 10,   # RWrist → RWrist
    12: 11,  # LHip → LHip
    9: 12,   # RHip → RHip
    13: 13,  # LKnee → LKnee
    10: 14,  # RKnee → RKnee
    14: 15,  # LAnkle → LAnkle
    11: 16,  # RAnkle → RAnkle
}

def convert_openpose_to_coco(pose_keypoints_2d):
    # OpenPose 存储格式: [x0, y0, c0, x1, y1, c1, ...]，每个关键点 3 维
    kpts = np.zeros((17, 2), dtype=np.float32)
    confs = np.zeros(17, dtype=np.float32)
    for op_idx, coco_idx in OPENPOSE_TO_COCO.items():
        base = op_idx * 3
        kpts[coco_idx] = [pose_keypoints_2d[base], pose_keypoints_2d[base+1]]
        confs[coco_idx] = pose_keypoints_2d[base+2]
    return kpts, confs
```

**三类片段的分类与风险标签**：

```python
# build_prevfall_dataset.py 中的标签定义
CLASS_SAFE   = 0
CLASS_FALLEN = 1

RISK_NORMAL    = 0.0    # Normal 片段 → 安全步行
RISK_ABNORMAL  = 0.5    # Abnormal 片段 → 异常步态，高跌倒风险
RISK_FALL      = 1.0    # Fall 片段 → 已跌倒
NORMAL_TIME    = 999     # 时间占位符
```

**三类片段的标签分配逻辑**：

| 片段类型 | 类标签 | 风险值 | 时间值 | 说明 |
|-----------|---------|---------|-------|------|
| Normal | SAFE(0) | 0.0 | 999 | 正常行走，无风险 |
| Abnormal | SAFE(0) | 0.5 | 999 | 异常步态但未跌倒，类标签仍为 SAFE，用风险值表达预警 |
| Fall | FALLEN(1) | 1.0 | 0 | 已跌倒 |

Pre-VFall 不产生 SITTING 和 NON_PERSON 类。Abnormal 片段的特殊之处在于**类标为 SAFE 但风险值为 0.5**，这使得模型在分类头上看到的是 SAFE，但回归头需要输出中等风险值，从而学会在 SAFE 类内部区分正常和异常步态。

**合并到 V2 数据集**：

```python
# build_dataset_v2.py main() 末尾
if os.path.exists(pv_x_path):
    pv_x = np.load("prevfall_x_v2.npy")         # (N, 30, 192)
    pv_class = np.load("prevfall_class_v2.npy")  # (N,)  0=SAFE, 1=FALLEN
    pv_risk = np.load("prevfall_risk_v2.npy")    # (N,)  0.0/0.5/1.0
    pv_time = np.load("prevfall_time_v2.npy")    # (N,)  999/999/0
    pv_ids = np.load("prevfall_ids_v2.npy")      # (N, 2) scene_id, subject_id

    # 直接追加到 Le2i 采样后得到的列表中
    all_x.extend([pv_x[i] for i in range(len(pv_x))])
    all_classes.extend(pv_class.tolist())
    all_risks.extend(pv_risk.tolist())
    all_times.extend(pv_time.tolist())
    video_ids_list.extend(pv_ids.tolist())
```

### 2.4 NON_PERSON 样本提取

NON_PERSON 类不从标注视频生成，而是从已知的家具误检视频中自动提取：

- **源视频**：Home_02 场景中视频 47、57（YOLO 持续将沙发/椅子检测为人）
- **提取流程**：YOLO 提取关键点 → 特征构建 → 滑动窗口 → 全部标为 NON_PERSON
- **样本量**：约 95 个窗口，每个 (30, 192)

这些样本没有对应的真实人体，关键点分布与真人完全不同（集中、低置信度、几乎无运动），模型通过对比学习学会区分"假人"。

### 坐姿判定函数 `is_sitting_posture_v2`

检测流程：

```python
def is_sitting_posture_v2(keypoints):
    # 1. 归一化：以髋中心为原点，缩放到目标身高 100
    nk, _ = normalize_keypoints(keypoints, np.ones(17))
    
    # 2. 构建特征获取关节角度
    feat = FeatureBuilderV2.build(nk, np.ones(17))
    躯干角 = feat[52]             # 肩-髋连线与垂直方向的夹角
    左膝角, 右膝角 = feat[53], feat[54]
    左髋角, 右髋角 = feat[61], feat[62]
    平均膝角 = (左膝角 + 右膝角) / 2
    平均髋角 = (左髋角 + 右髋角) / 2
    
    # 3. 判定
    if 躯干角 >= 25.0:   return False   # 排除横躺（横躺 70-90°）
    if 平均膝角 >= 130.0: return False   # 排除站立（站立时膝角接近 180°）
    if 平均髋角 >= 140.0: return False   # 排除站立
    if 平均膝角 <= 30.0 and 平均髋角 <= 30.0: return False  # 排除蜷缩/不可见
    return True
```

**几何意义**：

| 姿态 | 躯干角 | 平均膝角 | 平均髋角 |
|------|--------|---------|---------|
| 站立 | <10° | ~170° | ~170° |
| 坐下 | <25° | ~90° | ~90° |
| 横躺 | ~70-90° | 不定 | 不定 |
| 蜷缩 | <25° | <30° | <30° |

### 正常视频的坐姿标注

对于 Office/Lecture_room 等确认无跌倒的正常视频，坐姿判定简化为：

```python
后10帧坐姿 > 一半? → SITTING : SAFE
```

---

## 三、分类模型架构

### 3.1 `RiskClassifierV2` 分类头

完整模型 `classifier/risk_classifier_v2.py` 有三个并行预测头，分类头是其中之一：

```
输入: (batch, 30, 192)
  │
  ▼
LSTM (2层, hidden=128, dropout=0.3, batch_first=True)
  │  输出: (batch, 30, 128)
  │
  ▼
池化: pooled = 0.5 × out[:, -1, :] + 0.5 × out.mean(dim=1)
  │  输出: (batch, 128)
  │
  ├──►  分类投影: Linear(128→64) → ReLU → LayerNorm → Dropout(0.2)
  │         └──►  分类头: Linear(64→4) → class_logits (batch, 4)
  │
  ├──►  风险投影: Linear(128→64) → ReLU → LayerNorm → Dropout(0.1)
  │         └──►  风险头: Linear(64→1) → Sigmoid → risk ∈ [0, 1]
  │
  └──►  时间投影: Linear(128→64) → ReLU → LayerNorm → Dropout(0.15)
            └──►  时间头: Linear(64→1) → time_pred
```

