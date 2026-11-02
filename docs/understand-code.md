# V2 数据集构建与训练代码解读

本文说明 V2 流程中两个核心脚本的职责、数据流和关键规则：

1. [`build_dataset_v2.py`](../scripts/build/build_dataset_v2.py)：把 Le2i 和已转换的 Pre-VFall 数据整理成训练数组。
2. [`train_v2.py`](../scripts/train/train_v2.py)：读取这些数组，训练分类、风险和时间三个预测头，并保存模型。

## 一、整体数据流

```text
Le2i 关键点/置信度/检测框分数缓存 ─┐
                                  ├─ build_dataset_v2.py ─> processed/*_v2.npy
Pre-VFall 预处理数组 ──────────────┘                              │
                                                                 v
                                                    train_v2.py
                                                      │       │
                                             三折评估与校准   全量重训
                                                      │       │
                                                      └──> models/*
```

两步处理的对象都是**按时间排列的姿态特征窗口**。每帧有 192 维特征，每个样本包含连续 30 帧，因此模型输入的典型形状为 `(样本数, 30, 192)`，而不是一个样本只有 192 个数。

## 二、`build_dataset_v2.py`：构建训练数组

### 2.1 数据来源

- **Le2i 跌倒场景**：Coffee_room_01/02、Home_01/02。脚本读取 `yolo_keypoints/` 中预先生成的关键点、关节置信度和检测框 score 缓存，并解析相应的跌倒标注文件。
- **Le2i 正常场景**：Lecture_room 和 Office。只处理代码中列出的确认正常视频 ID，并通过姿态角度启发式把部分窗口标为坐姿。
- **Pre-VFall**：脚本不直接解析原始 OpenPose JSON；它只在 `processed/prevfall_*_v2.npy` 存在时把这些预处理结果拼进 Le2i 数据。原始 JSON 到 Pre-VFall 数组的转换由另一个构建脚本负责。

因此，若要把 Pre-VFall 纳入数据集，需要先运行 `scripts/build/build_prevfall_dataset.py` 生成对应的 `prevfall_*_v2.npy` 文件，再运行本脚本。Pre-VFall 的 Normal、Abnormal、Fall 片段标签分别为 `SAFE/风险 0.0/时间 999`、`SAFE/风险 0.5/时间 999`、`FALLEN/风险 1.0/时间 0`；其中 Abnormal 在分类头仍属于 SAFE，但给风险头提供 0.5 的监督。

### 2.2 特征如何形成

`build_window_features()` 对每帧关键点做归一化，再由 `FeatureBuilderV2` 提取 64 维姿态特征。连续帧差分得到速度和加速度，沿特征维拼接：

```text
每帧：64 维位置/姿态 + 64 维速度 + 64 维加速度 = 192 维
窗口：30 帧 × 192 维
```

Le2i 路径还会把检测框 score 写入对应特征维；Pre-VFall 来源没有框分数，score 及其有效标志代表“没有这路信号”，而不是低分。

窗口长度为 30，步长为 3。脚本中的 `subsample_to_10fps()` 名称容易误导：当前实现 `step = 1`，实际保留所有帧，没有降采样到 10 FPS；`TARGET_FPS` 也没有参与这个过程。因此窗口实际时长取决于输入缓存的帧率。

### 2.3 Le2i 标签规则

每个窗口得到三个监督目标：

- **类别**：`SAFE=0`、`FALLEN=1`、`SITTING=2`、`NON_PERSON=3`。
- **风险**：连续值，正常为 `0.0`，跌倒后为 `1.0`，跌倒前按距离跌倒的帧数指数增长。
- **时间**：跌倒相关窗口记录归一化后的剩余时间；非跌倒/普通安全窗口用 `999` 作为无时间标签的占位值。

跌倒视频按以下顺序判标签：

1. 窗口末帧达到或超过跌倒结束帧：`FALLEN / risk=1.0 / time=0`。
2. 窗口末帧在跌倒区间内，且逐帧标注属于代码识别的跌倒标签（3、4、7、8）：同样标为 `FALLEN / 1.0 / 0`。
3. 最近 10 帧中至少 60% 被姿态规则判为坐姿：标为 `SITTING / 0.0 / 999`。
4. 逐帧标注与坐姿姿态判断共同满足代码阈值时，也标为 `SITTING / 0.0 / 999`。
5. 其余有跌倒标注的视频，若窗口进入跌倒前 120 帧范围，则类别仍为 `SAFE`，风险为 `exp(-time_frames / 50)`，时间为 `time_frames / 120`；更早的窗口是 `SAFE / 0.0 / 999`。
6. 没有跌倒标注的正常视频为 `SAFE / 0.0 / 999`。

坐姿判断依据归一化姿态的躯干、膝和髋角度。它是启发式规则，不是人工坐姿标注模型。

### 2.4 输出文件

脚本把 Le2i 窗口和可用的 Pre-VFall 窗口合并后写入 `processed/`：

| 文件 | 内容 |
|---|---|
| `x_v2.npy` | 特征，形状通常为 `(N, 30, 192)` |
| `class_labels_v2.npy` | 每个窗口的类别，`int64` |
| `risk_labels_v2.npy` | 每个窗口的风险值，`float32` |
| `time_labels_v2.npy` | 每个窗口的时间标签，`float32` |
| `video_ids_v2.npy` | 每个窗口的 `[scene_id, video_id]`，用于按场景分组 |

注意：当前 builder 生成的 Le2i 类别定义虽然包含 `NON_PERSON`，但可见的 Le2i 标注规则主要产生 SAFE、FALLEN 和 SITTING；其他类别样本可能来自合并的数据源。

## 三、`train_v2.py`：训练与评估

### 3.1 加载、score 通道策略与标准化

`main()` 调用 `load_and_prepare_data()` 读取上表五个数组。默认关闭检测框 score 通道，将对应的 6 个位置/速度/加速度维度清零；`--score-channel` 可启用，但只有数据确实带有 0/1 有效标志时才允许继续。

随后按特征维计算全数据的均值和标准差，对所有帧标准化，并保存为 `models/norm_v2{后缀}.npz`，供推理使用。score 通道关闭时，其均值固定为 0、标准差固定为 1，避免被极小标准差放大。

### 3.2 模型输入和三个预测头

模型实现见 [`risk_classifier_v2.py`](../classifier/risk_classifier_v2.py)。默认使用 2 层 LSTM 编码 30 帧时序，每层隐藏维度为 128；默认 `lastmean` 池化把最后一帧输出与全窗口均值各取一半。也可通过参数选择 TCN 编码器或其他池化方式。

共享时序表示后接三个任务头：

1. **类别头**：输出 SAFE、FALLEN、SITTING、NON_PERSON 四类 logits。
2. **风险头**：Sigmoid 输出 `[0, 1]` 范围内的连续风险。
3. **时间头**：回归时间标签值；不带输出范围限制。

总损失为：

```text
分类交叉熵 + 0.30 × 风险 MSE + 0.10 × 时间 SmoothL1
```

分类交叉熵使用按类别频率反比计算的类别权重。时间损失只对 `risk > 0` 且 `time < 1.0` 的样本计算，因而 `999` 这类占位值以及 Pre-VFall Abnormal 的无时间标签不会被当作真实时间监督。

### 3.3 数据增强与单折训练

`FallDatasetV2` 只在训练集上启用增强：给部分特征加小幅随机噪声、偶尔用相邻帧插值替换帧，以及以较低概率模拟少数关节缺失。验证/留出数据不增强。

`train_fold()` 每折创建新模型，使用 AdamW、学习率调度和梯度裁剪；训练最多 50 轮，若验证损失连续 12 轮不改善则早停，并返回验证损失最低时的权重。每轮训练和验证分别由 `_run_train_epoch()`、`_run_validation_epoch()` 承担。

### 3.4 三折按场景交叉验证

`assign_cv_groups()` 把原始 `scene_id=10` 的 Pre-VFall 数据按受试者编号对 3 取模，分散到 10、11、12。Le2i 使用 0–5；ID 6–9 没有在当前映射中分配。三组留出场景为：

```text
[0, 1, 10]
[2, 3, 11]
[4, 5, 12]
```

`run_cross_validation()` 轮流留出每组，另外两组训练，收集每个样本的类别 logits 和风险预测。一个需要理解评估结果的注意事项：当前留出组同时作为 `train_fold()` 的验证集，用于学习率调度、早停和选取最佳权重，然后又用于该折指标；此外，标准化统计量也在全体数据上预先计算。因此这些折的指标不是完全独立的盲测估计。

### 3.5 温度校准、指标与全量训练

- `calibrate_temperature()` 用所有折的留出类别 logits 拟合单个温度参数，最小化分类负对数似然，只校准分类头，不作用于风险头。
- `report_class_metrics()` 输出分类报告和混淆矩阵；`report_risk_metrics()` 输出风险 MAE、MSE，以及阈值 `0.35` 的准确率。
- `train_full_dataset()` 再用全部数据和训练增强重训最终模型。由于这里没有独立验证集，它按训练损失调度并选择权重。

### 3.6 训练产物与常用参数

最终模型、标准化参数和温度参数写入 `models/`，另有每折模型和最终模型元信息：

| 文件模式 | 用途 |
|---|---|
| `lstm_v2{后缀}.pt` | 全量数据训练得到的最终模型权重 |
| `lstm_v2_fold{1..3}{后缀}.pt` | 三折模型权重 |
| `norm_v2{后缀}.npz` | 输入标准化均值和标准差 |
| `temperature_v2{后缀}.npz` | 分类头温度参数 |
| `lstm_v2{后缀}.meta.json` | 编码器、池化、种子、数据后缀及 score 通道配置 |

常用参数：

- `--suffix _nonperson`：读取带该后缀的数据文件，默认也给产物使用同一后缀。
- `--out-suffix _base`：仅改变产物后缀，避免覆盖默认模型文件。
- `--encoder lstm|tcn`、`--pooling ...`：选择时序结构和池化方式。
- `--seed N`：设置 Python、NumPy 和 PyTorch 随机种子；不指定时不固定。
- `--score-channel`：启用检测框 score 特征；默认关闭。

可在仓库根目录运行：

```bash
python scripts/build/build_dataset_v2.py
python scripts/train/train_v2.py
```

数据路径在构建脚本顶部配置，脚本不会自动下载或抽取原始视频/关键点；训练前需要相应缓存和数据文件已经存在。