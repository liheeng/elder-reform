import torch
import torch.nn as nn

class TCNEncoder(nn.Module):
    """膨胀一维卷积时序编码器，用来替换 LSTM 做受控消融。

    输入 x 为 (B, T, input_dim)：B 是批大小，T 是窗口帧数；
    输出为 (B, T, hidden_dim)，时间长度不变，每帧映射到 hidden_dim。
    进入 Conv1d 前会转成 (B, hidden_dim, T)，因为 Conv1d 把中间轴视为通道。

    设计约束是"与默认的 2×128 LSTM 同量级"，这样比较的才是结构而不是容量：
    隐藏维 128 一致、dropout 一致，参数量约 222k vs LSTM 约 295k。
    4 层膨胀 1/2/4/8 的感受野为 31 ≥ 窗口长度 30，整窗可见（非因果即可，
    因为滑窗式推理每帧都拿得到完整 30 帧）。
    """

    def __init__(self, input_dim=192, hidden_dim=128, num_layers=4,
                 kernel_size=3, dropout=0.3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数，否则等长 padding 不是整数")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(nn.ModuleDict({
                "conv": nn.Conv1d(
                    hidden_dim, hidden_dim, kernel_size,
                    # 每侧 (k-1)*d/2，保持时间长度不变；累加感受野 = 1 + Σ(k-1)d = 31 ≥ 30
                    padding=(kernel_size - 1) * dilation // 2, dilation=dilation,
                ),
                "norm": nn.LayerNorm(hidden_dim),
            }))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, T, input_dim) -> (B, T, hidden_dim)"""
        # 逐帧线性投影只改变特征宽度，然后换轴以匹配 Conv1d 的 (B, C, T) 输入格式。
        h = self.input_proj(x).transpose(1, 2)
        for blk in self.blocks:
            # 膨胀卷积沿时间轴提取不同跨度的变化，再用 GELU 引入非线性。
            res = nn.functional.gelu(blk["conv"](h))
            # LayerNorm 的最后一维是 hidden_dim，因此暂时转为 (B, T, C) 做归一化。
            # 归一化后转回卷积布局，供残差相加。
            res = blk["norm"](res.transpose(1, 2)).transpose(1, 2)
            # 仅对卷积分支做 dropout，再与块输入相加；残差保留已有表示并利于训练深层网络。
            h = h + self.dropout(res)
        # 还原成与 LSTM 相同的 (B, T, hidden_dim) 布局，便于后续统一池化。
        return h.transpose(1, 2)


# 池化方式；lastmean 是既有生产行为，其余为消融候选
POOLINGS = ("lastmean", "last", "mean", "max", "attn")

class AttentionPool(nn.Module):
    """按时间步学习权重再加权求和，用来对比固定池化。

    只多出 hidden_dim+1 个参数，因此与其它池化变体的容量几乎相同 ——
    消融要比较的是"聚合方式"，不是参数量。
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, out):
        """将 (B, T, hidden_dim) 的时序特征加权汇总为 (B, hidden_dim)。"""
        # 为每个时间步产生一个标量重要性分数，dim=1 表示在时间维归一化。
        weight = torch.softmax(self.score(out), dim=1)
        # 每帧向量乘以对应权重后沿时间维求和，得到每段序列的表示。
        return (weight * out).sum(dim=1)


class RiskClassifierV2(nn.Module):
    """对一段姿态特征序列联合预测类别、跌倒风险和距跌倒时间。

    输入形状为 (B, T, input_dim)；返回类别 logits (B, num_classes)、
    风险值 (B,) 和时间回归值 (B,)。
    """

    def __init__(self, input_dim=192, hidden_dim=128, num_layers=None,
                 dropout=0.3, num_classes=4, encoder="lstm", pooling="lastmean"):
        super().__init__()
        # 未指定层数时使用各编码器的默认深度：LSTM 2 层，TCN 4 块以覆盖 30 帧窗口。
        if num_layers is None:
            num_layers = 2 if encoder == "lstm" else 4
        self.encoder_name = encoder
        if encoder == "lstm":
            # batch_first 使输入/输出保持 (B, T, 特征)；多层 LSTM 之间应用 dropout。
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif encoder == "tcn":
            # TCN 也返回 (B, T, hidden_dim)，因此后续池化和任务头不依赖编码器类型。
            self.tcn = TCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"未知 encoder: {encoder!r}（可选 lstm / tcn）")

        # 池化把时间维 T 汇总掉；所有选项都输出 (B, hidden_dim)，保持任务头输入一致。
        if pooling not in POOLINGS:
            raise ValueError(f"未知 pooling: {pooling!r}（可选 {', '.join(POOLINGS)}）")
        self.pooling = pooling
        if pooling == "attn":
            # 只有注意力池化需要额外学习每个时间步的权重。
            self.attn = AttentionPool(hidden_dim)
        # 分类分支：hidden_dim -> 64 的特征变换，再输出每个类别的未归一化分数。
        self.class_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(64, num_classes)
        # 风险分支独立学习回归表示；Sigmoid 把单一风险预测限制在 [0, 1]。
        self.risk_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1)
        )
        self.risk_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        # 时间分支也独立学习表示；输出层不加激活，允许连续时间值自由回归。
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.15)
        )
        self.time_head = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 先对窗口中每一帧编码，得到保留时间轴的隐藏状态 (B, T, hidden_dim)。
        if self.encoder_name == "lstm":
            out, _ = self.lstm(x)
        else:
            out = self.tcn(x)

        # 将整段时序压缩成单个 (B, hidden_dim) 向量，再交给三个任务头。
        if self.pooling == "lastmean":       # 既有生产行为：末帧与均值各半
            # 同时保留窗口末端状态和全窗口概况，二者等权组合。
            pooled = 0.5 * out[:, -1, :] + 0.5 * out.mean(dim=1)
        elif self.pooling == "last":
            # 只使用窗口最后一帧的编码结果。
            pooled = out[:, -1, :]
        elif self.pooling == "mean":
            # 对所有时间步求平均，保留整个窗口的平均运动/姿态信息。
            pooled = out.mean(dim=1)
        elif self.pooling == "max":
            # 每个隐藏通道取时间维最大响应，突出窗口内曾出现的特征。
            pooled = out.max(dim=1).values
        else:                                # attn
            # 根据内容学习各时间步的贡献，再对时序特征加权汇总。
            pooled = self.attn(out)

        # 分类头输出 logits；softmax 通常由损失函数或推理端按需计算。
        class_logits = self.classifier(self.class_proj(pooled))
        # 风险头输出经 Sigmoid 约束的连续风险分数。
        risk = self.risk_head(self.risk_proj(pooled))
        # 时间头输出未经范围限制的回归值。
        time_pred = self.time_head(self.time_proj(pooled))
        # 移除最后的单元素输出轴；保留批次轴，批大小为 1 时形状仍为 (1,)。
        return class_logits, risk.squeeze(-1), time_pred.squeeze(-1)
