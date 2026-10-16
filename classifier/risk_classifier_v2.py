import torch
import torch.nn as nn

class TCNEncoder(nn.Module):
    """膨胀一维卷积时序编码器，用来替换 LSTM 做受控消融。

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
        h = self.input_proj(x).transpose(1, 2)
        for blk in self.blocks:
            res = nn.functional.gelu(blk["conv"](h))
            res = blk["norm"](res.transpose(1, 2)).transpose(1, 2)
            h = h + self.dropout(res)
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
        weight = torch.softmax(self.score(out), dim=1)
        return (weight * out).sum(dim=1)


class RiskClassifierV2(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128, num_layers=None,
                 dropout=0.3, num_classes=4, encoder="lstm", pooling="lastmean"):
        super().__init__()
        # num_layers 按编码器取各自默认：LSTM 2 层，TCN 4 层（感受野需覆盖 30 帧）
        if num_layers is None:
            num_layers = 2 if encoder == "lstm" else 4
        self.encoder_name = encoder
        if encoder == "lstm":
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif encoder == "tcn":
            self.tcn = TCNEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"未知 encoder: {encoder!r}（可选 lstm / tcn）")

        # 池化：所有变体的输出维度都保持 hidden_dim，保证头部的参数量一致
        if pooling not in POOLINGS:
            raise ValueError(f"未知 pooling: {pooling!r}（可选 {', '.join(POOLINGS)}）")
        self.pooling = pooling
        if pooling == "attn":
            self.attn = AttentionPool(hidden_dim)
        # 分类头：128 → 64 → 4
        self.class_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(64, num_classes)
        # 回归头：128 → 64 → 1（Sigmoid 约束到 [0,1]）
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
        # 时间回归头：128 → 64 → 1（无激活，连续值）
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
        if self.encoder_name == "lstm":
            out, _ = self.lstm(x)
        else:
            out = self.tcn(x)
        if self.pooling == "lastmean":       # 既有生产行为：末帧与均值各半
            pooled = 0.5 * out[:, -1, :] + 0.5 * out.mean(dim=1)
        elif self.pooling == "last":
            pooled = out[:, -1, :]
        elif self.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.pooling == "max":
            pooled = out.max(dim=1).values
        else:                                # attn
            pooled = self.attn(out)
        class_logits = self.classifier(self.class_proj(pooled))
        risk = self.risk_head(self.risk_proj(pooled))
        time_pred = self.time_head(self.time_proj(pooled))
        return class_logits, risk.squeeze(-1), time_pred.squeeze(-1)
