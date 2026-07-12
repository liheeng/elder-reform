import torch
import torch.nn as nn

class RiskClassifierV2(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128, num_layers=2,
                 dropout=0.3, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
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
        out, _ = self.lstm(x)
        pooled = 0.5 * out[:, -1, :] + 0.5 * out.mean(dim=1)
        class_logits = self.classifier(self.class_proj(pooled))
        risk = self.risk_head(self.risk_proj(pooled))
        time_pred = self.time_head(self.time_proj(pooled))
        return class_logits, risk.squeeze(-1), time_pred.squeeze(-1)
