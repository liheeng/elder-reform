import torch
import torch.nn as nn

class RiskClassifier(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=128,
                 num_layers=2, dropout=0.3, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x, return_logits=True):
        out, _ = self.lstm(x)
        pooled = 0.5 * out[:, -1, :] + 0.5 * out.mean(dim=1)
        feat = self.feature_proj(pooled)
        logits = self.classifier(feat)
        if return_logits:
            return logits
        return torch.softmax(logits, dim=1)
