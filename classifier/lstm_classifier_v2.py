import json
import os
import numpy as np
from pathlib import Path

import torch
from classifier.risk_classifier_v2 import RiskClassifierV2
from utils.device import resolve_device

CLASS_NAMES = ["SAFE", "FALLEN", "SITTING", "NON_PERSON"]
PRE_FALL = 120  # 和 build_dataset_v2.py 保持一致

class LSTMClassifierV2:
    def __init__(self, model_path, norm_path, temperature_path=None, device=None,
                 encoder=None, pooling=None):
        self.device = resolve_device(device) if device else resolve_device("auto")
        state = torch.load(model_path, map_location=self.device)
        # 权重自描述：训练时写的 sidecar meta 优先，缺失时按 state_dict 键推断。
        # 之所以必须有 meta —— encoder="lstm" 且 pooling 非 attn 时，各变体的键名
        # 完全相同，只靠键是分不出用的是哪种池化的，会静默套错。
        meta = {}
        meta_path = Path(model_path).with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        if encoder is None:
            encoder = meta.get("encoder") or (
                "tcn" if any(k.startswith("tcn.") for k in state) else "lstm")
        if pooling is None:
            pooling = meta.get("pooling") or (
                "attn" if any(k.startswith("attn.") for k in state) else "lastmean")
        self.encoder = encoder
        self.pooling = pooling
        self.model = RiskClassifierV2(input_dim=192, num_classes=4,
                                      encoder=encoder, pooling=pooling)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        norm = np.load(norm_path)
        self.mean = torch.tensor(norm["mean"].astype(np.float32), device=self.device)
        self.std = torch.tensor(norm["std"].astype(np.float32), device=self.device) + 1e-8

        self.temperature = 1.0
        if temperature_path and os.path.exists(temperature_path):
            self.temperature = float(np.load(temperature_path)["T"])
        print(f"[ClassifierV2] 4-class+risk model loaded "
              f"(encoder={self.encoder}, pooling={self.pooling}, T={self.temperature:.3f})")

    def normalize(self, sequence):
        return (sequence - self.mean.cpu().numpy()) / self.std.cpu().numpy()

    @torch.no_grad()
    def predict(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.shape[1] != 192:
            raise ValueError(f"Expected feature dim 192, got {sequence.shape[1]}")

        sequence = self.normalize(sequence)
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        class_logits, risk, time_pred = self.model(x)

        class_logits = class_logits / self.temperature
        probs = torch.softmax(class_logits, dim=1).squeeze(0)
        class_id = int(probs.argmax().cpu())
        risk_val = float(risk.cpu())
        time_val = float(torch.relu(time_pred).cpu() * PRE_FALL)

        return {
            "probs": {name: float(probs[i].cpu()) for i, name in enumerate(CLASS_NAMES)},
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "risk": risk_val,
            "time_to_fall": time_val,
        }
