import os
import numpy as np
import torch
import torch.nn as nn
from classifier.risk_classifier import RiskClassifier

CLASS_NAMES = ["SAFE", "WARNING", "FALLEN", "NON_PERSON", "SITTING"]

class LSTMClassifier:
    def __init__(self, model_path, norm_path, temperature_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RiskClassifier(input_dim=192, num_classes=5)
        sd = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()

        norm = np.load(norm_path)
        self.mean = torch.tensor(norm["mean"].astype(np.float32), device=self.device)
        self.std = torch.tensor(norm["std"].astype(np.float32), device=self.device) + 1e-8

        self.temperature = 1.0
        if temperature_path and os.path.exists(temperature_path):
            self.temperature = float(np.load(temperature_path)["T"])
        print(f"[Classifier] 5-class model loaded (T={self.temperature:.3f})")

    def normalize(self, sequence):
        return (sequence - self.mean.cpu().numpy()) / self.std.cpu().numpy()

    @torch.no_grad()
    def predict(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim == 2 and sequence.shape[1] == 34:
            T = sequence.shape[0]
            padded = np.zeros((T, 192), dtype=np.float32)
            for b in range(3):
                off = b * 64
                padded[:, off:off+34] = sequence[:, :34]
            sequence = padded
        elif sequence.shape[1] != 192:
            raise ValueError(f"Expected feature dim 192, got {sequence.shape[1]}")


        sequence = self.normalize(sequence)
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x) / self.temperature
        probs = torch.softmax(logits, dim=1).squeeze(0)
        class_id = int(probs.argmax().cpu())
        risk = float(1.0 - probs[0].cpu())
        if len(probs) >= 5:
            risk -= float(probs[4].cpu())
        risk = max(0.0, min(1.0, risk))
        return {
            "probs": {name: float(probs[i].cpu()) for i, name in enumerate(CLASS_NAMES)},
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "risk": risk,
        }

    @torch.no_grad()
    def predict_risk(self, sequence):
        return self.predict(sequence)["risk"]

    @torch.no_grad()
    def predict_label(self, sequence):
        return self.predict(sequence)["class_id"]
