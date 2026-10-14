"""
train_5class.py

5-class LSTM training with:
- 4-fold Leave-One-Scene-Out cross-validation
- Focal Loss (gamma=2.0) with class weights
- Data augmentation (5 types, in-dataloader)
- Temperature scaling post-hoc calibration
- Gradient clipping, AdamW, ReduceLROnPlateau
- Final retrain on all data

Classes:
  0 = SAFE
  1 = WARNING
  2 = FALLEN
  3 = NON_PERSON
  4 = SITTING
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from classifier.risk_classifier import RiskClassifier
from utils.device import get_device, autocast_context, make_grad_scaler

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "processed")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = get_device()
SCALER = make_grad_scaler(DEVICE)

EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
NUM_CLASSES = 5
INPUT_DIM = 192
HIDDEN_SIZE = 128
NUM_LAYERS = 2
PATIENCE = 12

CLASS_NAMES = ["SAFE", "WARNING", "FALLEN", "NON_PERSON", "SITTING"]

# ==================================================
# Focal Loss
# ==================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight,
            reduction="none", label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean()

# ==================================================
# Dataset with Augmentation
# ==================================================
class FallDataset(Dataset):
    def __init__(self, X, y, aug=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.aug = aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.aug:
            x = self._augment(x)
        return x, self.y[i]

    def _augment(self, x):
        T, F = x.shape
        blk = 64
        # 1. Coordinate noise (68% of dims) �?保留，快速张量操�?        x[:, :34] += torch.randn(T, 34) * 0.015
        # 2. Temporal dropout (30%) �?保留，中等开销
        if T > 3 and torch.rand(1).item() > 0.3:
            nd = torch.randint(1, 4, (1,)).item()
            idxs = torch.randperm(T-2)[:nd] + 1
            for t in idxs:
                a = torch.rand(1).item()
                x[t] = x[t-1] * (1-a) + x[t+1] * a
        # 3. Keypoint occlusion (20%) �?保留，中等开销
        if torch.rand(1).item() > 0.8:
            nk = torch.randint(1, 4, (1,)).item()
            nf = torch.randint(1, 4, (1,)).item()
            for _ in range(nk):
                kidx = torch.randint(0, 17, (1,)).item()
                fstart = torch.randint(0, T-nf, (1,)).item()
                for off in [0, blk, blk*2]:
                    x[fstart:fstart+nf, off+kidx*3:off+kidx*3+2] = 0
        return x

# ==================================================
# Model
# ==================================================
LSTMModel5Class = RiskClassifier  # alias for backward compat

# ==================================================
# Train One Fold
# ==================================================
def train_fold(train_loader, val_loader, class_weights, fold_id):
    model = LSTMModel5Class().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = FocalLoss(gamma=2.0, weight=class_weights.to(DEVICE))

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast_context(DEVICE):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            SCALER.scale(loss).backward()
            SCALER.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            SCALER.step(optimizer)
            SCALER.update()
            train_loss += loss.item() * len(X_batch)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        all_logits, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)
                with autocast_context(DEVICE):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                all_logits.append(logits.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())
        val_loss /= n_val

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            preds = np.argmax(np.concatenate(all_logits, axis=0), axis=1)
            acc = np.mean(preds == np.array(all_true))
            print(f"  Fold {fold_id} Epoch {epoch:>3}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={acc*100:.2f}%  lr={optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= PATIENCE:
            print(f"  Fold {fold_id}: Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model

# ==================================================
# Main
# ==================================================
def main():
    print("=" * 60)
    print("5-Class LSTM Training with 4-fold LOSO CV")
    print("=" * 60)

    # ---- 1. Load Data ----
    print("\nLoading data...")
    X = np.load(os.path.join(PROCESSED_DIR, "x.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "class_labels_5class.npy"))
    vids = np.load(os.path.join(PROCESSED_DIR, "video_ids.npy"))
    print(f"  X: {X.shape}, y: {y.shape}, vids: {vids.shape}")

    # ---- 2. Normalize ----
    print("\nNormalizing...")
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - mean) / std
    np.savez(os.path.join(MODEL_DIR, "norm_5class.npz"),
             mean=mean.squeeze(), std=std.squeeze())
    print(f"  Norm params saved -> {MODEL_DIR}/norm_5class.npz")

    # ---- 3. Class Weights ----
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = np.where(counts == 0, 0.0, len(y) / (counts * NUM_CLASSES))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"\nClass distribution: {dict(zip(CLASS_NAMES, counts.tolist()))}")
    print(f"Class weights:       {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in weights]))}")

    # ---- 4. LOSO with merged scenes (3 fold) ----
    # �?PV (scene_id=10) �?video_id 分散�?3 个子场景
    for i in range(len(vids)):
        if vids[i, 0] == 10:
            vids[i, 0] = 10 + int(vids[i, 1]) % 3

    FOLD_GROUPS = [
        [0, 1, 10],     # Coffee_room_01 + Coffee_room_02 + PV group 0
        [2, 3, 11],     # Home_01 + Home_02 + PV group 1
        [4, 5, 12],     # Lecture_room + Office + PV group 2
    ]
    print(f"\nFold groups: {FOLD_GROUPS}")

    all_test_logits = []
    all_test_true = []

    for fold_idx, test_scenes in enumerate(FOLD_GROUPS):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(FOLD_GROUPS)}  Test Scenes = {test_scenes}")
        print(f"{'='*60}")

        test_mask = np.isin(vids[:, 0], test_scenes)
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        train_vids = len(np.unique(vids[train_mask], axis=0))
        test_vids = len(np.unique(vids[test_mask], axis=0))
        print(f"  Train: {X_train.shape[0]} samples, {train_vids} videos")
        print(f"  Test:  {X_test.shape[0]} samples, {test_vids} videos")

        train_dataset = FallDataset(X_train, y_train, aug=True)
        test_dataset = FallDataset(X_test, y_test, aug=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 pin_memory=True)

        model = train_fold(train_loader, test_loader, weights_tensor, fold_idx + 1)

        # Collect test logits
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                logits = model(X_batch)
                all_test_logits.append(logits.cpu().numpy())
                all_test_true.extend(y_batch.numpy())

        fold_path = os.path.join(MODEL_DIR, f"lstm_5class_fold{fold_idx+1}.pt")
        torch.save(model.state_dict(), fold_path)
        print(f"  Model saved -> {fold_path}")

    # ---- 5. Temperature Scaling ----
    print(f"\n{'='*60}")
    print("Temperature Scaling")
    print(f"{'='*60}")

    all_test_logits = np.concatenate(all_test_logits, axis=0)
    all_test_true = np.array(all_test_true)

    def nll(T):
        if T <= 0:
            return 1e10
        s = all_test_logits / T
        ps = np.exp(s - s.max(axis=1, keepdims=True))
        ps /= ps.sum(axis=1, keepdims=True)
        ps = np.clip(ps, 1e-10, 1 - 1e-10)
        return -np.mean([np.log(ps[i, int(all_test_true[i])])
                         for i in range(len(all_test_true))])

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    best_T = float(res.x)
    print(f"  Best temperature: {best_T:.4f}")

    np.savez(os.path.join(MODEL_DIR, "temperature_scaling.npz"), T=best_T)
    print(f"  Saved -> {MODEL_DIR}/temperature_scaling.npz")

    # ---- 6. Final Metrics ----
    print(f"\n{'='*60}")
    print("Final Metrics (all folds combined, calibrated)")
    print(f"{'='*60}")

    calibrated_logits = all_test_logits / best_T
    calibrated_preds = np.argmax(calibrated_logits, axis=1)

    print("\nClassification Report:")
    print(classification_report(
        all_test_true, calibrated_preds,
        target_names=CLASS_NAMES, digits=4,
        labels=list(range(NUM_CLASSES)),
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_test_true, calibrated_preds, labels=list(range(NUM_CLASSES)))
    header = "         " + "  ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i,j]:>8}" for j in range(NUM_CLASSES))
        print(f"{name:>10}: {row}")

    # ---- 7. Final Retrain on All Data ----
    print(f"\n{'='*60}")
    print("Final Retrain on All Data")
    print(f"{'='*60}")

    full_dataset = FallDataset(X, y, aug=True)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             pin_memory=True)

    model = LSTMModel5Class().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = FocalLoss(gamma=2.0, weight=weights_tensor.to(DEVICE))

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_full = len(full_dataset)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in full_loader:
            X_batch, y_batch = X_batch.to(DEVICE, non_blocking=True), y_batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast_context(DEVICE):
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
            SCALER.scale(loss).backward()
            SCALER.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            SCALER.step(optimizer)
            SCALER.update()
            train_loss += loss.item() * len(X_batch)
        train_loss /= n_full

        scheduler.step(train_loss)

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_val = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:>3}/{EPOCHS}  loss={train_loss:.4f}  lr={lr_val:.6f}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    final_path = os.path.join(MODEL_DIR, "lstm_5class.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved -> {final_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

