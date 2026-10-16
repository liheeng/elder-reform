"""
train_v2.py

4-class + risk + time multi-task training:
  - 4 classes: SAFE / FALLEN / SITTING / NON_PERSON
  - 1 continuous risk value: trained with MSELoss (exponential labels)
  - 1 time-to-fall value: trained with SmoothL1Loss (fall samples only)
  - Multi-task loss: CrossEntropy + lambda_risk * MSE + lambda_time * SmoothL1
  - 3-fold Leave-One-Scene-Out cross-validation
  - Temperature scaling for class head only
  - Final retrain on all data

用法:
    python scripts/train/train_v2.py                     # 用生产数据与路径
    python scripts/train/train_v2.py --suffix _nonperson  # 用 x_v2_nonperson.npy 等，产物同后缀
    python scripts/train/train_v2.py --out-suffix _base  # 用生产数据训练但产物另存（不覆盖生产模型）
"""

import os
import argparse
import json
import sys
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from classifier.risk_classifier_v2 import RiskClassifierV2
from features.feature_builder_v2 import (
    HAS_SCORE_DIM,
    SCORE_DIM,
    score_channel_window_dims,
)
from utils.device import get_device, autocast_context, make_grad_scaler

PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "processed")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = get_device()
SCALER = make_grad_scaler(DEVICE)

EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
NUM_CLASSES = 4
INPUT_DIM = 192
HIDDEN_SIZE = 128
NUM_LAYERS = 2
PATIENCE = 12
LAMBDA_RISK = 0.30
LAMBDA_TIME = 0.10

CLASS_NAMES = ["SAFE", "FALLEN", "SITTING", "NON_PERSON"]


# ==================================================
# Multi-task Loss
# ==================================================
class MultiTaskLoss(nn.Module):
    def __init__(self, class_weights, lambda_risk=0.30, lambda_time=0.10):
        super().__init__()
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.risk_loss = nn.MSELoss()
        self.time_loss = nn.SmoothL1Loss()
        self.lambda_risk = lambda_risk
        self.lambda_time = lambda_time

    def forward(self, class_logits, risk_pred, time_pred,
                class_target, risk_target, time_target):
        loss_cls = self.class_loss(class_logits, class_target)
        loss_risk = self.risk_loss(risk_pred, risk_target)
        # time 损失只在 risk>0 �?time<1.0 的样本上计算
        # (time>=1.0 �?NORMAL_TIME 占位符或 Pre-VFall Abnormal 无时间标�?
        valid_time = (risk_target > 0) & (time_target < 1.0)
        if valid_time.sum() > 0:
            loss_time = self.time_loss(time_pred[valid_time], time_target[valid_time])
        else:
            loss_time = torch.tensor(0.0, device=loss_cls.device)
        total = loss_cls + self.lambda_risk * loss_risk + self.lambda_time * loss_time
        return total, loss_cls, loss_risk, loss_time


# ==================================================
# Dataset with Augmentation
# ==================================================
class FallDatasetV2(Dataset):
    def __init__(self, X, y_class, y_risk, y_time, aug=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_risk = torch.tensor(y_risk, dtype=torch.float32)
        self.y_time = torch.tensor(y_time, dtype=torch.float32)
        self.aug = aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.aug:
            x = self._augment(x)
        return x, self.y_class[i], self.y_risk[i], self.y_time[i]

    def _augment(self, x):
        T, F = x.shape
        blk = 64
        x[:, :34] += torch.randn(T, 34) * 0.015
        if T > 3 and torch.rand(1).item() > 0.3:
            nd = torch.randint(1, 4, (1,)).item()
            idxs = torch.randperm(T-2)[:nd] + 1
            for t in idxs:
                a = torch.rand(1).item()
                x[t] = x[t-1] * (1-a) + x[t+1] * a
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
# Train One Fold
# ==================================================
def train_fold(train_loader, val_loader, class_weights, fold_id, encoder="lstm",
               pooling="lastmean"):
    model = RiskClassifierV2(encoder=encoder, pooling=pooling).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = MultiTaskLoss(class_weights.to(DEVICE), LAMBDA_RISK, LAMBDA_TIME)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_risk_loss = 0.0
        train_time_loss = 0.0
        for X_batch, y_cls, y_risk, y_time in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_cls = y_cls.to(DEVICE, non_blocking=True)
            y_risk = y_risk.to(DEVICE, non_blocking=True)
            y_time = y_time.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast_context(DEVICE):
                class_logits, risk_pred, time_pred = model(X_batch)
                loss, loss_cls, loss_risk, loss_time = criterion(
                    class_logits, risk_pred, time_pred,
                    y_cls, y_risk, y_time)
            SCALER.scale(loss).backward()
            SCALER.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            SCALER.step(optimizer)
            SCALER.update()
            train_loss += loss.item() * len(X_batch)
            train_cls_loss += loss_cls.item() * len(X_batch)
            train_risk_loss += loss_risk.item() * len(X_batch)
            train_time_loss += loss_time.item() * len(X_batch)
        train_loss /= n_train
        train_cls_loss /= n_train
        train_risk_loss /= n_train
        train_time_loss /= n_train

        model.eval()
        val_loss = 0.0
        all_logits, all_true, all_risks_pred, all_risks_true = [], [], [], []
        with torch.no_grad():
            for X_batch, y_cls, y_risk, y_time in val_loader:
                X_batch = X_batch.to(DEVICE, non_blocking=True)
                y_cls = y_cls.to(DEVICE, non_blocking=True)
                y_risk = y_risk.to(DEVICE, non_blocking=True)
                y_time = y_time.to(DEVICE, non_blocking=True)
                with autocast_context(DEVICE):
                    class_logits, risk_pred, time_pred = model(X_batch)
                    loss, _, _, _ = criterion(
                        class_logits, risk_pred, time_pred,
                        y_cls, y_risk, y_time)
                val_loss += loss.item() * len(X_batch)
                all_logits.append(class_logits.cpu().numpy())
                all_true.extend(y_cls.cpu().numpy())
                all_risks_pred.extend(risk_pred.cpu().numpy())
                all_risks_true.extend(y_risk.cpu().numpy())
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
            risk_mae = np.mean(np.abs(np.array(all_risks_pred) - np.array(all_risks_true)))
            print(f"  Fold {fold_id} Epoch {epoch:>3}/{EPOCHS}  "
                  f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"cls={train_cls_loss:.4f}  risk={train_risk_loss:.4f}  "
                  f"time={train_time_loss:.4f}  "
                  f"val_acc={acc*100:.2f}%  risk_mae={risk_mae:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= PATIENCE:
            print(f"  Fold {fold_id}: Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


# ==================================================
# Main
# ==================================================
def main():
    ap = argparse.ArgumentParser(description="v2 4 类 + risk 训练（3 折 LOSO）")
    ap.add_argument("--suffix", default="",
                    help="数据与产物文件名后缀；默认空=使用生产文件与路径")
    ap.add_argument("--out-suffix", default=None,
                    help="仅产物后缀；默认与 --suffix 一致。"
                         "用于「用生产数据训练但写到别的文件名」，避免覆盖生产模型")
    ap.add_argument("--encoder", default="lstm", choices=["lstm", "tcn"],
                    help="时序编码器；用于受控消融，默认 lstm 保持原行为")
    ap.add_argument("--pooling", default="lastmean",
                    choices=["lastmean", "last", "mean", "max", "attn"],
                    help="时序池化；默认 lastmean 保持原行为")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子；默认 None=不固定（保持既有行为）。"
                         "对比实验请显式指定，否则跑两次结果有差异")
    ap.add_argument("--score-channel", action="store_true",
                    help="启用检测框 score 那一路特征（dims 51/57 及其 vel/acc 共 6 维）。"
                         "默认关闭 —— 2026-10-16 的受控消融显示它会把 held-out 非人"
                         "判定率拉低约 7 个点，见 docs/tuning-log.md。"
                         "此开关仅用于复现该实验")
    args = ap.parse_args()
    sfx = args.suffix
    out_sfx = sfx if args.out_suffix is None else args.out_suffix

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print("=" * 60)
    print("V2 4-Class + Risk Training with 3-fold LOSO CV")
    print("=" * 60)
    print(f"时序编码器: {args.encoder}   池化: {args.pooling}")
    if sfx:
        print(f"文件名后缀: {sfx}")

    # ---- 1. Load Data ----
    print("\nLoading data...")
    X = np.load(os.path.join(PROCESSED_DIR, f"x_v2{sfx}.npy"))
    y_class = np.load(os.path.join(PROCESSED_DIR, f"class_labels_v2{sfx}.npy"))
    y_risk = np.load(os.path.join(PROCESSED_DIR, f"risk_labels_v2{sfx}.npy"))
    y_time = np.load(os.path.join(PROCESSED_DIR, f"time_labels_v2{sfx}.npy"))
    vids = np.load(os.path.join(PROCESSED_DIR, f"video_ids_v2{sfx}.npy"))
    print(f"  X: {X.shape}, y_class: {y_class.shape}, y_risk: {y_risk.shape}, y_time: {y_time.shape}, vids: {vids.shape}")

    # ---- 1b. 检测框 score 通道 ----
    # 51/57 是原地复用的废维（见 features/feature_builder_v2.py）。判断"这份数据
    # 到底带不带 score 通道"，用 57 是不是严格 0/1 掩码最可靠 —— 只看 std 会被骗：
    # 旧数据集这两维的 std 也有 14，全靠极少数回退帧塞进去的原始像素撑着。
    score_dims = score_channel_window_dims()
    mask_dim = X[:, :, HAS_SCORE_DIM]
    mask_is_binary = bool(((mask_dim == 0.0) | (mask_dim == 1.0)).all())
    if not args.score_channel:
        # 默认路径：pos / vel / acc 六维必须一起清（见 score_channel_window_dims 的说明）
        X[:, :, list(score_dims)] = 0.0
        uses_score = False
        print(f"  score 通道: 关闭（默认；{len(score_dims)} 个维已置零: {list(score_dims)}）")
    else:
        if not (mask_is_binary and bool((mask_dim == 1.0).any())):
            raise SystemExit(
                "[中止] dim %d 不是 0/1 掩码，这份数据集没有 score 通道。\n"
                "  先重抽缓存并重建数据集（见 docs/tuning-log.md「如何复现」第 9a 条）；\n"
                "  或去掉 --score-channel —— 默认就是关闭的。" % HAS_SCORE_DIM)
        uses_score = True
        print(f"  score 通道: 启用  dim{SCORE_DIM} 均值 {X[:, :, SCORE_DIM].mean():.4f} "
              f"± {X[:, :, SCORE_DIM].std():.4f}，"
              f"掩码为 1 的占比 {(mask_dim == 1.0).mean():.3f}")

    # ---- 2. Normalize ----
    print("\nNormalizing...")
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    if not uses_score:
        # 对照臂这 6 维在训练数据里被精确置 0，std 会塌到 1e-8。而推理侧算出来
        # 的是 ±5e-6 的浮点噪声，除以 1e-8 就是 ±500 的离群输入 —— 模型会恒报
        # DANGER，且看不出哪里错了。钉成 mean=0 / std=1 把这 6 维原样摘出去。
        for _d in score_dims:
            mean[0, 0, _d] = 0.0
            std[0, 0, _d] = 1.0
    X = (X - mean) / std
    np.savez(os.path.join(MODEL_DIR, f"norm_v2{out_sfx}.npz"),
             mean=mean.squeeze(), std=std.squeeze())
    print(f"  Norm params saved -> {MODEL_DIR}/norm_v2{out_sfx}.npz")

    # ---- 3. Class Weights ----
    counts = np.bincount(y_class, minlength=NUM_CLASSES).astype(np.float32)
    weights = np.where(counts == 0, 0.0, len(y_class) / (counts * NUM_CLASSES))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"\nClass distribution: {dict(zip(CLASS_NAMES, counts.tolist()))}")
    print(f"Class weights:       {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in weights]))}")

    # ---- 4. LOSO with merged scenes (3 fold) ----
    for i in range(len(vids)):
        if vids[i, 0] == 10:
            vids[i, 0] = 10 + int(vids[i, 1]) % 3

    FOLD_GROUPS = [
        [0, 1, 10],
        [2, 3, 11],
        [4, 5, 12],
    ]
    print(f"\nFold groups: {FOLD_GROUPS}")

    all_test_cls_logits = []
    all_test_risk_preds = []
    all_test_true_class = []
    all_test_true_risk = []

    for fold_idx, test_scenes in enumerate(FOLD_GROUPS):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(FOLD_GROUPS)}  Test Scenes = {test_scenes}")
        print(f"{'='*60}")

        test_mask = np.isin(vids[:, 0], test_scenes)
        train_mask = ~test_mask

        X_train, yc_train = X[train_mask], y_class[train_mask]
        yr_train, yt_train = y_risk[train_mask], y_time[train_mask]
        X_test, yc_test = X[test_mask], y_class[test_mask]
        yr_test, yt_test = y_risk[test_mask], y_time[test_mask]

        train_vids = len(np.unique(vids[train_mask], axis=0))
        test_vids = len(np.unique(vids[test_mask], axis=0))
        print(f"  Train: {X_train.shape[0]} samples, {train_vids} videos")
        print(f"  Test:  {X_test.shape[0]} samples, {test_vids} videos")

        train_dataset = FallDatasetV2(X_train, yc_train, yr_train, yt_train, aug=True)
        test_dataset = FallDatasetV2(X_test, yc_test, yr_test, yt_test, aug=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 pin_memory=True)

        model = train_fold(train_loader, test_loader, weights_tensor, fold_idx + 1,
                           encoder=args.encoder, pooling=args.pooling)

        model.eval()
        with torch.no_grad():
            for X_batch, y_cls_batch, y_risk_batch, y_time_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                class_logits, risk_pred, time_pred = model(X_batch)
                all_test_cls_logits.append(class_logits.cpu().numpy())
                all_test_risk_preds.append(risk_pred.cpu().numpy())
                all_test_true_class.extend(y_cls_batch.numpy())
                all_test_true_risk.extend(y_risk_batch.numpy())

        torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"lstm_v2_fold{fold_idx+1}{out_sfx}.pt"))
        print(f"  Model saved -> lstm_v2_fold{fold_idx+1}.pt")

    # ---- 5. Temperature Scaling (class head only) ----
    print(f"\n{'='*60}")
    print("Temperature Scaling (class head only)")
    print(f"{'='*60}")

    all_test_cls_logits = np.concatenate(all_test_cls_logits, axis=0)
    all_test_true_class = np.array(all_test_true_class)

    def nll(T):
        if T <= 0:
            return 1e10
        s = all_test_cls_logits / T
        ps = np.exp(s - s.max(axis=1, keepdims=True))
        ps /= ps.sum(axis=1, keepdims=True)
        ps = np.clip(ps, 1e-10, 1 - 1e-10)
        return -np.mean([np.log(ps[i, int(all_test_true_class[i])])
                         for i in range(len(all_test_true_class))])

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    best_T = float(res.x)
    print(f"  Best temperature: {best_T:.4f}")

    np.savez(os.path.join(MODEL_DIR, f"temperature_v2{out_sfx}.npz"), T=best_T)
    print(f"  Saved -> {MODEL_DIR}/temperature_v2{out_sfx}.npz")

    # ---- 6. Final Class Metrics ----
    print(f"\n{'='*60}")
    print("Final Class Metrics (all folds combined, calibrated)")
    print(f"{'='*60}")

    calibrated_logits = all_test_cls_logits / best_T
    calibrated_preds = np.argmax(calibrated_logits, axis=1)

    print("\nClassification Report:")
    print(classification_report(
        all_test_true_class, calibrated_preds,
        target_names=CLASS_NAMES, digits=4,
        labels=list(range(NUM_CLASSES)),
    ))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_test_true_class, calibrated_preds, labels=list(range(NUM_CLASSES)))
    header = "         " + "  ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i,j]:>8}" for j in range(NUM_CLASSES))
        print(f"{name:>10}: {row}")

    # ---- 7. Risk Metrics ----
    print(f"\n{'='*60}")
    print("Final Risk Metrics")
    print(f"{'='*60}")

    all_test_risk_preds = np.concatenate(all_test_risk_preds)
    all_test_true_risk = np.array(all_test_true_risk)
    risk_mae = np.mean(np.abs(all_test_risk_preds - all_test_true_risk))
    risk_mse = np.mean((all_test_risk_preds - all_test_true_risk)**2)
    print(f"  Risk MAE: {risk_mae:.4f}")
    print(f"  Risk MSE: {risk_mse:.4f}")

    threshold = 0.35
    risk_thres_acc = np.mean((all_test_risk_preds >= threshold) == (all_test_true_risk >= threshold))
    print(f"  Risk threshold accuracy (t={threshold}): {risk_thres_acc*100:.2f}%")

    # ---- 8. Final Retrain on All Data ----
    print(f"\n{'='*60}")
    print("Final Retrain on All Data")
    print(f"{'='*60}")

    full_dataset = FallDatasetV2(X, y_class, y_risk, y_time, aug=True)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             pin_memory=True)

    model = RiskClassifierV2(encoder=args.encoder, pooling=args.pooling).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    criterion = MultiTaskLoss(weights_tensor.to(DEVICE), LAMBDA_RISK, LAMBDA_TIME)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    n_full = len(full_dataset)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_cls, y_risk, y_time in full_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_cls = y_cls.to(DEVICE, non_blocking=True)
            y_risk = y_risk.to(DEVICE, non_blocking=True)
            y_time = y_time.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with autocast_context(DEVICE):
                class_logits, risk_pred, time_pred = model(X_batch)
                loss, _, _, _ = criterion(
                    class_logits, risk_pred, time_pred,
                    y_cls, y_risk, y_time)
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
    final_path = os.path.join(MODEL_DIR, f"lstm_v2{out_sfx}.pt")
    torch.save(model.state_dict(), final_path)
    # 权重自描述：non-attn 的各种池化权重键名完全相同，缺了这份 meta
    # 评估侧会静默套用错误的池化方式，结果错但不报错。
    meta_path = Path(final_path).with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"encoder": args.encoder, "pooling": args.pooling,
                   "seed": args.seed, "data_suffix": sfx,
                   "out_suffix": out_sfx, "uses_score": uses_score,
                   "score_dims_pinned": not uses_score},
                  fh, ensure_ascii=False, indent=2)
    print(f"  元信息 -> {meta_path}")
    print(f"\nFinal model saved -> {final_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

