#!/usr/bin/env python3
"""按数据来源 + 标签语义切分，验证 0.5 预警档是否真的学到。"""
import sys, os, numpy as np, torch
repo, mdir, xf, rf, vf, tag = sys.argv[1:7]
sys.path.insert(0, repo)
from classifier.risk_classifier_v2 import RiskClassifierV2

dev = "cuda" if torch.cuda.is_available() else "cpu"
X = np.load(xf).astype(np.float32); R = np.load(rf).astype(np.float32)
V = np.load(vf)
norm = np.load(os.path.join(mdir, "norm_v2_nonperson.npz"))
Xn = (X - norm["mean"].astype(np.float32)) / (norm["std"].astype(np.float32) + 1e-8)

m = RiskClassifierV2(input_dim=192, num_classes=4)
m.load_state_dict(torch.load(os.path.join(mdir, "lstm_v2_nonperson.pt"), map_location=dev)); m.to(dev).eval()
P = np.empty(len(Xn), dtype=np.float32)
with torch.no_grad():
    for i in range(0, len(Xn), 512):
        P[i:i+512] = m(torch.tensor(Xn[i:i+512]).to(dev))[1].cpu().numpy().ravel()

pv = V[:, 0] == 10                      # Pre-VFall 来源
thr = 0.35
print(f"===== {tag} | 总 {len(R)} 样本 =====")
print(f"[来源] Pre-VFall {int(pv.sum())} | Le2i {int((~pv).sum())}")
print()
print("-- Pre-VFall 部分按真实标签 --")
for lab, name in [(0.0, "Normal(0.0)"), (0.5, "Abnormal(0.5)"), (1.0, "Fall(1.0)")]:
    s = pv & np.isclose(R, lab)
    if s.sum() == 0:
        print(f"   {name:>14}: 0 条"); continue
    print(f"   {name:>14}: {int(s.sum()):>5} 条  预测均值 {P[s].mean():.4f}  std {P[s].std():.4f}"
          f"  预警占比 {np.mean(P[s]>=thr)*100:5.1f}%")
print()
print("-- Le2i 部分按真实标签 --")
for name, s in [("risk=0.0", ~pv & np.isclose(R, 0.0)),
                ("0<risk<1(衰减)", ~pv & (R > 1e-6) & (R < 0.999)),
                ("risk=1.0", ~pv & np.isclose(R, 1.0))]:
    if s.sum() == 0:
        print(f"   {name:>16}: 0 条"); continue
    print(f"   {name:>16}: {int(s.sum()):>5} 条  预测均值 {P[s].mean():.4f}  std {P[s].std():.4f}"
          f"  预警占比 {np.mean(P[s]>=thr)*100:5.1f}%")
print()
ab = pv & np.isclose(R, 0.5)
print(f">>> 关键: 真异常(0.5)样本数 = {int(ab.sum())}, 其中被识别为预警 = "
      f"{int((P[ab]>=thr).sum())} ({np.mean(P[ab]>=thr)*100:.1f}%)" if ab.sum() else
      f">>> 关键: 真异常(0.5)样本数 = 0 —— 该数据集不存在预警档，无从学习")
print(f">>> 全体 Risk MAE = {np.mean(np.abs(P-R)):.4f}")
