# RiskClassifierV2 Source Comments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add detailed Chinese explanations to `risk_classifier_v2.py` so readers can follow the forward pass and tensor shapes without changing model behavior.

**Architecture:** Keep all executable statements byte-for-byte unchanged and add only comments/docstrings beside meaningful operations. Document the TCN encoder, attention pooling, encoder selection, temporal pooling, and three output heads in source order.

**Tech Stack:** Python, PyTorch.

## Global Constraints

- This is documentation-only: do not alter module names, parameter names, initialization, layer ordering, tensor operations, activation functions, dropout, defaults, forward outputs, or checkpoint compatibility.
- Explain tensor dimensions using the existing conventions: input `(B, T, input_dim)`, hidden states `(B, T, hidden_dim)`, TCN convolution layout `(B, hidden_dim, T)`, and pooled representation `(B, hidden_dim)`.
- Do not translate syntax-only statements into redundant comments.

---

### Task 1: Explain encoder, pooling, and prediction-head computations

**Files:**
- Modify: `classifier/risk_classifier_v2.py`
- Test: Syntax compilation and PyTorch forward-pass shape smoke check; no dedicated unit tests were found for this module.

**Interfaces:**
- No public interface changes. Existing classes, constructors, parameter names, and return tuple remain exactly as they are.

- [x] **Step 1: Add a shape-oriented class docstring to `TCNEncoder`**

Document that input `x` has shape `(B, T, input_dim)`, `input_proj` maps the last dimension to `hidden_dim`, and Conv1d expects `(B, C, T)`. Explain that each residual block applies dilated convolution, GELU, LayerNorm, dropout, and residual addition while preserving the time length, then returns `(B, T, hidden_dim)`. Keep the existing receptive-field explanation.

- [x] **Step 2: Explain the TCN forward operations**

Add comments beside the existing statements with this level of detail:

```python
# 先把每帧特征投影到 hidden_dim，再转成 Conv1d 所需的 (B, C, T)。
h = self.input_proj(x).transpose(1, 2)
for blk in self.blocks:
    # 膨胀卷积聚合相邻时间信息；GELU 提供非线性。
    res = nn.functional.gelu(blk["conv"](h))
    # LayerNorm 作用于 hidden_dim，所以暂时转回 (B, T, C)，再转回卷积布局。
    res = blk["norm"](res.transpose(1, 2)).transpose(1, 2)
    # Dropout 正则化卷积分支，残差相加保留输入表示并便于深层优化。
    h = h + self.dropout(res)
# 转回 (B, T, hidden_dim)，与 LSTM 编码器输出约定一致。
return h.transpose(1, 2)
```

Retain existing operations verbatim; comments must not imply causal convolution because this TCN is non-causal.

- [x] **Step 3: Explain attention pooling**

Document that `AttentionPool.forward()` receives `(B, T, hidden_dim)`, produces one scalar score per time step, softmax normalizes scores over `T`, and the weighted sum returns one `(B, hidden_dim)` representation per sequence.

- [x] **Step 4: Explain model initialization by responsibility**

Add a class docstring for `RiskClassifierV2` describing its input `(B, T, input_dim)` and outputs `(B, num_classes)`, `(B,)`, `(B,)`. Expand the existing constructor comments to clarify:

- `num_layers=None` selects 2 LSTM layers or 4 TCN blocks.
- The selected encoder maps the frame sequence to `(B, T, hidden_dim)`.
- Pooling reduces time to `(B, hidden_dim)`; all pooling choices keep the same feature width.
- The class, risk, and time projection branches are separate task-specific MLPs.
- Sigmoid bounds risk to `[0, 1]`; the time head is intentionally linear.

Preserve existing layer definitions and defaults exactly.

- [x] **Step 5: Explain `RiskClassifierV2.forward()` step by step**

Add comments beside the existing logic to explain:

1. The selected encoder transforms per-frame input into temporal hidden states.
2. `lastmean` averages the final hidden state and the sequence mean equally; `last`, `mean`, and `max` use their named temporal reductions; `attn` uses learned weights.
3. Each separate projection/head maps the pooled representation to class logits, bounded risk, and an unconstrained time estimate.
4. `squeeze(-1)` removes only the final singleton output dimension, preserving batch shape `(B,)`, including batch size one.

Do not change the existing branch ordering or return tuple.

- [x] **Step 6: Verify comments did not change behavior**

Run from the repository root:

```bash
./.conda/bin/python -m py_compile classifier/risk_classifier_v2.py
./.conda/bin/python - <<'PY'
import torch
from classifier.risk_classifier_v2 import POOLINGS, RiskClassifierV2

x = torch.randn(2, 30, 192)
for encoder in ("lstm", "tcn"):
    for pooling in POOLINGS:
        model = RiskClassifierV2(encoder=encoder, pooling=pooling).eval()
        with torch.no_grad():
            class_logits, risk, time = model(x)
        assert class_logits.shape == (2, 4), (encoder, pooling, class_logits.shape)
        assert risk.shape == (2,), (encoder, pooling, risk.shape)
        assert time.shape == (2,), (encoder, pooling, time.shape)
print("all encoder/pooling output-shape checks passed")
PY
git diff --check
```

Expected: compilation succeeds, the smoke check prints `all encoder/pooling output-shape checks passed`, and `git diff --check` succeeds. Verified for all configured encoders and pooling choices.

- [x] **Step 7: Review the source-only diff**

Review `git diff -- classifier/risk_classifier_v2.py`. Confirm the diff adds comments/docstrings only and does not modify executable statements, class/parameter names, defaults, or state-dict keys.
