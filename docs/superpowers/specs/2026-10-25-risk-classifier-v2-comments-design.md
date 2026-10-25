# RiskClassifierV2 Forward-Pass Comments

## Goal

Explain the computation in `classifier/risk_classifier_v2.py` directly in the
source so readers can follow each encoder, pooling, and prediction-head step.

## Design

Add Chinese explanatory comments and docstrings around the existing code,
covering expected tensor shapes and the purpose of key operations. Document
TCN input projection, channel/time transposes, dilated residual blocks, and
return shape; attention score normalization and weighted aggregation; encoder
selection; each temporal pooling option; and how the classification, risk, and
time heads turn the pooled representation into outputs.

Use line-near explanations for meaningful computations, but do not paraphrase
syntax-only statements such as `super().__init__()`. Preserve useful existing
comments and clarify them where needed.

## Behavior Constraints

This is documentation-only: do not alter module names, parameter names,
initialization, layer ordering, tensor operations, activation functions,
dropout, defaults, forward outputs, or checkpoint compatibility.

## Validation

Run syntax compilation and a small forward-pass shape smoke check for both
`lstm` and `tcn` encoders across the supported pooling methods. Ensure the
working-tree diff contains only explanatory comments/docstrings in the target
Python file.
