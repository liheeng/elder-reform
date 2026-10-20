# train_v2 Readability Refactor

## Goal

Make `scripts/train/train_v2.py` easier to follow by grouping its workflow into
small, clearly named methods and keeping `main()` as a concise orchestration
entry point. Add comments for non-obvious training and evaluation rules.

## Scope

- Extract the stages in `main()` into focused helpers for argument/seeding,
  loading and preparing data, cross-validation, temperature calibration and
  metrics, final full-data training, and artifact saving.
- Clarify `train_fold()` by separating its training and validation epoch work
  where this reduces nesting and improves readability.
- Reuse training logic only where doing so preserves the distinct stopping
  criteria for cross-validation (validation loss) and final training (training
  loss).
- Keep `main()` as a sequence of named workflow calls.

## Behavior Constraints

Preserve CLI options and defaults, input/output filenames, score-channel
handling, normalization, class weighting, fold membership, augmentation,
losses, optimizer/scheduler settings, early-stopping behavior, calibration,
reported metrics, and saved artifact contents and formats. Do not change the
existing fold/evaluation methodology as part of this readability-only refactor.

## Validation

Run focused syntax/static checks and relevant tests if available. Review the
diff to confirm the refactor did not change the listed behavior or overwrite
pre-existing user edits in `train_v2.py`.
