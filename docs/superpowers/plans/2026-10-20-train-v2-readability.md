# train_v2 Readability Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `train_v2.py` into clearly named workflow and epoch helpers without changing training behavior.

**Architecture:** Keep model and loss definitions unchanged. Extract data preparation, cross-validation, calibration/metrics, final retraining, and artifact persistence into focused functions; reduce `main()` to orchestration. Split repeated epoch mechanics only where it preserves the distinct validation-loss and training-loss stopping criteria.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn, SciPy.

## Global Constraints

- Preserve CLI options and defaults, input/output filenames, score-channel handling, normalization, class weighting, fold membership, augmentation, losses, optimizer/scheduler settings, early-stopping behavior, calibration, reported metrics, and saved artifact contents and formats.
- Do not change the existing fold/evaluation methodology.
- Preserve pre-existing user edits in `scripts/train/train_v2.py`; do not overwrite unrelated work.
- Add only comments that clarify non-obvious behavior.

---

### Task 1: Extract the training workflow into named stages

**Files:**
- Modify: `scripts/train/train_v2.py`
- Test: Python syntax compilation and focused structural/behavior checks; this repository has no discovered `test_*.py` suite.

**Interfaces:**
- `parse_args()` returns the parsed `argparse.Namespace`.
- `load_and_prepare_data(suffix, use_score_channel, out_suffix)` returns `(X, y_class, y_risk, y_time, vids, uses_score)` and saves the existing normalization artifact.
- `compute_class_weights(y_class)` returns a `torch.Tensor` of four class weights.
- `assign_cv_groups(vids)` preserves the current Pre-VFall remapping and returns the three existing held-out groups.
- `run_cv_fold(X, y_class, y_risk, y_time, vids, test_scenes, fold_idx, class_weights, out_suffix, encoder, pooling)` trains one fold and returns its class logits/targets and risk predictions/targets.
- `run_cross_validation(X, y_class, y_risk, y_time, vids, class_weights, out_suffix, encoder, pooling)` returns combined held-out class logits, class targets, risk predictions, and risk targets.
- `calibrate_temperature(logits, targets, out_suffix)` returns the fitted scalar temperature and writes the same temperature artifact.
- `report_class_metrics(logits, targets, temperature)` and `report_risk_metrics(predictions, targets)` print the same metrics as before.
- `train_full_dataset(X, y_class, y_risk, y_time, class_weights, encoder, pooling)` returns the selected final model.
- `_run_train_epoch(model, loader, optimizer, criterion)` runs one training epoch and returns the four sample-weighted losses.
- `_run_validation_epoch(model, loader, criterion)` returns validation loss and the predictions/targets used for fold logging.
- `save_final_artifacts(model, out_suffix, data_suffix, encoder, pooling, seed, uses_score)` writes the existing final model and metadata artifacts.
- `main()` sequences these operations and keeps current console stage headings.

- [ ] **Step 1: Capture the current script's command-line contract**

Run `python scripts/train/train_v2.py --help` and save the output for comparison during the refactor. Confirm all existing options remain present: `--suffix`, `--out-suffix`, `--encoder`, `--pooling`, `--seed`, and `--score-channel`.

- [ ] **Step 2: Extract argument parsing and random seeding**

Move the existing parser declarations without changing names, defaults, choices, or help strings into `parse_args()`. Move the existing optional random/NumPy/PyTorch seeding into `seed_everything(seed)` and call it from `main()` immediately after parsing.

- [ ] **Step 3: Extract data loading, score-channel handling, and normalization**

Move the existing five `np.load` calls, score-channel checks/zeroing, mean/std computation, score-dimension pinning, normalized `X` calculation, and `norm_v2{out_suffix}.npz` write into `load_and_prepare_data(suffix, use_score_channel, out_suffix)`. Preserve the exact `SystemExit` path and normalization values.

- [ ] **Step 4: Extract class weights and fold assignment**

Move the existing inverse-frequency weight calculation into `compute_class_weights(y_class)`. Move the current Pre-VFall `scene_id == 10` remapping and the `FOLD_GROUPS` constant to `assign_cv_groups(vids)`. Preserve the exact groups `[[0, 1, 10], [2, 3, 11], [4, 5, 12]]` and in-place/returned ID semantics used by the caller.

- [ ] **Step 5: Extract cross-validation execution and prediction collection**

Extract one fold's work into `run_cv_fold(...)`, then leave `run_cross_validation(...)` to iterate the existing groups and collect fold outputs. Keep each fold's masks, dataset augmentation flags, loaders, call to `train_fold`, fold checkpoint filename, prediction order, and returned arrays unchanged. Add a concise comment that the held-out fold is passed as `val_loader`, matching current behavior.

- [ ] **Step 6: Split one-fold epoch work without changing optimization**

Extract `_run_train_epoch(model, loader, optimizer, criterion)` and `_run_validation_epoch(model, loader, criterion)` from `train_fold()`. Keep autocast, gradient scaling, unscaling, clipping, scheduler inputs, metric accumulation, checkpoint selection, and early stopping equivalent. `train_fold()` should orchestrate epochs and retain validation-loss-based checkpoint selection.

- [ ] **Step 7: Extract calibration and reporting**

Move the current NLL closure and bounded `minimize_scalar` call into `calibrate_temperature(logits, targets, out_suffix)`. Move the existing calibrated classification report/confusion matrix and risk MAE/MSE/threshold-accuracy output into `report_class_metrics(...)` and `report_risk_metrics(...)`. Keep metrics, labels, precision, thresholds, print layout, and artifact values unchanged.

- [ ] **Step 8: Extract full-data retraining and final artifact persistence**

Move the final full-data training loop into `train_full_dataset(...)`. Preserve training-loss-based scheduling, early stopping, augmentation, optimizer, gradient scaler, and selected-state behavior. Move final `.pt` and `.meta.json` persistence into `save_final_artifacts(...)`, preserving exact JSON keys and filenames. Keep `norm_v2*.npz` and `temperature_v2*.npz` writes in their existing data-preparation and calibration stages so each output is saved once.

- [ ] **Step 9: Reduce `main()` to named orchestration**

Wire the helpers in the existing order: parse and seed, load/prepare, compute weights, cross-validate, calibrate/report, retrain, save. Keep current console headings and output ordering. Do not modify model architecture, data labels, or fold methodology.

- [ ] **Step 10: Verify syntax and CLI compatibility**

Run:

```bash
python -m py_compile scripts/train/train_v2.py
python scripts/train/train_v2.py --help
```

Expected: both exit successfully, and `--help` contains the same six options, choices, defaults, and descriptions captured in Step 1.

- [ ] **Step 11: Review behavior-sensitive diff**

Review `git diff -- scripts/train/train_v2.py` and verify that constants, calculations, filenames, fold order, loss/optimizer/scheduler arguments, early-stop conditions, metrics, and serialized metadata are unchanged. Confirm the pre-existing import and `# noqa: UP031` edits remain intact.
