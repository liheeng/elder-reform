# Realtime Detection Flow Readability Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `realtime_detect.py` so its `main()` orchestrates clearly named helpers for capture, per-frame logic, per-track handling, and cleanup without changing runtime behavior.

**Architecture:** Keep the module-level `main(video_path=None)` API and module-level feature/drawing helpers. Extract small functions that receive pipeline objects and per-person state explicitly, while preserving the current detection/tracking/prediction order and output behavior.

**Tech Stack:** Python, OpenCV, NumPy, project pose/tracker/sequence/classifier/risk-engine components.

## Global Constraints

- Preserve `main(video_path=None)` and its return behavior, detector/tracker update order, per-ID state cleanup, missing-target feature fill, score-channel compatibility behavior, sequence readiness and `(WINDOW_SIZE, 192)` input shape, risk-engine calls, overlay text and drawing semantics, keyboard exit behavior, and processed-frame reporting.
- Do not change model, risk, tracking, feature, or buffer algorithms.
- Keep `scripts/eval/eval_pipeline_regress.py` compatible with `rd.main(path)` and monkeypatching `draw_result`.
- Add concise comments only for non-obvious per-frame and per-track rules.

---

### Task 1: Extract the real-time flow into focused helpers

**Files:**
- Modify: `realtime_detect.py`
- Reference: `scripts/eval/eval_pipeline_regress.py` calls `rd.main(str(path))` and wraps `rd.draw_result`.
- Reference: `scripts/eval/trace_shapes.py` imports `build_192_feature`.

**Interfaces:**
- `_create_pipeline()` returns `(pose_ext, tracker, buffer, classifier, risk_engine, last_feature, display)`, using the existing constants and constructor arguments.
- `_open_video(video_path)` returns `(cap, source_name)`; when capture cannot open, it prints the existing error and returns `(None, source_name)`.
- `_draw_frame_status(frame, tracks, display, has_live_target, no_person_frames)` draws the existing lost-danger/no-person/target-lost banner and returns no value.
- `_process_live_track(frame, track, buffer, classifier, risk_engine, last_feature, display)` creates the normalized feature using the existing score compatibility rules, updates the sequence, predicts only when ready, updates display state, draws the live result, and returns the possibly modified frame.
- `_process_track(frame, track, has_live_target, buffer, classifier, risk_engine, last_feature, display)` delegates live tracks to `_process_live_track`; missing tracks use the existing feature-fill and lost-danger rendering rules.
- `_process_frame(frame, pose_ext, tracker, buffer, classifier, risk_engine, last_feature, display, no_person_frames)` extracts persons, updates tracks, clears removed IDs, updates banner state, processes each track in order, and returns `(frame, next_no_person_frames)`.
- `_release_video(cap)` releases the capture and destroys OpenCV windows.
- `main(video_path=None)` remains the externally callable loop and preserves the existing `None` return behavior on success or camera-open failure.

- [x] **Step 1: Record the current integration contract**

Inspect `scripts/eval/eval_pipeline_regress.py` and verify it relies on `rd.main(video_path)`, `rd.draw_result`, `rd.LSTM_MODEL`, `rd.NORM_PARAMS`, and `rd.TEMP_PARAMS`. Do not rename or move those module-level symbols.

- [x] **Step 2: Extract pipeline initialization**

Move the five existing constructors and the two per-person dictionaries from `main()` into:

```python
def _create_pipeline():
    pose_ext = PoseExtractor(model_path=POSE_MODEL)
    tracker = PersonTracker(max_missing=MISSING_TOLERANCE)
    buffer = SequenceBufferV3(seq_len=WINDOW_SIZE)
    classifier = LSTMClassifierV2(
        model_path=LSTM_MODEL,
        norm_path=NORM_PARAMS,
        temperature_path=TEMP_PARAMS,
    )
    risk_engine = RiskEngineV2()
    last_feature = {}
    display = {}
    return pose_ext, tracker, buffer, classifier, risk_engine, last_feature, display
```

Add a concise comment explaining that the two dictionaries retain per-ID feature continuity and last alert state.

- [x] **Step 3: Extract capture opening**

Move camera/video selection, `VideoCapture`, and open failure reporting into `_open_video(video_path)`. Preserve `source_name` exactly (`"Camera"` or `os.path.basename(video_path)`) and preserve the existing error print. `main()` retains the FPS fallback to 25 FPS and the source/model/window startup messages. It returns `None` if the returned capture is `None`.

- [x] **Step 4: Extract frame status banner**

Move `lost_danger` selection and the three existing banner cases into `_draw_frame_status(...)`. In `_process_frame`, update the counter exactly as before (`+1` with no detected persons; reset to zero otherwise), then pass the resulting counter into the status helper. Keep banner strings, coordinates, colors, fonts, and thicknesses unchanged.

- [x] **Step 5: Extract live-track feature construction and inference**

Move the current live-track body into `_process_live_track(...)`. Preserve the three score-channel cases exactly:

```python
if getattr(classifier, "uses_score", False):
    kw = {"score": bbox_score, "has_score": 1.0}
elif getattr(classifier, "score_dims_pinned", False):
    kw = {"score": 0.0, "has_score": 0.0}
else:
    kw = {}
```

Keep normalization fallback, `last_feature` update, sequence update/readiness check, exact shape guard `(WINDOW_SIZE, 192)`, `classifier.predict`, `risk_engine.update`, `display` update, and the existing `draw_result(...)` call and arguments. Add a comment that these branches keep inference input compatible with score-enabled, score-pinned, and legacy weights.

- [x] **Step 6: Extract missing-track behavior and cleanup**

Move per-track dispatch into `_process_track(...)`. If `track["missing"] > 0`, keep filling the sequence with `last_feature[person_id]` when available; explain that repeating the last feature avoids an artificial velocity spike. Continue to draw only when the last display state is DANGER and there is no live target in the current frame. For removed IDs, keep removing the ID from the buffer and risk engine and popping both dictionaries.

- [x] **Step 7: Extract frame processing**

Implement `_process_frame(...)` preserving the exact call order:

```text
pose_ext.extract(frame)
tracker.update(persons)
clear removed IDs
update no-person counter and draw status
process tracks in tracker order
return frame and counter
```

Use `_process_track(...)` for each track and return the possibly annotated frame.

- [x] **Step 8: Simplify `main()` and guarantee cleanup**

Keep `main(video_path=None)` as the compatibility API. It should initialize the pipeline, open the source, loop over frames, increment `frame_count` after each successful read, call `_process_frame`, draw the frame counter, show the window, preserve the `q` key check, and print the processed frame total. Use `try/finally` after a successful open so the capture is released and `cv2.destroyAllWindows()` runs on normal exit or an exception. Do not add new catches or change the current capture-open failure path.

- [x] **Step 9: Run syntax and focused mock-flow checks**

Run:

```bash
python -m py_compile realtime_detect.py
```

Then run a short Python smoke check that monkeypatches the pose extractor, tracker, buffer, classifier, risk engine, `cv2.VideoCapture`, `cv2.imshow`, and `cv2.waitKey`. Verify all of the following:

- A live track warms its buffer, predicts once it is ready, and invokes the module-level `draw_result`.
- A missing track repeats its last feature and only draws a historical box for a previous DANGER state when there is no live person.
- A removed track is removed from the buffer, risk engine, `last_feature`, and `display`.
- The frame loop exits when `waitKey` returns `q`, releases the capture, destroys windows, and reports the successful frame count.

Expected: all assertions pass; the smoke check does not load actual model weights or open a camera. Verified using the repository's `.conda` Python environment.

- [x] **Step 10: Check end-to-end regression assets**

Check that the configured model files and the videos selected by `scripts/eval/eval_pipeline_regress.py` exist. If they do, run the narrowest relevant command:

```bash
python scripts/eval/eval_pipeline_regress.py --which normal
```

All seven normal-video fixtures are absent from the configured archive mount, so the end-to-end regression was skipped. Do not download datasets or models.

- [x] **Step 11: Review the behavior-sensitive diff**

Run `git diff --check` and inspect `git diff -- realtime_detect.py`. Confirm unchanged public names, drawing text, score branches, state cleanup, tracker/risk calls, sequence shape, and loop output. Do not modify unrelated files.
