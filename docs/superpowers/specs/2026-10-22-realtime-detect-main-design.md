# Realtime Detection Flow Readability Refactor

## Goal

Make `realtime_detect.py` easier to understand by extracting the responsibilities
inside `main()` into clearly named methods and documenting the important
per-frame and per-track rules.

## Design

Keep the existing `main(video_path=None)` function as the public entry point so
callers such as `scripts/eval/eval_pipeline_regress.py` continue to work. Keep
the current module-level helpers for feature construction and drawing.

Extract the main workflow into a small set of focused functions:

- Initialize the pose extractor, tracker, sequence buffer, classifier, risk
  engine, and per-person state.
- Open the camera or video and report its source and effective FPS.
- Process one frame: detect people, update tracks, clear state for expired IDs,
  draw the no-person/lost-danger banner, and dispatch each track.
- Process one track: handle missing detections using the existing last-feature
  buffer fill and lost-danger display rules, or pass a live detection into
  feature construction and prediction.
- Build a live person's feature using the existing score-channel compatibility
  branches, normalize keypoints, update their sequence, and run inference only
  once the window is ready.
- Release the capture and destroy OpenCV windows on normal loop exit.

Pass shared collaborators and mutable state explicitly where practical. Avoid
introducing a detector class or changing the pipeline's algorithms.

## Behavior Constraints

Preserve `main(video_path=None)` and its return behavior, detector/tracker update
order, per-ID state cleanup, missing-target feature fill, score-channel
compatibility behavior, sequence readiness and `(WINDOW_SIZE, 192)` input shape,
risk-engine calls, overlay text and drawing semantics, keyboard exit behavior,
and processed-frame reporting.

Add concise comments only for non-obvious rules, especially why a missing
person's last feature is fed into the sequence, why score inputs differ between
model generations, and when a lost target's historical box is shown.

## Validation

Run Python syntax compilation and focused tests or mock checks that exercise
normal live-track processing, missing-track behavior, removal cleanup, and
frame-loop termination. Run the existing end-to-end regression only if its
video and model assets are available. Review the diff for preserved call order
and output behavior.
