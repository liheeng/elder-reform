# -*- coding: utf-8 -*-
"""
eval_pipeline_regress.py

端到端回归：在单进程内把真实的 realtime_detect 主循环跑在一组固定视频上，
汇总每个视频的报警帧统计。用于改模型 / 改阈值前后的 A/B 对比。

为什么必须端到端:
    离线抽样只覆盖已标注的帧，会漏掉未标注的情形。这个坑在车类交叉验证上
    真实出现过 —— 离线统计说"0 误杀"，端到端实测误杀 9/105，
    因为出问题的帧根本不在标注集里。

做法:
    monkeypatch realtime_detect 的模型路径常量与屏幕输出，跑完所有视频，
    避免为每个视频重复加载模型。

基准值（lstm_v2.pt，改动前）:
    正常 7 段：WARNING / DANGER 全为 0
    跌倒 6 段：见 docs/tuning-log.md 记录的首 DANGER 帧
    真实 6 段：见 docs/tuning-log.md

用法:
    python scripts/eval/eval_pipeline_regress.py
    python scripts/eval/eval_pipeline_regress.py --suffix _np
    python scripts/eval/eval_pipeline_regress.py --which normal
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE = Path("/Volumes/media-backup/AI-data/dataset/falldown/Le2i-v2")
REAL_DIR = Path("/Volumes/media-backup/AI-data/real-resource/Yolo视觉素材/人摔倒")

NORMALS = [
    ("Office 19", ARCHIVE / "Office/Office/video (19).avi"),
    ("Office 23", ARCHIVE / "Office/Office/video (23).avi"),
    ("Office 29", ARCHIVE / "Office/Office/video (29).avi"),
    ("Office 33", ARCHIVE / "Office/Office/video (33).avi"),
    ("Lecture 15", ARCHIVE / "Lecture_room/Lecture room/video (15).avi"),
    ("Lecture 20", ARCHIVE / "Lecture_room/Lecture room/video (20).avi"),
    ("Lecture 26", ARCHIVE / "Lecture_room/Lecture room/video (26).avi"),
]
FALLS = [
    ("Office 1", ARCHIVE / "Office/Office/video (1).avi"),
    ("Office 5", ARCHIVE / "Office/Office/video (5).avi"),
    ("Lecture 1", ARCHIVE / "Lecture_room/Lecture room/video (1).avi"),
    ("Home_01 1", ARCHIVE / "Home_01/Home_01/Videos/video (1).avi"),
    ("Home_01 10", ARCHIVE / "Home_01/Home_01/Videos/video (10).avi"),
    ("Home_02 1", ARCHIVE / "Home_02/Home_02/Videos/video (1).avi"),
]
REAL = [
    ("pd1", REAL_DIR / "person_falldown-1.mp4"),
    ("pd2", REAL_DIR / "person_falldown-2.mp4"),
    ("pd3", REAL_DIR / "person_falldown-3.mp4"),
    ("pd4", REAL_DIR / "person_falldown-4.mp4"),
    ("stair1", REAL_DIR / "fall_staircase-1.mp4"),
    ("stair2", REAL_DIR / "fall_staircase-2.mp4"),
]

STATE = {"marks": [], "rows": [], "results": []}


def install_spies(rd):
    """截获 draw_result（拿到每帧状态）与 Frame:N 文本（拿到帧号）。"""
    orig_draw = rd.draw_result
    orig_put_text = cv2.putText

    def spy_draw(frame, bbox, state, risk_score, person_id, time_to_fall=0):
        STATE["rows"].append((state, person_id, round(float(risk_score), 4)))
        return orig_draw(frame, bbox, state, risk_score, person_id, time_to_fall)

    def spy_put_text(img, text, *a, **k):
        if text.startswith("Frame: "):
            STATE["marks"].append((int(text.split(":")[1]), len(STATE["rows"])))
        return orig_put_text(img, text, *a, **k)

    rd.draw_result = spy_draw
    cv2.putText = spy_put_text


def run_one(rd, tag, path):
    STATE["marks"].clear()
    STATE["rows"].clear()
    print(f"--- {tag} ---", flush=True)
    rd.main(str(path))

    rows, prev = [], 0
    for frame_no, mark in STATE["marks"]:
        for state, tid, risk in STATE["rows"][prev:mark]:
            rows.append((frame_no, tid, state, risk))
        prev = mark

    if not rows:
        print(f"RESULT {tag}: 无推理", flush=True)
        STATE["results"].append((tag, None, None, None, {}))
        return

    counter = Counter(r[2] for r in rows)
    danger = sorted({r[0] for r in rows if r[2] == "DANGER"})
    warn = sorted({r[0] for r in rows if r[2] == "WARNING"})
    total_frames = max(len(STATE["marks"]), 1)
    print(f"RESULT {tag}: draws={len(rows)} {dict(counter)} "
          f"首WARNING={warn[0] if warn else '-'} 首DANGER={danger[0] if danger else '-'} "
          f"DANGER帧占比={100 * len(danger) / total_frames:.1f}%", flush=True)
    STATE["results"].append(
        (tag, warn[0] if warn else None, danger[0] if danger else None, len(rows), dict(counter))
    )


def main():
    ap = argparse.ArgumentParser(description="端到端回归：真实 realtime_detect 主循环")
    ap.add_argument("--suffix", default="",
                    help="模型文件名后缀，如 _np / _baseline；默认空=生产模型")
    ap.add_argument("--which", default="all", choices=["all", "normal", "fall", "real"])
    args = ap.parse_args()

    model = PROJECT_ROOT / "models" / f"lstm_v2{args.suffix}.pt"
    norm = PROJECT_ROOT / "models" / f"norm_v2{args.suffix}.npz"
    temp = PROJECT_ROOT / "models" / f"temperature_v2{args.suffix}.npz"
    for path in (model, norm):
        if not path.exists():
            print(f"[错误] 缺少模型文件: {path}")
            return 1

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    import realtime_detect as rd

    rd.LSTM_MODEL = str(model)
    rd.NORM_PARAMS = str(norm)
    if temp.exists():
        rd.TEMP_PARAMS = str(temp)

    print("=" * 60)
    print(f"端到端回归  模型={model.name}" + (f"  温度={temp.name}" if temp.exists() else ""))
    print("=" * 60)

    install_spies(rd)
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: 0

    groups = []
    if args.which in ("all", "normal"):
        groups.append(("NORMAL", NORMALS))
    if args.which in ("all", "fall"):
        groups.append(("FALL", FALLS))
    if args.which in ("all", "real"):
        groups.append(("REAL", REAL))

    missing = []
    for label, items in groups:
        for tag, path in items:
            if not Path(path).exists():
                missing.append(f"{label} {tag}: {path}")
                continue
            run_one(rd, f"{label} {tag}", path)

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"{'视频':<16}{'draws':>7}{'首WARNING':>11}{'首DANGER':>11}  状态分布")
    for tag, warn, danger, draws, counter in STATE["results"]:
        print(f"{tag:<16}{draws if draws is not None else '-':>7}"
              f"{warn if warn is not None else '-':>11}"
              f"{danger if danger is not None else '-':>11}  {counter}")

    bad = [r[0] for r in STATE["results"] if r[0].startswith("NORMAL") and (r[1] or r[2])]
    print()
    if bad:
        print(f"❌ 正常视频出现报警（应为 0）: {bad}")
    else:
        print("✅ 所有正常视频 WARNING / DANGER 均为 0")
    if missing:
        print("\n[跳过] 文件不存在:")
        for m in missing:
            print(f"  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
