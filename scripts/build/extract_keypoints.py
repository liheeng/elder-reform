# -*- coding: utf-8 -*-
"""
extract_keypoints.py

从 Le2i 原始视频预提取 YOLO-Pose 关键点缓存，供 build_dataset_v2.py / verify_real_videos.py 读取。

产出（每个视频两个文件，写入 dataset/yolo_keypoints/）:
    {scene_name}_video_{video_id}_keypoints.npy   (T, 17, 2) float32  像素坐标
    {scene_name}_video_{video_id}_confs.npy       (T, 17)    float32  关键点置信度

T 为视频原始帧数（不降采样，10fps 抽样由 build 阶段完成）。
漏检帧/无人帧写入全 0 行，与 features.interpolate_missing_frames 的零填充约定一致。

逐帧走 PoseExtractor（含物品误检质量门与分裂检测合并），多人场景取
kpts_conf.sum() 最大者，使训练前置逻辑与 realtime_detect.py 推理保持一致。

用法:
    python scripts/build/extract_keypoints.py                      # 全部场景，跳过已有缓存
    python scripts/build/extract_keypoints.py --scenes Coffee_room_01 --ids 1-5
    python scripts/build/extract_keypoints.py --dry-run            # 只列出将处理的视频
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pose.pose_extractor import PoseExtractor  # noqa: E402

ARCHIVE = r"/Volumes/media-backup/AI-data/falldown/Le2i-v2"
KEYPOINTS_DIR = PROJECT_ROOT / "dataset" / "yolo_keypoints"

# 仓库 models/ 缺权重时的回退位置（与 pose_extractor.py 使用的同一份 yolo26n-pose.pt）
FALLBACK_MODEL = "/Users/henry/git/AI-Projects/YOLO/yolo-mlx/models/yolo26n-pose.pt"

# 与 build_dataset_v2.py 的 SCENES / NORMAL_SCENES 对齐
SCENES = {
    "Coffee_room_01": os.path.join(ARCHIVE, "Coffee_room_01", "Coffee_room_01", "Videos"),
    "Coffee_room_02": os.path.join(ARCHIVE, "Coffee_room_02", "Coffee_room_02", "Videos"),
    "Home_01": os.path.join(ARCHIVE, "Home_01", "Home_01", "Videos"),
    "Home_02": os.path.join(ARCHIVE, "Home_02", "Home_02", "Videos"),
    "Lecture_room": os.path.join(ARCHIVE, "Lecture_room", "Lecture room"),
    "Office": os.path.join(ARCHIVE, "Office", "Office"),
}

NUM_KEYPOINTS = 17


@contextlib.contextmanager
def suppress_native_stderr():
    """把 C 层直接写入 fd 2 的日志重定向到 devnull。

    Le2i 视频缺音轨，ffmpeg 每次打开都会打印 "[mp3float] Header missing"。
    该输出绕过 Python 的 sys.stderr，无法用 contextlib.redirect_stderr 屏蔽，
    只能从文件描述符层面重定向。
    """
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def cache_paths(scene_name, video_id, out_dir):
    """返回该视频的 (keypoints 路径, confs 路径)。"""
    stem = "%s_video_%s" % (scene_name, video_id)
    return out_dir / (stem + "_keypoints.npy"), out_dir / (stem + "_confs.npy")


def iter_videos(scene_name, video_dir, id_filter=None):
    """按 video id 数值升序产出 (video_id, 视频路径)。"""
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        return []
    videos = []
    for path in video_dir.iterdir():
        if path.suffix.lower() != ".avi":
            continue
        video_id = path.stem.replace("video (", "").replace(")", "")
        if not video_id.isdigit():
            continue
        if id_filter is not None and int(video_id) not in id_filter:
            continue
        videos.append((video_id, path))
    videos.sort(key=lambda item: int(item[0]))
    return videos


def parse_ids(spec):
    """解析 --ids 参数，支持 "1,2,5" 与 "1-10" 混写；None 表示不过滤。"""
    if not spec:
        return None
    ids = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(chunk))
    return ids


def select_person(persons):
    """从当前帧的候选里选人：取关键点置信度和最大者。

    返回 (keypoints(17,2), confs(17)) 或 None。
    """
    best = None
    best_score = -1.0
    for person in persons:
        kpts_conf = person.get("kpts_conf")
        if kpts_conf is None:
            continue
        score = float(np.sum(kpts_conf))
        if score > best_score:
            best_score = score
            best = person
    if best is None:
        return None
    return (np.asarray(best["keypoints"], dtype=np.float32),
            np.asarray(best["kpts_conf"], dtype=np.float32))


def extract_video(video_path, extractor, log_every=200):
    """逐帧抽点，返回 (keypoints (T,17,2), confs (T,17), 检出帧数)。

    解码或推理异常时返回 (None, None, 0)，由调用方决定是否跳过该视频。
    整个循环置于 fd 2 抑制内，以静音 ffmpeg 在读取过程中输出的缺音轨告警。
    """
    kpts_list = []
    confs_list = []
    hit = 0
    cap = None
    with suppress_native_stderr():
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None, None, 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                selected = select_person(extractor.extract(frame))
                if selected is None:
                    kpts_list.append(np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32))
                    confs_list.append(np.zeros(NUM_KEYPOINTS, dtype=np.float32))
                else:
                    kpts_list.append(selected[0])
                    confs_list.append(selected[1])
                    hit += 1
                if log_every and len(kpts_list) % log_every == 0:
                    print("      ... %d frames (检出 %d)" % (len(kpts_list), hit), flush=True)
        except Exception as exc:
            print("  [警告] 抽点失败 (%s): %s" % (type(exc).__name__, exc))
            return None, None, 0
        finally:
            if cap is not None:
                cap.release()

    kpts = np.stack(kpts_list) if kpts_list else np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32)
    confs = np.stack(confs_list) if confs_list else np.zeros((0, NUM_KEYPOINTS), dtype=np.float32)
    return kpts, confs, hit


def save_cache(kpts, confs, kp_path, cf_path):
    """先写临时文件再 os.replace，避免中断留下半截缓存。"""
    for arr, path in ((kpts, kp_path), (confs, cf_path)):
        tmp = path.with_name(path.name + ".tmp")
        # 显式 open，避免 np.save 给不含 .npy 的文件名自动补后缀
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, path)


def resolve_model(model):
    """优先用仓库内 models/ 权重，其次回退到工作区里实际存在的 yolo26n-pose.pt。"""
    if model:
        return model
    for candidate in (PROJECT_ROOT / "models" / "yolo26n-pose.pt", Path(FALLBACK_MODEL)):
        if Path(candidate).exists():
            return str(candidate)
    return str(PROJECT_ROOT / "models" / "yolo26n-pose.pt")


def resolve_device(device):
    """auto → 有 MPS 用 MPS，有 CUDA 用 CUDA，否则 CPU。"""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="从 Le2i 视频预提取 YOLO-Pose 关键点缓存")
    parser.add_argument("--scenes", default=",".join(SCENES),
                        help="逗号分隔的场景名，默认全部: " + ",".join(SCENES))
    parser.add_argument("--ids", default=None, help="视频 id 过滤，如 1-10,15")
    parser.add_argument("--model", default=None,
                        help="YOLO-Pose 权重路径，默认 models/yolo26n-pose.pt（缺失时回退工作区副本）")
    parser.add_argument("--device", default="auto", help="auto / cpu / mps / cuda")
    parser.add_argument("--output-dir", default=str(KEYPOINTS_DIR), help="缓存输出目录")
    parser.add_argument("--overwrite", action="store_true", help="已存在时重新提取（默认跳过）")
    parser.add_argument("--dry-run", action="store_true", help="只列出将处理的视频")
    parser.add_argument("--limit", type=int, default=None, help="每个场景最多处理多少个视频")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    model = resolve_model(args.model)
    id_filter = parse_ids(args.ids)
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    unknown = [s for s in scenes if s not in SCENES]
    if unknown:
        parser.error("未知场景: %s（可选: %s）" % (",".join(unknown), ",".join(SCENES)))

    plan = []
    for scene_name in scenes:
        videos = iter_videos(scene_name, SCENES[scene_name], id_filter)
        if args.limit is not None:
            videos = videos[:args.limit]
        plan.append((scene_name, videos))

    if args.dry_run:
        for scene_name, videos in plan:
            print("[%s] %d videos" % (scene_name, len(videos)))
            for video_id, path in videos:
                print("  video (%s)  %s" % (video_id, path.name))
        return 0

    if not os.path.exists(model):
        print("[错误] 找不到 YOLO-Pose 权重: %s" % model)
        print("       请放置 models/yolo26n-pose.pt，或用 --model 指定路径")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print("权重: %s" % model)
    print("设备: %s" % device)
    print("输出: %s" % out_dir)

    extractor = PoseExtractor(model)
    if device != "cpu":
        try:
            extractor.model.to(device)
        except Exception as exc:
            print("[警告] 切换设备 %s 失败（%s），回退 CPU" % (device, exc))
            device = "cpu"

    done = skipped = failed = 0
    frames_total = 0
    hit_total = 0
    try:
        for scene_name, videos in plan:
            print("\n[%s] %d videos" % (scene_name, len(videos)))
            for video_id, video_path in videos:
                kp_path, cf_path = cache_paths(scene_name, video_id, out_dir)
                if kp_path.exists() and cf_path.exists() and not args.overwrite:
                    skipped += 1
                    print("  跳过 video (%s)（缓存已存在）" % video_id)
                    continue

                print("  video (%s) ..." % video_id, flush=True)
                kpts, confs, hit = extract_video(video_path, extractor)
                if kpts is None or len(kpts) == 0:
                    failed += 1
                    print("  [警告] 无法读取或为空视频，跳过: %s" % video_path.name)
                    continue

                save_cache(kpts, confs, kp_path, cf_path)
                done += 1
                frames_total += len(kpts)
                hit_total += hit
                print("  -> %s  %s  %d 帧, 检出 %d (%.1f%%)"
                      % (kp_path.name, tuple(kpts.shape), len(kpts),
                         hit, 100.0 * hit / len(kpts)))
    except KeyboardInterrupt:
        print("\n[中断] 已完成的缓存均已落盘，可重跑续传（默认跳过已有文件）")

    print("\n完成: 新增 %d, 跳过 %d, 失败 %d" % (done, skipped, failed))
    if frames_total:
        print("总帧数 %d, 平均检出率 %.1f%%" % (frames_total, 100.0 * hit_total / frames_total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
