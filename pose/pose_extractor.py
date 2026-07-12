"""
pose_extractor.py

功能：
    使用 YOLO-Pose 提取人体关键点

输入：
    一张图片(frame)

输出：
    [
        {
            "bbox": ndarray(4,),
            "keypoints": ndarray(17,2),
            "score": float,
            "kpts_conf": ndarray(17,)   # [阶段0-fix5] 关键点置信度
        }
    ]

[阶段0-fix3] 跌倒后检测分裂修复:
    问题: 人倒地后 YOLO-Pose 将一个人检测成多个目标(头一个、身体一个)
    修复: 检测后对重叠度高的 bbox 进行合并, 取关键点置信度最高的目标

[阶段0-fix5] 物品误识为人修复 (P1):
    问题: conf=0.05 过低, 咖啡机/椅子/植物等物品被 YOLO-Pose 检测为"人",
          生成垃圾关键点, 导致后续 LSTM 误报跌倒 (coffee50/67 等).
    修复: 新增 _is_valid_person() 关键点质量门, 过滤"关键点稀疏+塌缩"的假人:
          - 有效关键点(conf>0.3)数量 >= min_valid_kpts(4)
          - 有效关键点空间跨度 >= min_spread(8px) 至少一个维度
          - 平均关键点置信度 >= min_mean_conf(0.3)
          真人 n_valid>=12 / spread>=20px / mean>=0.5; 物品 n_valid<=2 / spread<=2px / mean<=0.12,
          阈值取中间安全区, 保留跌倒后低置信度真人(仍有>=4个可信关键点).
"""

from ultralytics import YOLO
import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
modelPath = str(MODELS_DIR / "yolo11n-pose.pt")

os.environ['YOLO_CONFIG_DIR'] = str(PROJECT_ROOT / "models")


def _iou(boxA, boxB):
    """计算两个 bbox 的 IoU."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def _containment(boxA, boxB):
    """[阶段0-fix5] 计算 containment = intersection / min(areaA, areaB).

    当小框大部分在大框内时(分裂检测), IoU 可能<0.3 但 containment 接近1.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    min_area = min(areaA, areaB)
    return inter / min_area if min_area > 0 else 0.0


def _merge_overlapping_detections(persons, iou_threshold=0.3, containment_threshold=0.6):
    """[阶段0-fix3][阶段0-fix5] 合并重叠/包含的检测目标.

    跌倒后 YOLO-Pose 常将一个人检测成多个 bbox:
      - 重叠(IoU>0.3): 两个差不多大的框重叠
      - 包含(containment>0.6): 小框大部分在大框内 (头一个+身体一个, coffee50分裂)
    合并策略: 满足任一条件 → 保留 score 更高的, 丢弃另一个.
    """
    if len(persons) <= 1:
        return persons

    keep = [True] * len(persons)
    for i in range(len(persons)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(persons)):
            if not keep[j]:
                continue
            iou_v = _iou(persons[i]["bbox"], persons[j]["bbox"])
            cont_v = _containment(persons[i]["bbox"], persons[j]["bbox"])
            if iou_v > iou_threshold or cont_v > containment_threshold:
                # 保留 score 更高的
                if persons[j]["score"] > persons[i]["score"]:
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    return [persons[i] for i in range(len(persons)) if keep[i]]


def _is_valid_person(kpts, kpts_conf, bbox, score=1.0,
                     min_kp_conf=0.3, min_valid_kpts=4,
                     min_spread=8.0, min_mean_conf=0.3,
                     low_score_thres=0.15, low_score_min_spread=35.0):
    """[阶段0-fix5] 关键点质量门: 判定检测是否为真人.

    真人: 有效关键点>=12, 跨度>=40px, 平均置信度>=0.5, bbox score>=0.3
    物品: 有效关键点<=2, 跨度<=2px, 平均置信度<=0.12, bbox score<=0.1
          OR (低score + 小跨度) 墙画/电视等有8个中等关键点但 score<0.15 跨度<35px
    阈值取安全中间区, 兼容跌倒后低置信度真人(仍有>=4个可信关键点+合理跨度).

    Parameters
    ----------
    score : float            bbox 检测置信度 (低score+小跨度 → 物品误检)
    low_score_thres : float  低于此 score 时, 要求 spread >= low_score_min_spread
    low_score_min_spread : float  低 score 检测的最小关键点跨度

    Returns
    -------
    bool : True 保留, False 丢弃
    """
    if kpts_conf is None:
        return True  # 无关键点置信度信息 → 不能判定, 保留 (向后兼容)

    kc = np.asarray(kpts_conf, dtype=np.float32).reshape(-1)
    if kc.shape[0] == 0:
        return False

    # 1. 有效关键点数量
    valid_mask = kc > min_kp_conf
    n_valid = int(valid_mask.sum())
    if n_valid < min_valid_kpts:
        return False

    # 2. 有效关键点空间跨度 (至少一个维度 >= min_spread)
    kpts_arr = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
    valid_kpts = kpts_arr[valid_mask]
    if len(valid_kpts) == 0:
        return False
    kw = float(valid_kpts[:, 0].max() - valid_kpts[:, 0].min())
    kh = float(valid_kpts[:, 1].max() - valid_kpts[:, 1].min())
    spread = max(kw, kh)
    if spread < min_spread:
        return False

    # 3. 平均关键点置信度
    if float(kc.mean()) < min_mean_conf:
        return False

    # 4. [阶段0-fix5b] 低 bbox score + 小跨度 → 物品误检 (墙画/电视/海报等)
    #    真人(含跌倒后)即使 bbox score 低, 关键点跨度仍为全身(>=35px);
    #    墙画等物品 score<0.15 且关键点跨度<35px.
    if score < low_score_thres and spread < low_score_min_spread:
        return False

    return True


class PoseExtractor:
    """
    人体姿态提取器
    """

    def __init__(self, model_path=modelPath, conf=0.05):

        # 加载YOLO Pose模型
        self.model = YOLO(model_path)
        # [阶段0-fix2] conf 0.25→0.05: 人倒地后 YOLO-Pose 检测置信度大幅下降,
        # 默认 0.25 导致跌倒后检测率仅 5%, 降到 0.05 后提升至 55%
        self.conf = conf

    def extract(self, frame):
        """
        提取当前画面所有人的关键点

        Parameters
        ----------
        frame : ndarray
            OpenCV读取的图片

        Returns
        -------
        persons : list
        """

        persons = []

        # 推理 [阶段0-fix2] 使用降低后的 conf 阈值, [阶段0-fix3] 提高 iou 阈值抑制分裂检测
        results = self.model(frame, verbose=False, conf=self.conf, iou=0.5)

        result = results[0]

        # 没有人
        if result.boxes is None:
            return persons

        # bbox
        boxes = result.boxes.xyxy.cpu().numpy()

        # 检测置信度
        scores = result.boxes.conf.cpu().numpy()

        # 17关键点
        keypoints = result.keypoints.xy.cpu().numpy()

        # [阶段0-fix5] 关键点置信度 (供质量门判定)
        kpts_conf = None
        if result.keypoints.conf is not None:
            kpts_conf = result.keypoints.conf.cpu().numpy()

        for i, (bbox, score, kpts) in enumerate(zip(
                boxes,
                scores,
                keypoints)):

            # [阶段0-fix5] 关键点质量门: 过滤物品误检 (咖啡机/椅子/植物/墙画等)
            kc = kpts_conf[i] if kpts_conf is not None else None
            if not _is_valid_person(kpts, kc, bbox, score=float(score)):
                continue

            # === [REFORM] 置信度过滤：平均关键点置信度 < 0.3 → 丢弃 ===
            if kc is not None:
                avg_conf = float(np.mean(kc))
                if avg_conf < 0.3:
                    continue

            person = {
                "bbox": bbox,
                "keypoints": kpts,
                "score": float(score),
                "kpts_conf": kc,  # [阶段0-fix5] 暴露关键点置信度
            }

            persons.append(person)

        # [阶段0-fix3] 合并重叠检测 (跌倒后检测分裂修复)
        persons = _merge_overlapping_detections(persons, iou_threshold=0.3)

        return persons

    @staticmethod
    def build_feature(person):
        """
        把单个人的keypoints转成模型输入
        """

        kpts = person["keypoints"]  # (17,2)

        # 拉平
        feature = kpts.reshape(-1)  # (34,)

        return feature
