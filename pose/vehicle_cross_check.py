"""
vehicle_cross_check.py

用途: 剔除"被 pose 模型当成人的车辆"（典型: 带车罩的停放电动车 —— 红头盔像头、
蓝车罩像躯干、轮子像腿，pose 模型会给出 17 个高置信关键点，质量门无法识别）。

判据（车类交叉验证, 2026-10 验证见 docs/tuning-log.md）:

    人框被车类框包含(包含度 >= cont_min)  且  车分 - 人分 > margin  -> 丢弃

为什么用"置信度差"而不是更直觉的 IoU / 单纯比大小:
    "人骑车"同样满足"车框包含人框"，甚至车分也略高于人分，用 IoU 或
    "车分 > 人分" 都会把真人骑手删掉（实测会误杀 138 个骑手框中的若干）。
    实测两类的差值分布可分:
        车被当成人:  车分 - 人分 = +0.08 ~ +0.82
        人骑车:      车分 - 人分 = -0.54 ~ +0.19
    margin 取 0.20 时，离线分析显示骑手误杀为 0、pd4 电动车清除 152/158 (96%)。

!! 端到端实测未通过，当前不得启用 !!
    难例集 105 个已标注正样本上，启用后新增漏检 9 帧 (8.6%)；
    pd4 f210/213/214/253 的人框由 2 个降到 0 个 —— 连躺在地上的老人一起被删。
    原因: 老人躺在停放电动车旁边，检测器的 motorcycle 框把他一并框住，于是他也
    满足"被车框包含 + 车分显著更高"。
    离线分析误报"0 误杀"是因为那几帧没有人工标注、不在其覆盖范围内。
    结论: 该判据在"人躺/坐在车辆旁"这一常见状况下不成立，需换判据或转 NON_PERSON。

局限: 只解决"车"这一类，无法替代 NON_PERSON（海报/墙面等误检与此无关）；
      检测器自身漏检车时（夜间/远距）该判据失效。
"""

import numpy as np

from pose.pose_extractor import _containment
from utils.device import get_device

VEHICLE_CLASS_NAMES = ("bicycle", "car", "motorcycle", "bus", "truck")

DEFAULT_MODEL = None  # 由调用方给路径；见 PoseExtractor 的用法


class VehicleCrossChecker:
    """给一批 pose 人框 + 一帧图像，返回剔除车辆误检后的人框。"""

    def __init__(self, model_path, conf=0.15, cont_min=0.6, margin=0.20, device=None):
        from ultralytics import YOLO        # 延迟导入: 不用该功能时不必付代价

        self.device = device or get_device()
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.cont_min = cont_min
        self.margin = margin
        self.vehicle_class_ids = [i for i, n in self.model.names.items()
                                  if n in VEHICLE_CLASS_NAMES]
        if not self.vehicle_class_ids:
            raise ValueError("检测模型里找不到车类: %s" % (VEHICLE_CLASS_NAMES,))

    def filter(self, persons, frame):
        """返回应保留的人框（可能为空列表）。"""
        if not persons:
            return persons

        result = self.model(frame, verbose=False, conf=self.conf, iou=0.5,
                            classes=self.vehicle_class_ids, device=self.device)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return persons

        v_boxes = boxes.xyxy.cpu().numpy()
        v_scores = boxes.conf.cpu().numpy()

        kept = []
        for person in persons:
            p_box = np.asarray(person["bbox"], dtype=np.float32)
            p_score = float(person.get("score", 0.0))
            is_vehicle = False
            for v_box, v_score in zip(v_boxes, v_scores):
                if (_containment(p_box, v_box) >= self.cont_min
                        and float(v_score) - p_score > self.margin):
                    is_vehicle = True
                    break
            if not is_vehicle:
                kept.append(person)
        return kept

    def __repr__(self):
        return ("VehicleCrossChecker(conf=%.2f, cont_min=%.2f, margin=%.2f)"
                % (self.conf, self.cont_min, self.margin))
