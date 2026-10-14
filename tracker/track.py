"""
track.py

功能：
    保存一个人的跟踪信息

例如：

ID=1
bbox
keypoints

持续更新

--------------------------------------------------

Track维护信息：

id
bbox
keypoints
kpts_conf
    关键点置信度 (17,)，供 FeatureBuilderV2 构建 conf 维使用
score
    bbox 检测置信度

missing
    连续丢失帧数

age
    Track总存在帧数

hit_count
    成功匹配次数
"""

import numpy as np


class Track:
    """
    单目标Track
    """

    def __init__(
            self,
            track_id,
            bbox,
            keypoints,
            kpts_conf=None,
            score=1.0):

        # ---------------------------------
        # 基本信息
        # ---------------------------------

        self.id = track_id

        self.bbox = bbox

        self.keypoints = keypoints

        # 关键点置信度 (17,) 与 bbox 置信度
        # 必须跟着 Track 一起走: FeatureBuilderV2 用 confs 构建 17 个 conf 维,
        # 丢失后会静默退化成常量 1.0, 与训练特征分布不一致.
        self.kpts_conf = kpts_conf

        self.score = score

        # ---------------------------------
        # 跟踪状态
        # ---------------------------------

        # 连续丢失帧数
        self.missing = 0

        # 存活总帧数
        self.age = 1

        # 成功匹配次数
        self.hit_count = 1

    # ==================================================
    # 更新Track
    # ==================================================

    def update(
            self,
            bbox,
            keypoints,
            kpts_conf=None,
            score=1.0):
        """
        更新目标
        """

        self.bbox = bbox

        self.keypoints = keypoints

        self.kpts_conf = kpts_conf

        self.score = score

        # 找回目标
        self.missing = 0

        # 更新统计
        self.age += 1

        self.hit_count += 1

    # ==================================================
    # 丢失
    # ==================================================

    def mark_missing(self):
        """
        当前帧未匹配
        """

        self.missing += 1

        self.age += 1

    # ==================================================
    # 中心点
    # ==================================================

    def center(self):
        """
        返回bbox中心点

        bbox:

            [x1,y1,x2,y2]
        """

        x1, y1, x2, y2 = self.bbox

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        return np.array(
            [cx, cy],
            dtype=np.float32
        )

    # ==================================================
    # 导出
    # ==================================================

    def to_dict(self):
        """
        导出Track信息
        """

        return {
            "id": self.id,
            "bbox": self.bbox,
            "keypoints": self.keypoints,
            "kpts_conf": self.kpts_conf,
            "score": self.score,
            "missing": self.missing,
            "age": self.age,
            "hit_count": self.hit_count
        }

    # ==================================================
    # 是否应被移除
    # ==================================================

    def is_lost(self, max_missing, min_hits, max_missing_unconfirmed):
        """判定该 Track 是否已丢失到应被移除。

        未被确认的轨道（命中次数 < min_hits）只给较短的容忍窗口：
        它们多半是单帧垃圾检测，长期挂着会让一个过期 bbox 去抢占后续匹配。
        已确认的轨道仍按 max_missing 容忍遮挡。
        """

        if self.hit_count < min_hits:
            return self.missing > max_missing_unconfirmed

        return self.missing > max_missing

    # ==================================================
    # 调试
    # ==================================================

    def __repr__(self):

        return (
            f"Track("
            f"id={self.id}, "
            f"missing={self.missing}, "
            f"age={self.age}, "
            f"hits={self.hit_count})"
        )