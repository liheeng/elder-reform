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
            keypoints):

        # ---------------------------------
        # 基本信息
        # ---------------------------------

        self.id = track_id

        self.bbox = bbox

        self.keypoints = keypoints

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
            keypoints):
        """
        更新目标
        """

        self.bbox = bbox

        self.keypoints = keypoints

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
            "missing": self.missing,
            "age": self.age,
            "hit_count": self.hit_count
        }

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