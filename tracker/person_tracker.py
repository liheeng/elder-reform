"""
person_tracker.py

功能：
    多目标跟踪器(Centroid Tracking)

作用：
    为每个人分配固定ID

例如：

Frame1:
    PersonA -> ID=1
    PersonB -> ID=2

Frame2:
    PersonA -> ID=1
    PersonB -> ID=2

--------------------------------------------------

输入:

[
    {
        "bbox": ndarray(4,),
        "keypoints": ndarray(17,2),
        "score": float
    }
]

--------------------------------------------------

输出:

tracks:

[
    {
        "id": 1,
        "bbox": ...,
        "keypoints": ...
    }
]

removed_ids:

[
    3,
    7
]

表示：

ID=3
ID=7

已经被删除
"""

import numpy as np

from tracker.track import Track


class PersonTracker:
    """
    多目标跟踪器(Centroid Tracking)
    """

    def __init__(
            self,
            max_missing=30,
            distance_threshold=100):

        # 当前Track列表
        self.tracks = []

        # 下一可用ID
        self.next_id = 1

        # 最大允许丢失帧数
        self.max_missing = max_missing

        # 中心点匹配距离阈值
        self.distance_threshold = distance_threshold

    # ==================================================
    # 工具函数
    # ==================================================

    def bbox_center(self, bbox):
        """
        bbox:

            [x1,y1,x2,y2]
        """

        x1, y1, x2, y2 = bbox

        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2
        ], dtype=np.float32)

    def distance(self, bbox1, bbox2):
        """
        计算两个bbox中心点距离
        """

        c1 = self.bbox_center(bbox1)
        c2 = self.bbox_center(bbox2)

        return np.linalg.norm(c1 - c2)

    # ==================================================
    # Track管理
    # ==================================================

    def create_track(self, person):
        """
        创建新Track
        """

        track = Track(
            track_id=self.next_id,
            bbox=person["bbox"],
            keypoints=person["keypoints"]
        )

        self.next_id += 1

        self.tracks.append(track)

        return track

    def get_tracks(self):
        """
        导出所有Track
        """

        return [
            track.to_dict()
            for track in self.tracks
        ]

    # ==================================================
    # 核心更新
    # ==================================================

    def update(self, persons):
        """
        Parameters
        ----------
        persons : list

            PoseExtractor输出结果

        Returns
        -------
        tracks : list

        removed_ids : list
        """

        removed_ids = []

        # ==================================================
        # 当前帧无人
        # ==================================================

        if len(persons) == 0:

            alive_tracks = []

            for track in self.tracks:

                track.mark_missing()

                if track.missing > self.max_missing:

                    removed_ids.append(
                        track.id
                    )

                else:

                    alive_tracks.append(
                        track
                    )

            self.tracks = alive_tracks

            return self.get_tracks(), removed_ids

        # ==================================================
        # 第一帧
        # ==================================================

        if len(self.tracks) == 0:

            for person in persons:
                self.create_track(person)

            return self.get_tracks(), []

        # ==================================================
        # 已匹配Track
        # ==================================================

        matched_tracks = set()

        # ==================================================
        # 遍历当前检测结果
        # ==================================================

        for person in persons:

            best_track = None

            best_distance = float("inf")

            # ----------------------------------------------
            # 寻找最近Track
            # ----------------------------------------------

            for track in self.tracks:

                if track.id in matched_tracks:
                    continue

                dist = self.distance(
                    person["bbox"],
                    track.bbox
                )

                if dist < best_distance:

                    best_distance = dist
                    best_track = track

            # ----------------------------------------------
            # 匹配成功
            # ----------------------------------------------

            if (
                best_track is not None
                and
                best_distance < self.distance_threshold
            ):

                best_track.update(
                    person["bbox"],
                    person["keypoints"]
                )

                matched_tracks.add(
                    best_track.id
                )

            # ----------------------------------------------
            # 创建新Track
            # ----------------------------------------------

            else:

                new_track = self.create_track(
                    person
                )

                # 新Track视为已匹配
                matched_tracks.add(
                    new_track.id
                )

        # ==================================================
        # 更新未匹配Track
        # ==================================================

        alive_tracks = []

        for track in self.tracks:

            if track.id not in matched_tracks:

                track.mark_missing()

            if track.missing > self.max_missing:

                removed_ids.append(
                    track.id
                )

            else:

                alive_tracks.append(
                    track
                )

        self.tracks = alive_tracks

        return self.get_tracks(), removed_ids

    # ==================================================
    # 重置
    # ==================================================

    def reset(self):
        """
        清空Tracker
        """

        self.tracks.clear()

        self.next_id = 1