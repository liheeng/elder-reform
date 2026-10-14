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
        "score": float,
        "kpts_conf": ndarray(17,)
    }
]

--------------------------------------------------

输出:

tracks:

[
    {
        "id": 1,
        "bbox": ...,
        "keypoints": ...,
        "kpts_conf": ...,
        "score": ...
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

    匹配规则:
        1. 主匹配: 与最近质心的距离 < distance_threshold
        2. 兜底匹配: 质心超出阈值, 但 IoU >= iou_threshold
           (人跌倒时 bbox 由竖长变横宽, 质心大幅下移会被主匹配漏掉)

    轨道确认:
        hit_count < min_hits 的轨道视为"未确认", 只给
        max_missing_unconfirmed 帧的丢失容忍, 超时即移除 —— 用于淘汰
        单帧垃圾检测产生的伪轨道.
    """

    def __init__(
            self,
            max_missing=30,
            distance_threshold=100,
            iou_threshold=0.3,
            min_hits=2,
            max_missing_unconfirmed=15):

        # 当前Track列表
        self.tracks = []

        # 下一可用ID
        self.next_id = 1

        # 最大允许丢失帧数
        self.max_missing = max_missing

        # 中心点匹配距离阈值
        self.distance_threshold = distance_threshold

        # 兜底匹配的 IoU 阈值 (质心超阈值但框仍重叠时认定同一目标)
        self.iou_threshold = iou_threshold

        # 轨道确认所需的最小命中次数
        self.min_hits = min_hits

        # 未确认轨道的丢失容忍帧数
        self.max_missing_unconfirmed = max_missing_unconfirmed

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

    @staticmethod
    def iou(bbox1, bbox2):
        """
        计算两个bbox的 IoU
        """

        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
        area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])

        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

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
            keypoints=person["keypoints"],
            kpts_conf=person.get("kpts_conf"),
            score=float(person.get("score", 1.0))
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

                if track.is_lost(
                        self.max_missing,
                        self.min_hits,
                        self.max_missing_unconfirmed):

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

            # ----------------------------------------------
            # 主匹配: 最近质心
            # ----------------------------------------------

            best_distance = float("inf")

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

            matched = (
                best_track is not None
                and
                best_distance < self.distance_threshold
            )

            # ----------------------------------------------
            # 兜底匹配: 质心超出阈值, 但框仍有明显重叠
            #
            # 人跌倒时 bbox 由"竖长"突变为"横宽", 质心大幅下移, 距离很容易
            # 超过 distance_threshold 而匹配失败 → 新建 ID → 又要 30 帧预热,
            # 而跌倒恰恰发生在这段盲区内. 此时用 IoU 仍能判定为同一目标.
            # ----------------------------------------------

            if not matched:

                best_iou = self.iou_threshold

                for track in self.tracks:

                    if track.id in matched_tracks:
                        continue

                    iou_value = self.iou(
                        person["bbox"],
                        track.bbox
                    )

                    if iou_value >= best_iou:

                        best_iou = iou_value
                        best_track = track
                        matched = True

            # ----------------------------------------------
            # 匹配成功
            # ----------------------------------------------

            if matched:

                best_track.update(
                    person["bbox"],
                    person["keypoints"],
                    person.get("kpts_conf"),
                    float(person.get("score", 1.0))
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

            if track.is_lost(
                    self.max_missing,
                    self.min_hits,
                    self.max_missing_unconfirmed):

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