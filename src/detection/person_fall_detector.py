from collections import defaultdict
from collections import defaultdict


class FallDetector:
    def __init__(
        self,
        horizontal_threshold=70,
        confirm_frames=5,
    ):
        self.horizontal_threshold = horizontal_threshold
        self.confirm_frames = confirm_frames
        self.counter = defaultdict(int)

    def update(self, track_id, pose):

        if pose is None:
            self.counter[track_id] = 0
            return False

        keypoints = pose["keypoints"]

        required = [
            "nose",
            "left_hip",
            "right_hip",
            "left_ankle",
            "right_ankle",
        ]

        if not all(k in keypoints for k in required):
            self.counter[track_id] = 0
            return False

        nose = keypoints["nose"]

        hip_y = (keypoints["left_hip"]["y"] + keypoints["right_hip"]["y"]) / 2

        ankle_y = (keypoints["left_ankle"]["y"] + keypoints["right_ankle"]["y"]) / 2

        body_height = ankle_y - nose["y"]

        if body_height <= 0:
            self.counter[track_id] = 0
            return False

        horizontal = (abs(nose["y"] - hip_y) < self.horizontal_threshold)

        if horizontal:
            self.counter[track_id] += 1
        else:
            self.counter[track_id] = 0

        return self.counter[track_id] >= self.confirm_frames

    def reset(self, track_id):
        self.counter.pop(track_id, None)