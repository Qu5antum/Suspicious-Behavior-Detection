import math
from collections import defaultdict, deque
import cv2

from src.database.db import SessionLocal
from src.events.event_manager import EventManager

db_session = SessionLocal()


class FallDetector:
    def __init__(
        self,
        history_size=15,
        angle_threshold=35,
        velocity_threshold=40,
        bbox_ratio_threshold=1.15,
        confirm_frames=5,
    ):
        """
        A fall detection system combining YOLO and skeleton keys tracks
        human joints (like the shoulders, hips, and knees). By analyzing the angles
        and speeds of these points, the system distinguishes between a standing
        posture and a fall
        """
        self.history = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self.counter = defaultdict(int)
        self.angle_threshold = angle_threshold
        self.velocity_threshold = velocity_threshold
        self.bbox_ratio_threshold = bbox_ratio_threshold
        self.confirm_frames = confirm_frames
        self.saved_events = defaultdict(bool)

        self.event_manager = EventManager(session=db_session)

    def update(self, track_id, pose, frame) -> bool | None:
        if pose is None:
            self.reset(track_id)
            return False

        kp = pose["keypoints"]

        required = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ]

        if not all(name in kp for name in required):
            self.reset(track_id)
            return False

        ls = kp["left_shoulder"]
        rs = kp["right_shoulder"]
        lh = kp["left_hip"]
        rh = kp["right_hip"]

        shoulder_x = (ls["x"] + rs["x"]) / 2
        shoulder_y = (ls["y"] + rs["y"]) / 2

        hip_x = (lh["x"] + rh["x"]) / 2
        hip_y = (lh["y"] + rh["y"]) / 2

        dx = hip_x - shoulder_x
        dy = hip_y - shoulder_y

        angle = abs(math.degrees(math.atan2(dy, dx)))

        horizontal = (
            angle < self.angle_threshold
            or angle > (180 - self.angle_threshold)
        )

        bbox = pose["bbox"]

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        aspect_ratio = (
            width / height
            if height > 0 else 0
        )

        horizontal_bbox = (
            aspect_ratio > self.bbox_ratio_threshold
        )

        self.history[track_id].append(
            {
                "hip_y": hip_y,
                "angle": angle,
                "ratio": aspect_ratio,
            }
        )

        fast_drop = False

        if len(self.history[track_id]) >= 2:
            previous = self.history[track_id][-2]

            velocity = hip_y - previous["hip_y"]

            fast_drop = (
                velocity > self.velocity_threshold
            )

        score = 0

        if horizontal:
            score += 3

        if horizontal_bbox:
            score += 2

        if fast_drop:
            score += 3

        if score >= 5:
            self.counter[track_id] += 1
        else:
            self.counter[track_id] = 0

        filename = f"events_file/frame_{track_id}.jpg"

        x1, y1, x2, y2 = pose["bbox"]

        crop = frame[y1:y2, x1:x2]

        cv2.imwrite(filename, crop)

        fall_detected = (
            self.counter[track_id] >= self.confirm_frames
        )

        if fall_detected and not self.saved_events[track_id]:

            self.event_manager.add_event(
                person_id=track_id,
                bag_id=None,
                reason="Fall detected",
                image_path=filename,
                event_type="FALL"
            )

            self.saved_events[track_id] = True

        elif not fall_detected:
            self.saved_events[track_id] = False

            return (
                self.counter[track_id]
                >= self.confirm_frames
            )

    def reset(self, track_id):

        self.history.pop(track_id, None)
        self.counter.pop(track_id, None)