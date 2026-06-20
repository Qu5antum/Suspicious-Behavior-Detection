from collections import defaultdict, deque


class FallDetector:
    def __init__(
        self,
        history_size=10,
        aspect_ratio_threshold=1.2,
        height_drop_threshold=0.65,
        confirm_frames=5
    ):
        self.history_size = history_size
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.height_drop_threshold = height_drop_threshold
        self.confirm_frames = confirm_frames

        self.history = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

        self.fall_counter = defaultdict(int)

    def update(self, track_id, bbox):
        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        if height <= 0:
            return False

        aspect_ratio = width / height

        self.history[track_id].append({
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio
        })

        if len(self.history[track_id]) < 2:
            return False

        first = self.history[track_id][0]
        last = self.history[track_id][-1]

        height_ratio = last["height"] / first["height"]

        horizontal_state = (
            last["aspect_ratio"] > self.aspect_ratio_threshold
        )

        rapid_height_drop = (
            height_ratio < self.height_drop_threshold
        )

        if horizontal_state and rapid_height_drop:
            self.fall_counter[track_id] += 1
        else:
            self.fall_counter[track_id] = 0

        return self.fall_counter[track_id] >= self.confirm_frames

    def reset(self, track_id):
        if track_id in self.history:
            del self.history[track_id]

        if track_id in self.fall_counter:
            del self.fall_counter[track_id]