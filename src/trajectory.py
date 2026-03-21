from collections import defaultdict
import math


class TrajectoryManager:
    """
    Nesne yörüngelerini saklamak ve güncellemek için kullanılan bir sınıf
    """
    def __init__(self, max_length=50):
        self.trajectories = defaultdict(list)
        self.max_length = max_length

    def update(self, tracks):
        for t in tracks:
            track_id = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            self.trajectories[track_id].append((cx, cy))

            if len(self.trajectories[track_id]) > self.max_length:
                self.trajectories[track_id].pop(0)

    def get(self, track_id):
        return self.trajectories.get(track_id, [])

    def get_all(self):
        return self.trajectories