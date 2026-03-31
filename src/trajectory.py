from collections import defaultdict
import math
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


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
    
 
class TrajectoryAnalyzer:
    """Insanin tekrarlan yol tespiti sinifi"""
    def __init__(self, similarity_threshold=50, min_movement=20, window_size=20):
        self.similarity_threshold = similarity_threshold
        self.min_movement = min_movement
        self.window_size = window_size
        self.history = {}

    def trajectory_length(self, traj):
        total = 0
        for i in range(1, len(traj)):
            total += euclidean(traj[i], traj[i-1])
        return total

    def normalize(self, traj):
        cx = sum(p[0] for p in traj) / len(traj)
        cy = sum(p[1] for p in traj) / len(traj)
        return [(x - cx, y - cy) for x, y in traj]

    def update(self, track_id, trajectory):
        if len(trajectory) < self.window_size:
            return False

        current = trajectory[-self.window_size:]

        if self.trajectory_length(current) < self.min_movement:
            return False

        current = self.normalize(current)

        if track_id not in self.history:
            self.history[track_id] = []

        repeated = False

        for past_traj in self.history[track_id]:
            distance, _ = fastdtw(current, past_traj, dist=euclidean)
            if distance < self.similarity_threshold:
                repeated = True
                break

        self.history[track_id].append(current)

        if len(self.history[track_id]) > 5:
            self.history[track_id].pop(0)

        return repeated