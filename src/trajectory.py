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
    """
    Insanin tekrarlan yol tespiti sinifi
    """
    def __init__(self, similarity_treshold=50):
        self.similarity_treshhold = similarity_treshold
        self.history = {}

    def update(self, track_id, trajectory):
        repated = False

        if len(trajectory) < 5:
            return repated
        
        if track_id not in self.history:
            self.history[track_id] = []

        for past_traj in self.history[track_id]:
            distance, _ = fastdtw(trajectory, past_traj, dist=euclidean)
            if distance < self.similarity_treshhold:
                repated = True
                break

        self.history[track_id].append(list(trajectory))

        if len(self.history[track_id]) > 5:
            self.history[track_id].pop(0)

        return repated