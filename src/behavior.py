import math
from collections import defaultdict


class BehaviorAnalyzer:
    """
    Bir cismin yörüngesi boyunca davranışını analiz etmek için kullanılan bir sınıf
    """
    def __init__(self):
        self.loitering_threshold_time = 30  
        self.movement_threshold = 20        

    def analyze(self, track_id, trajectory):
        result = {
            "loitering": False,
            "total_distance": 0
        }

        if len(trajectory) < 2:
            return result

        total_distance = 0

        for i in range(1, len(trajectory)):
            x1, y1 = trajectory[i - 1]
            x2, y2 = trajectory[i]

            dist = math.hypot(x2 - x1, y2 - y1)
            total_distance += dist

        result["total_distance"] = total_distance
        
        """
        Analiz yöntemi, toplam hareket mesafesini hesaplar ve bir nesnenin az hareket edip uzun süre aynı yerde kalması durumunda,
        nesneyi (loitering) olarak tanımlar.
        """
        if total_distance < self.movement_threshold and len(trajectory) > self.loitering_threshold_time:
            result["loitering"] = True

        return result
    

class LookingAroundAnalyzer:
    """
    Insanin bas cevirme tespit sinifi
    """
    def __init__(self, threshold=0.35, history_size=15, min_switches=3):
        self.threshold = threshold
        self.history_size = history_size
        self.min_switches = min_switches
        self.yaw_history = defaultdict(list)

    def update(self, track_id, yaw):
        if yaw is None:
            return False

        history = self.yaw_history[track_id]
        history.append(yaw)
        if len(history) > self.history_size:
            history.pop(0)

        switches = 0
        for i in range(1, len(history)):
            if abs(history[i] - history[i-1]) > self.threshold:
                switches += 1

        return switches >= self.min_switches