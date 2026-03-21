import math


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