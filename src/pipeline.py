import cv2
from .detector import PersonDetector
from .tracker import PersonTracker
from .trajectory import TrajectoryManager
from .behavior import BehaviorAnalyzer


class VideoPipeline:
    """
    Video işleme sınıfı: insan tespiti, izleme, yörünge kaydı ve davranış analizi
    """
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        self.detector = PersonDetector()
        self.tracker = PersonTracker()
        self.trajectory_manager = TrajectoryManager()
        self.behavior_analyzer = BehaviorAnalyzer()

    def process(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(frame, detections)
            self.trajectory_manager.update(tracks)

            analyzed = []
            for t in tracks:
                traj = self.trajectory_manager.get(t["id"])
                behavior = self.behavior_analyzer.analyze(t["id"], traj)

                t["behavior"] = behavior
                analyzed.append(t)

            self._draw(frame, analyzed)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
    """
    draw metodu, nesnelerin sınırlayıcı kutusunu, kimliklerini ve yörüngelerini çizerek şüpheli davranışları vurgular
    """
    def _draw(self, frame, tracks):
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["id"]
            behavior = t["behavior"]

            color = (0, 255, 0)

            if behavior["loitering"]:
                color = (0, 0, 255) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"ID {track_id}"

            if behavior["loitering"]:
                text += " LOITERING"

            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
                
        for traj in self.trajectory_manager.get_all().values():
            for i in range(1, len(traj)):
                cv2.line(frame, traj[i - 1], traj[i], (255, 0, 0), 2)