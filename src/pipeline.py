import cv2
from .person_detector import PersonDetector, PersonTracker, HeadPoseEstimator, FaceDetector
from .trajectory import TrajectoryManager, TrajectoryAnalyzer
from .behavior import BehaviorAnalyzer, LookingAroundAnalyzer


class VideoPipeline:
    """
    Video pipeline:
    - Person detection + tracking
    - Trajectory recording
    - Behavior analysis: loitering, repeated path, looking around
    - Optional face detection + head pose for looking around
    """

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Video source açılamadı")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.detector = PersonDetector()
        self.tracker = PersonTracker()
        self.trajectory_manager = TrajectoryManager()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer(similarity_threshold=50)

        self.face_detector = FaceDetector()
        self.head_pose = HeadPoseEstimator()
        self.looking_around = LookingAroundAnalyzer()

        self.frame_count = 0

    def process(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            frame_h, frame_w, _ = frame.shape

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(frame, detections)
            self.trajectory_manager.update(tracks)

            analyzed_tracks = []

            for t in tracks:
                bbox = t["bbox"]
                track_id = t["id"]

                x1, y1, x2, y2 = bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)
                bbox = [x1, y1, x2, y2]

                traj = self.trajectory_manager.get(track_id)
                behavior = self.behavior_analyzer.analyze(track_id, traj)
                behavior.setdefault("loitering", False)
                behavior.setdefault("repeated_path", False)
                behavior.setdefault("looking_around", False)

                behavior["repeated_path"] = self.trajectory_analyzer.update(track_id, traj)

                head_y2 = y1 + int((y2 - y1) * 0.4)
                head_bbox = [x1, y1, x2, head_y2]

                yaw = None
                if self.frame_count % 3 == 0:
                    yaw = self.head_pose.estimate(head_bbox, frame_w)

                behavior["looking_around"] = self.looking_around.update(track_id, yaw)

                t["bbox"] = bbox
                t["behavior"] = behavior
                analyzed_tracks.append(t)

            faces = self.face_detector.detect(frame)
            for face_bbox in faces:
                face_yaw = None
                if self.frame_count % 3 == 0:
                    face_yaw = self.head_pose.estimate(face_bbox, frame_w)
                face_looking = self.looking_around.update(track_id=id(face_bbox), yaw=face_yaw)

                x1, y1, x2, y2 = face_bbox
                color = (0, 0, 255) if face_looking else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = "LOOKING AROUND" if face_looking else "NORMAL"
                cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self._draw(frame, analyzed_tracks)

            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _draw(self, frame, tracks):
        """Draw bounding boxes, trajectories, and suspicious scores"""
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            track_id = t["id"]
            behavior = t["behavior"]

            score = 0
            if behavior.get("loitering"):
                score += 2
            if behavior.get("repeated_path"):
                score += 3
            if behavior.get("looking_around"):
                score += 2

            color = (0, 255, 0)
            if score >= 5:
                color = (0, 0, 255)
            elif score >= 3:
                color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"ID {track_id} | Score: {score}"
            if behavior.get("loitering"):
                text += " LOITERING"
            if behavior.get("repeated_path"):
                text += " REPEATED"
            if behavior.get("looking_around"):
                text += " LOOKING AROUND"
            cv2.putText(frame, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for traj in self.trajectory_manager.get_all().values():
            for i in range(1, len(traj)):
                cv2.line(frame, traj[i-1], traj[i], (255, 0, 0), 2)
