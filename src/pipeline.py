import cv2
import platform

import time 
import os

from .person_detector import PersonDetector, HeadPoseEstimator, FaceDetector
from .trajectory import TrajectoryManager, TrajectoryAnalyzer
from .behavior import BehaviorAnalyzer, LookingAroundAnalyzer

from .object_tracking import (
    BagDetector,
    BagTracker,
    OwnershipAnalyzer,
    SuspicionState
)

class Color:
    GREEN = (34, 197,  94)
    ORANGE = (34, 165, 255)
    RED = (60, 60, 220)
    CYAN = (220, 220, 0)
    BLUE = (220, 120, 40)
    WHITE = (240, 240, 240)
    GRAY = (120, 120, 120)

STATE_STYLE = {
    SuspicionState.NORMAL: {"color": Color.GREEN, "tag": "", "thickness": 1},
    SuspicionState.WARNING: {"color": Color.ORANGE, "tag": "WARNING", "thickness": 2},
    SuspicionState.ALERT: {"color": Color.RED, "tag": "! ALERT", "thickness": 3},
}

SCORE_LOITERING = 2
SCORE_REPEATED_PATH = 3
SCORE_LOOKING_AROUND = 2
SCORE_ABANDONED_ALERT = 4
SCORE_ABANDONED_WARN = 1

THRESHOLD_ORANGE = 3
THRESHOLD_RED = 6
ABANDONED_PERSON_DIST = 0.08 * 1280


class VideoPipeline:
    def __init__(self, source=0):
        self.cap = self._open_capture(source)

        self.detector = PersonDetector()
        self.trajectory_manager = TrajectoryManager()
        self.behavior_analyzer = BehaviorAnalyzer()
        self.trajectory_analyzer = TrajectoryAnalyzer()

        self.face_detector = FaceDetector()
        self.head_pose = HeadPoseEstimator()
        self.looking_around = LookingAroundAnalyzer()

        self.object_detector = BagDetector()
        self.object_tracker = BagTracker()
        self.abandoned_analyzer = None

        self.frame_count = 0
        self._active_track_ids = set()


    def _open_capture(self, source) -> cv2.VideoCapture:
        is_windows = platform.system() == "Windows"
        is_file = isinstance(source, str)

        backends = (
            [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            if is_windows and not is_file
            else [cv2.CAP_ANY]
        )

        for backend in backends:
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                print(f"[Pipeline] Backend: {backend}")
                self._configure_cap(cap)
                return cap
            cap.release()

        raise ValueError(f"Açık kaynak kodlu hale getirme başarısız oldu: {source}")

    @staticmethod
    def _configure_cap(cap: cv2.VideoCapture):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def process(self):
        last_save_time = 0

        save_dir = "screenshots"
        os.makedirs(save_dir, exist_ok=True)

        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                continue

            self.frame_count += 1
            frame_h, frame_w = frame.shape[:2]

            if self.abandoned_analyzer is None:
                self.abandoned_analyzer = OwnershipAnalyzer(
                    frame_w=frame_w,
                    frame_h=frame_h
                )

            # Insan takibi
            tracks = self.detector.detect_and_track(frame)

            self.trajectory_manager.update(tracks)
            self._flush_lost_tracks({t["id"] for t in tracks})

            # Nesneler
            obj_dets = self.object_detector.detect(frame)
            objects  = self.object_tracker.update(obj_dets)

            abandoned_results = self.abandoned_analyzer.update(
                objects, tracks, self.frame_count
            )

            analyzed_tracks = [
                self._analyze_person(t, frame, frame_w, frame_h, abandoned_results)
                for t in tracks
            ]

            self._draw_objects(frame, abandoned_results)
            self._draw_persons(frame, analyzed_tracks)
            self._draw_trajectories(frame)
            self._draw_hud(frame, abandoned_results)
              

            """current_time = time.time()
            if current_time - last_save_time >= 1.0:
                filename = os.path.join(
                    save_dir,
                    f"screenshot_{int(time.time())}.jpg"
                )
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                last_save_time = current_time"""

            cv2.imshow("Surveillance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _analyze_person(self, track, frame, frame_w, frame_h, abandoned_results):
        tid = track["id"]
        x1, y1, x2, y2 = track["bbox"]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame_w, x2)
        y2 = min(frame_h, y2)

        traj = self.trajectory_manager.get(tid)

        behavior = self.behavior_analyzer.analyze(tid, traj)

        behavior["repeated_path"] = self.trajectory_analyzer.update(tid, traj)

        behavior["looking_around"], face_bbox = self._estimate_head_pose(
            frame, tid, x1, y1, x2, y2
        )

        behavior["abandoned_state"] = self._nearest_abandoned_state(
            x1, y1, x2, y2, abandoned_results
        )

        track["behavior"] = behavior
        track["face_bbox"] = face_bbox

        return track
    
    def _estimate_head_pose(self, frame, track_id, x1, y1, x2, y2):
        head_y2 = y1 + int((y2 - y1) * 0.4)
        head_crop = frame[y1:head_y2, x1:x2]

        if head_crop is None or head_crop.size == 0:
            return self.looking_around.update(track_id, None), None

        faces = self.face_detector.detect(head_crop)
        if not faces:
            return self.looking_around.update(track_id, None), None

        fx1, fy1, fx2, fy2 = faces[0]
        face_bbox = [x1 + fx1, y1 + fy1, x1 + fx2, y1 + fy2]
        yaw = self.head_pose.estimate([x1, y1, x2, head_y2], face_bbox)
        looking = self.looking_around.update(track_id, yaw)
        return looking, face_bbox

    def _nearest_abandoned_state(self, x1, y1, x2, y2, abandoned_results):
        px, py = (x1 + x2) / 2, (y1 + y2) / 2
        priority = {
            SuspicionState.NORMAL: 0,
            SuspicionState.WARNING: 1,
            SuspicionState.ALERT: 2,
        }
        best = SuspicionState.NORMAL

        for obj in abandoned_results:
            if obj["state"] == SuspicionState.NORMAL:
                continue
            ox = (obj["bbox"][0] + obj["bbox"][2]) / 2
            oy = (obj["bbox"][1] + obj["bbox"][3]) / 2
            if ((px - ox) ** 2 + (py - oy) ** 2) ** 0.5 < ABANDONED_PERSON_DIST:
                if priority[obj["state"]] > priority[best]:
                    best = obj["state"]

        return best

    def _flush_lost_tracks(self, current_ids: set[int]):
        for tid in self._active_track_ids - current_ids:
            self.looking_around.reset(tid)
        self._active_track_ids = current_ids

    @staticmethod
    def _risk_score(behavior: dict) -> int:
        score = 0
        if behavior.get("loitering"): score += SCORE_LOITERING
        if behavior.get("repeated_path"): score += SCORE_REPEATED_PATH
        if behavior.get("looking_around"): score += SCORE_LOOKING_AROUND

        ab = behavior.get("abandoned_state", SuspicionState.NORMAL)
        if ab == SuspicionState.ALERT: score += SCORE_ABANDONED_ALERT
        elif ab == SuspicionState.WARNING: score += SCORE_ABANDONED_WARN
        return score

    @staticmethod
    def _score_color(score: int):
        if score >= THRESHOLD_RED: return Color.RED
        if score >= THRESHOLD_ORANGE: return Color.ORANGE
        return Color.GREEN

    def _draw_objects(self, frame, abandoned_results):
        for obj in abandoned_results:
            x1, y1, x2, y2 = obj["bbox"]
            style = STATE_STYLE[obj["state"]]
            color = style["color"]
            thickness = style["thickness"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{obj['class'].upper()} #{obj['id']}"
            if style["tag"]:
                label += f"  {style['tag']}"
            cv2.putText(frame, label, (x1, y1 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

            if obj["owner_nearby"]:
                detail = "owner nearby"
            elif obj["had_owner"]:
                detail = f"owner gone: {obj['gone_frames']}f"
            else:
                detail = f"no owner  static:{obj['static_frames']}f"

            cv2.putText(frame, detail, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            if obj["state"] == SuspicionState.ALERT:
                alpha = abs(self.frame_count % 30 - 15) / 15 
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), Color.RED, 4)
                cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)

    def _draw_persons(self, frame, tracks):
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            behavior = t["behavior"]
            face_bbox = t.get("face_bbox")
            score = self._risk_score(behavior)
            color = self._score_color(score)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if face_bbox is not None:
                fx1, fy1, fx2, fy2 = face_bbox
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), Color.CYAN, 1)

                if behavior.get("looking_around"):
                    fcx  = (fx1 + fx2) // 2
                    fcy  = (fy1 + fy2) // 2
                    half = (fx2 - fx1) // 2
                    cv2.arrowedLine(frame, (fcx - half, fcy), (fcx + half, fcy), Color.ORANGE, 2, tipLength=0.4)
                    cv2.arrowedLine(frame, (fcx + half, fcy), (fcx - half, fcy), Color.ORANGE, 2, tipLength=0.4)

            tags = []
            if behavior.get("loitering"): tags.append("LOITERING")
            if behavior.get("repeated_path"): tags.append("REPEATED PATH")
            if behavior.get("looking_around"): tags.append("LOOKING AROUND")

            ab = behavior.get("abandoned_state", SuspicionState.NORMAL)
            if ab == SuspicionState.ALERT: tags.append("OBJ-ALERT")
            elif ab == SuspicionState.WARNING: tags.append("OBJ-WARN")

            label = f"ID {t['id']}  score:{score}"
            if tags:
                label += "  " + " ".join(tags)

            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    def _draw_trajectories(self, frame):
        for traj in self.trajectory_manager.get_all().values():
            for i in range(1, len(traj)):
                cv2.line(frame, traj[i - 1], traj[i], Color.BLUE, 1)

    def _draw_hud(self, frame, abandoned_results):
        alerts = sum(1 for o in abandoned_results if o["state"] == SuspicionState.ALERT)
        warnings = sum(1 for o in abandoned_results if o["state"] == SuspicionState.WARNING)
        h = frame.shape[0]

        lines = [
            (f"ALERTS: {alerts}", Color.RED if alerts else Color.WHITE),
            (f"WARNINGS: {warnings}", Color.ORANGE if warnings else Color.WHITE),
            (f"FRAME: {self.frame_count}", Color.GRAY),
        ]

        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (12, h - 16 - i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
            