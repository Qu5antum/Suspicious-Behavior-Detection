from __future__ import annotations
from ultralytics import YOLO
from enum import Enum
from collections import deque
import math


def bbox_center(box: list) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def bbox_iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def bbox_dist(a: list, b: list) -> float:
    cx1, cy1 = bbox_center(a)
    cx2, cy2 = bbox_center(b)
    return math.hypot(cx1 - cx2, cy1 - cy2)


def expand_bbox(box: list, px: int, fw: int, fh: int) -> list:
    x1, y1, x2, y2 = box
    return [
        max(0, x1 - px),
        max(0, y1 - px),
        min(fw, x2 + px),
        min(fh, y2 + px),
    ]

class SuspicionState(Enum):
    NORMAL = "normal"    
    WARNING = "warning"   
    ALERT = "alert"    

class BagDetector:
    """
    YOLOv8 aracılığıyla çantaları, sırt çantalarını ve valizleri algılar.
    """
    TARGET = {"backpack", "handbag", "suitcase", "bag"}

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame) -> list[dict]:
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            label = self.model.names[int(box.cls[0])]
            if label not in self.TARGET:
                continue
            conf = float(box.conf[0])
            if conf < self.conf:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "class": label,
                "conf": conf,
            })

        return detections


class BagTracker:
    """
    IoU tabanlı ürün takip sistemi.
    """
    def __init__(self, iou_threshold: float = 0.2, max_lost: int = 20):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 0
        self._tracks: dict[int, dict] = {} 
        self._lost: dict[int, int]  = {}   

    def update(self, detections: list[dict]) -> list[dict]:
        updated: set[int] = set()

        unmatched_dets = list(range(len(detections)))

        for tid, track in list(self._tracks.items()):
            if not unmatched_dets:
                break

            best_i, best_iou = -1, self.iou_threshold
            for i in unmatched_dets:
                iou = bbox_iou(detections[i]["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_i = i

            if best_i >= 0:
                self._tracks[tid]["bbox"] = detections[best_i]["bbox"]
                self._tracks[tid]["class"] = detections[best_i]["class"]
                self._tracks[tid]["conf"] = detections[best_i]["conf"]
                self._lost[tid] = 0
                updated.add(tid)
                unmatched_dets.remove(best_i)

        for i in unmatched_dets:
            tid = self.next_id
            self.next_id += 1
            self._tracks[tid] = {
                "id": tid,
                "bbox": detections[i]["bbox"],
                "class": detections[i]["class"],
                "conf": detections[i]["conf"],
            }
            self._lost[tid] = 0
            updated.add(tid)

        for tid in list(self._tracks.keys()):
            if tid not in updated:
                self._lost[tid] += 1
                if self._lost[tid] > self.max_lost:
                    del self._tracks[tid]
                    del self._lost[tid]

        return list(self._tracks.values())


class BagState:
    def __init__(self):
        self.positions = deque(maxlen=10) 
        self.static_frames = 0
        self.owner_nearby = False
        self.had_owner = False
        self.owner_id = None
        self.state = SuspicionState.NORMAL
        self._owner_frames = {}

    def on_owner_appeared(self, person_id: int):
        self.had_owner = True
        self.owner_id = person_id
        self.owner_nearby = True
        self.owner_gone_frame = None   

    def on_owner_gone(self, frame_id: int):
        if self.owner_nearby:           
            self.owner_gone_frame = frame_id
        self.owner_nearby = False

    def frames_gone(self, frame_id: int) -> int:
        if self.owner_gone_frame is None:
            return 0
        return frame_id - self.owner_gone_frame

    def is_static(self) -> bool:
        return self.static_frames > 0  

class OwnershipAnalyzer:
    """
    Bir eşyanın kime ait olduğunu belirler ve durumunu takip eder.
    """
    def __init__(
        self,
        owner_expand_px: int = 80,    
        owner_iou_thresh: float = 0.05, 
        alert_frames: int = 10,   
        warning_frames: int = 100,  
        static_px:int = 6, 
        frame_w: int = 1280,
        frame_h: int = 720,
    ):
        self.owner_expand_px = owner_expand_px
        self.owner_iou_thresh = owner_iou_thresh
        self.alert_frames = alert_frames
        self.warning_frames = warning_frames
        self.static_px = static_px
        self.frame_w = frame_w
        self.frame_h = frame_h

        self._states: dict[int, BagState] = {}

    def _find_owner(
        self,
        bag_bbox: list,
        persons:  list[dict],
    ) -> tuple[int | None, float]:
        """
        (person_id, iou) en iyi sahip adayını döndürür.
        Öğenin genişletilmiş sınırlayıcı kutusunu kullanır.
        """
        expanded = expand_bbox(
            bag_bbox, self.owner_expand_px, self.frame_w, self.frame_h
        )
        best_id, best_iou = None, self.owner_iou_thresh

        for p in persons:
            iou = bbox_iou(expanded, p["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_id = p["id"]

        return best_id, best_iou


    def _update_movement(self, state: BagState, bbox: list) -> bool:
        """Öğenin konumunu günceller ve öğe sabitse True değerini döndürür."""
        cx, cy = bbox_center(bbox)
        state.positions.append((cx, cy))

        if len(state.positions) < 2:
            return False

        dx = abs(state.positions[-1][0] - state.positions[-2][0])
        dy = abs(state.positions[-1][1] - state.positions[-2][1])
        dist = math.hypot(dx, dy)

        if dist <= self.static_px:
            state.static_frames += 1
            return True
        else:
            state.static_frames = 0
            return False

    def update(
        self,
        bags: list[dict],
        persons: list[dict],
        frame_id: int,
    ) -> list[dict]:

        results = []

        for bag in bags:
            bid = bag["id"]
            bbox = bag["bbox"]

            if bid not in self._states:
                self._states[bid] = BagState()
            st = self._states[bid]

            is_static = self._update_movement(st, bbox)

            owner_id, owner_iou = self._find_owner(bbox, persons)

            if owner_id is not None:
                st.on_owner_appeared(owner_id)
            else:
                st.on_owner_gone(frame_id)

            gone_f = st.frames_gone(frame_id)

            if st.owner_nearby or not is_static:
                st.state = SuspicionState.NORMAL
            elif st.had_owner:
                if gone_f >= self.alert_frames:
                    st.state = SuspicionState.ALERT
                else:
                    st.state = SuspicionState.WARNING
            else:
                if st.static_frames >= self.warning_frames:
                    st.state = SuspicionState.ALERT
                else:
                    st.state = SuspicionState.WARNING

            results.append({
                "id": bid,
                "bbox": bbox,
                "class": bag["class"],
                "conf": bag["conf"],
                "state": st.state,
                "static_frames": st.static_frames,
                "had_owner": st.had_owner,
                "owner_nearby": st.owner_nearby,
                "owner_id": st.owner_id,
                "gone_frames":  gone_f,
            })

        active = {b["id"] for b in bags}
        for bid in list(self._states.keys()):
            if bid not in active:
                del self._states[bid]

        return results