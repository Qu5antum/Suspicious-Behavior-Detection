from __future__ import annotations
from ultralytics import YOLO
from enum import Enum
from collections import deque
import math

# Geometri yardımcıları
def bbox_center(box: list) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def bbox_iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
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
    return [max(0, x1 - px), max(0, y1 - px),
            min(fw, x2 + px), min(fh, y2 + px)]


# Durum enum
class SuspicionState(Enum):
    NORMAL = "normal"   
    WARNING = "warning"  
    ALERT = "alert" 


class AlertReason(Enum):
    NONE = "none"
    OWNER_LEFT = "owner_left"    
    STRANGER_NEAR = "stranger_near"  
    LONG_STATIC = "long_static"    


# Çanta dedektörü — YOLOv8
class BagDetector:
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
            detections.append({"bbox": [x1, y1, x2, y2], "class": label, "conf": conf})
        return detections

  
# IoU tabanlı çanta takipçisi
class BagTracker:
    def __init__(self, iou_threshold: float = 0.2, max_lost: int = 20):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.next_id = 0
        self._tracks: dict[int, dict] = {}
        self._lost: dict[int, int] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        updated: set[int] = set()
        unmatched = list(range(len(detections)))

        for tid, track in list(self._tracks.items()):
            if not unmatched:
                break
            best_i, best_iou = -1, self.iou_threshold
            for i in unmatched:
                iou = bbox_iou(detections[i]["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_i = i
            if best_i >= 0:
                self._tracks[tid].update({
                    "bbox": detections[best_i]["bbox"],
                    "class": detections[best_i]["class"],
                    "conf": detections[best_i]["conf"],
                })
                self._lost[tid] = 0
                updated.add(tid)
                unmatched.remove(best_i)

        for i in unmatched:
            tid = self.next_id
            self.next_id += 1
            self._tracks[tid] = {"id": tid, **detections[i]}
            self._lost[tid] = 0
            updated.add(tid)

        for tid in list(self._tracks.keys()):
            if tid not in updated:
                self._lost[tid] += 1
                if self._lost[tid] > self.max_lost:
                    del self._tracks[tid]
                    del self._lost[tid]

        return list(self._tracks.values())


# Tek bir çantanın durumu
class BagState:
    def __init__(self):
        #hareket geçmişi
        self.positions: deque = deque(maxlen=30)
        self.static_frames: int   = 0

        #sahip bilgisi
        self.owner_id:int | None = None  
        self.owner_nearby: bool = False
        self.owner_gone_frame: int | None = None 
        self.had_owner: bool = False  
        self.owner_confirm_buf: int = 0      
        self.owner_candidate_id: int | None = None 

        self.state: SuspicionState = SuspicionState.NORMAL
        self.reason: AlertReason = AlertReason.NONE

    def confirm_owner(self, person_id: int, frame_id: int):
        self.had_owner = True
        self.owner_id = person_id
        self.owner_nearby = True
        self.owner_gone_frame = None  

    #sahip uzaklaştı / çıktı 
    def owner_left(self, frame_id: int):
        if self.owner_nearby:
            self.owner_gone_frame = frame_id 
        self.owner_nearby = False

    def frames_since_owner_left(self, frame_id: int) -> int:
        if self.owner_gone_frame is None:
            return 0
        return frame_id - self.owner_gone_frame


# Sahiplik analizörü  — üç senaryo
class OwnershipAnalyzer:
    """
    Sahip–çanta ilişkisini ve şüpheli durumları tespit eder.

    Üç ALERT senaryosu:
      1. OWNER_LEFT    — onaylı sahip uzaklaştı (>= alert_frames kare)
      2. STRANGER_NEAR — sahip gittikten sonra BAŞKA biri yanına geldi
      3. LONG_STATIC   — hiç sahip görülmedi, uzun süre hareketsiz

    WARNING senaryosu:
      — Sahip görüldü ama henüz alert_frames dolmadı  VEYA
        hiç sahip görülmedi, warning_frames kadar hareketsiz
    """

    def __init__(
        self,
        owner_dist_px: int = 600,  
        stranger_dist_px: int = 130,   
        confirm_frames: int = 4,   
        alert_frames:int = 10,   
        warning_frames: int = 30,    
        static_px: float = 5.0,   
        frame_w: int = 1280,
        frame_h: int = 720,
    ):
        self.owner_dist_px = owner_dist_px
        self.stranger_dist_px = stranger_dist_px
        self.confirm_frames = confirm_frames
        self.alert_frames = alert_frames
        self.warning_frames = warning_frames
        self.static_px = static_px
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.owner_lost_grace = 15
        self.owner_enter_dist = 300
        self.owner_exit_dist = 400

        self._states: dict[int, BagState] = {}

    # ── hareket güncelle ─────────────────────────────────────────────────

    def _update_movement(self, st: BagState, bbox: list) -> bool:
        """True döndürürse çanta hareketsizdir."""
        cx, cy = bbox_center(bbox)
        st.positions.append((cx, cy))

        if len(st.positions) < 2:
            return False
        
        # son iki kare arası mesafe
        dx = st.positions[-1][0] - st.positions[-2][0]
        dy = st.positions[-1][1] - st.positions[-2][1]
        dist = math.hypot(dx, dy)

        if dist <= self.static_px:
            st.static_frames += 1
            return True
        else:
            st.static_frames = 0
            return False
        
    # en yakın kişiyi bul
    def _closest_person(
        self, bag_bbox: list, persons: list[dict], dist_limit: float
    ) -> tuple[int | None, float]:
        """dist_limit içindeki en yakın kişiyi döndürür."""
        best_id, best_d = None, float("inf")
        bx, by = bbox_center(bag_bbox)
        for p in persons:
            d = bbox_dist(bag_bbox, p["bbox"])
            if d < dist_limit and d < best_d:
                best_d = d
                best_id = p["id"]
        return best_id, best_d

    # sahip onay mekanizmasi
    def _try_confirm_owner(
        self, st: BagState, bag_bbox: list,
        persons: list[dict], frame_id: int
    ):
        """
        Çantanın yanındaki kişiyi 'confirm_frames' kare boyunca
        sürekli görünce onaylı sahip yapar.
        """
        if st.owner_id is not None:
            return
        
        cand_id, _ = self._closest_person(
            bag_bbox, persons, self.owner_dist_px
        )

        if cand_id is None:
            st.owner_confirm_buf = 0
            st.owner_candidate_id = None
            return

        if cand_id == st.owner_candidate_id:
            st.owner_confirm_buf += 1
        else:
            st.owner_candidate_id = cand_id
            st.owner_confirm_buf = 1

        if st.owner_confirm_buf >= self.confirm_frames:
            # Eğer başka biri sahip olarak zaten belirlenmişse
            # ve o kişi hâlâ yakınsa → değiştirme
            if st.owner_id is not None and st.owner_id != cand_id:
                owner_still_near = any(
                    p["id"] == st.owner_id and
                    bbox_dist(bag_bbox, p["bbox"]) < self.owner_dist_px
                    for p in persons
                )
                if owner_still_near:
                    return   # gerçek sahip hala orada, yeni kişiyi sahip yapma

            st.confirm_owner(cand_id, frame_id)

    #ana güncelleme 
    def update(
        self,
        bags:     list[dict],
        persons:  list[dict],
        frame_id: int,
    ) -> list[dict]:

        results = []

        for bag in bags:
            bid  = bag["id"]
            bbox = bag["bbox"]

            if bid not in self._states:
                self._states[bid] = BagState()
            st = self._states[bid]

            # hareket
            is_static = self._update_movement(st, bbox)

            # Sahip onay mekanizması
            self._try_confirm_owner(st, bbox, persons, frame_id)

            # Onaylı sahip hala yakında mı?
            owner_currently_near = False

            if st.owner_id is not None:
                for p in persons:
                    if p["id"] != st.owner_id:
                        continue

                    d = bbox_dist(bbox, p["bbox"])

                    if st.owner_nearby:
                        owner_currently_near = d < self.owner_exit_dist
                    else:
                        owner_currently_near = d < self.owner_enter_dist

                    break

            if owner_currently_near:
                st.owner_nearby = True
                st.owner_gone_frame = None
            else:
                if st.owner_gone_frame is None:
                    st.owner_gone_frame = frame_id

                if frame_id - st.owner_gone_frame > self.owner_lost_grace:
                    st.owner_nearby = False
                else:
                    st.owner_nearby = True

            gone_f = st.frames_since_owner_left(frame_id)

            # Yabancı tespiti
            # Sahip gittikten sonra yanına gelen BAŞKA kişi
            stranger_near = False
            stranger_id   = None

            if not owner_currently_near and st.had_owner:
                for p in persons:
                    if p["id"] == st.owner_id:
                        continue   # bu sahip, atla
                    d = bbox_dist(bbox, p["bbox"])
                    if d < self.stranger_dist_px and d > 20:
                        stranger_near = True
                        stranger_id   = p["id"]
                        break

            # ── 5. Durum belirleme ───────────────────────────────────────
            #
            # Öncelik sırası (yüksekten düşüğe):
            #   ALERT  > WARNING > NORMAL
            #
            # ALERT koşulları:
            #   a) Sahip var, uzaklaştı, gone_f >= alert_frames, çanta hareketsiz
            #   b) Sahip gittikten sonra yabancı geldi
            #   c) Hiç sahip görülmedi, static_frames >= warning_frames * 2
            #
            # WARNING koşulları:
            #   a) Sahip var ama gone_f < alert_frames (henüz bekle)
            #   b) Hiç sahip görülmedi, static_frames >= warning_frames
            #

            owner_near = owner_currently_near
            has_owner  = st.had_owner
            gone_f     = st.frames_since_owner_left(frame_id)

            reason = AlertReason.NONE

            # PRIORITY 1: STRANGER
            if has_owner and not owner_near and stranger_near:
                state  = SuspicionState.ALERT
                reason = AlertReason.STRANGER_NEAR

            # PRIORITY 2: OWNER LEFT 
            elif has_owner and not owner_near:

                if gone_f >= self.alert_frames:
                    state  = SuspicionState.ALERT
                    reason = AlertReason.OWNER_LEFT

                else:
                    state  = SuspicionState.WARNING

            # PRIORITY 3: NO OWNER EVER (static)
            elif not has_owner:

                if st.static_frames >= self.warning_frames * 2:
                    state = SuspicionState.ALERT
                    reason = AlertReason.LONG_STATIC

                elif st.static_frames >= self.warning_frames:
                    state = SuspicionState.WARNING

                else:
                    state = SuspicionState.NORMAL

            # PRIORITY 4: OWNER PRESENT
            else:
                state = SuspicionState.NORMAL

            # Bir kez ALERT olan çanta,
            # sahip geri gelmedikçe NORMAL'e düşmez
            if (st.state == SuspicionState.ALERT
                    and state == SuspicionState.NORMAL
                    and not owner_currently_near):
                state = SuspicionState.ALERT
                reason = st.reason   # önceki sebebi koru

            st.state = state
            st.reason = reason

            results.append({
                "id":bid,
                "bbox":bbox,
                "class":bag["class"],
                "conf":bag["conf"],
                "state":st.state,
                "reason":st.reason,
                "owner_id":st.owner_id,
                "owner_nearby":owner_currently_near,
                "had_owner":st.had_owner,
                "static_frames":st.static_frames,
                "gone_frames":gone_f,
                "stranger_id":stranger_id,
            })

        # Silinen çantaların state'lerini temizle
        active = {b["id"] for b in bags}
        for bid in list(self._states.keys()):
            if bid not in active:
                del self._states[bid]

        return results