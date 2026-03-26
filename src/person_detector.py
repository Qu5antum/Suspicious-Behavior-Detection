from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonDetector:
    """
    YOLOv8 modelini kullanarak bir görüntüdeki kişileri tespit etmek için kullanılan bir sınıf.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]

        detections = []

        for box, conf, cls in zip(
            results.boxes.xyxy,
            results.boxes.conf,
            results.boxes.cls
        ):
            if int(cls) == 0: 
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1], 
                    "confidence": float(conf)
                })

        return detections
    

class PersonTracker:
    """
    Videoda insanları takip etme sınıfı,
    Onaylanmış her bir parça için benzersiz bir kimlik ve koordinatlar (x1, y1, x2, y2) içeren sözlüklerden oluşan bir liste döndürür.
    """
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            max_cosine_distance=0.3
        )
    """
    Güncelleme yöntemi bir çerçeve ve bir tespit listesi alır
    """
    def update(self, frame, detections):
        dets = [
            (det["bbox"], det["confidence"], "person")
            for det in detections
        ]

        tracks = self.tracker.update_tracks(dets, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            results.append({
                "id": track_id,
                "bbox": [l, t, r, b]
            })

        return results