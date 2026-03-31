from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np


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


class FaceDetector:
    """
    OpenCV DNN face detector (SSD, Caffe)
    """
    def __init__(self, conf_threshold=0.5):
        proto = "deploy.prototxt.txt"
        model = "res10_300x300_ssd_iter_140000.caffemodel"
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype(int))
        return faces

class HeadPoseEstimator:
    """
    Kafanın yaklaşık pozisyonu, kafa sınırlayıcı kutusu içindeki yüz pozisyonu kullanılarak belirlenir.
    """
    def estimate(self, head_bbox, face_bbox):
        if head_bbox is None or face_bbox is None:
            return None

        x1, y1, x2, y2 = head_bbox
        fx1, fy1, fx2, fy2 = face_bbox

        head_center_x = (x1 + x2) / 2

        face_center_x = (fx1 + fx2) / 2

        head_width = x2 - x1
        if head_width == 0:
            return None

        yaw = (face_center_x - head_center_x) / (head_width / 2)

        return yaw