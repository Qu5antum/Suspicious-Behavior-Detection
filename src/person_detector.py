from ultralytics import YOLO
import cv2
import numpy as np


from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],  
            verbose=False
        )[0]

        persons = []

        if results.boxes.id is None:
            return persons

        for box, track_id in zip(results.boxes.xyxy, results.boxes.id):
            x1, y1, x2, y2 = map(int, box)

            persons.append({
                "id": int(track_id),
                "bbox": [x1, y1, x2, y2]
            })

        return persons
    

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