from ultralytics import YOLO


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