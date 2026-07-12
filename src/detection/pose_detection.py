from ultralytics import YOLO

class PoseDetector:
    def __init__(self, model_name="yolo26n-pose.pt", conf=0.5):
        """
        Initialize the pose detector.
        :param model_name: Name or path to the YOLO-pose model
        :param conf: Confidence threshold for detection (0.0 - 1.0)
        """
        self.model = YOLO(model_name)
        self.conf = conf
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

    def detect(self, frame):
        """
        Returns:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.93,
                "keypoints": {
                    "nose": {
                        "x": ...,
                        "y": ...,
                        "confidence": ...
                    },
                    ...
                }
            }
        ]
        """

        results = self.model(
            frame,
            conf=self.conf,
            verbose=False
        )

        if not results:
            return []

        result = results[0]

        if result.boxes is None or result.keypoints is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy()
        kp_scores = result.keypoints.conf.cpu().numpy()

        detected_persons = []

        for bbox, score, person_kps, person_scores in zip(
            boxes,
            scores,
            keypoints,
            kp_scores
        ):

            person = {
                "bbox": bbox.tolist(),
                "confidence": float(score),
                "keypoints": {}
            }

            for name, point, kp_score in zip(
                self.keypoint_names,
                person_kps,
                person_scores
            ):

                x = int(point[0])
                y = int(point[1])

                if kp_score is None:
                    continue

                if kp_score < 0.3:
                    continue

                person["keypoints"][name] = {
                    "x": x,
                    "y": y,
                    "confidence": float(kp_score)
                }

            detected_persons.append(person)

        return detected_persons
