from ultralytics import YOLO

from config import (
    MODEL_PATH,
    DEFAULT_CONFIDENCE,
    PLAYER_CLASSES,
    REFEREE_CLASS,
    BALL_CLASS,
)


class YOLODetector:
    """
    YOLO-based object detector.

    Responsible only for object detection.
    It does not perform tracking or analytics.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence=DEFAULT_CONFIDENCE,
    ):
        self.model_path = model_path
        self.confidence = confidence

        print(f"Loading YOLO model: {model_path}")

        self.model = YOLO(str(model_path))

        print("YOLO model loaded successfully.")

    def detect(self, frame):
        """
        Detect objects in a single frame.

        Returns:
            list[dict]: Standard FootballIQ detection format.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        result = results[0]

        detections = []

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detection = {
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            }

            detections.append(detection)

        return detections


    def get_players(self, detections):
        """
        Return player and goalkeeper detections.
        """

        return [
            detection
            for detection in detections
            if detection["class_name"] in PLAYER_CLASSES
        ]

    def get_referees(self, detections):
        """
        Return referee detections.
        """

        return [
            detection
            for detection in detections
            if detection["class_name"] == REFEREE_CLASS
        ]

    def get_ball(self, detections):
        """
        Return ball detections.

        Returns:
            dict or None
        """

        balls = [
            detection
            for detection in detections
            if detection["class_name"] == BALL_CLASS
        ]

        if not balls:
            return None

        # Highest-confidence ball detection
        return max(
            balls,
            key=lambda detection: detection["confidence"],
        )

        