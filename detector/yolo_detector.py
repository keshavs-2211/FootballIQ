
from ultralytics import YOLO

from config import (
    MODEL_PATH,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MAX_DETECTIONS,
    PLAYER_CLASSES,
    REFEREE_CLASS,
    BALL_CLASS,
)


class YOLODetector:
    """
    FootballIQ YOLO object detector.

    Uses the project's trained football model.
    Tracking and team assignment are handled
    by other modules.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence=DEFAULT_CONFIDENCE,
        iou=DEFAULT_IOU,
        image_size=DEFAULT_IMAGE_SIZE,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

        print(f"Loading YOLO model: {model_path}")

        self.model = YOLO(str(model_path))

        print("YOLO model loaded successfully.")
        print(f"Confidence: {confidence}")
        print(f"IoU: {iou}")
        print(f"Image size: {image_size}")

    def detect(self, frame):
        """
        Return detections in the existing
        FootballIQ dictionary format.
        """

        if frame is None or frame.size == 0:
            return []

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            max_det=DEFAULT_MAX_DETECTIONS,
            device="cpu",
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        detections = []

        if result.boxes is None:
            return detections

        # Convert the small results table in one
        # operation instead of repeated GPU/CPU
        # conversions for individual boxes.
        boxes = result.boxes.cpu()

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            class_name = str(
                result.names[class_id]
            ).strip().lower()

            # Only football-related objects
            # are needed downstream.
            if class_name not in (
                "player",
                "goalkeeper",
                "referee",
                "ball",
            ):
                continue

            x1, y1, x2, y2 = (
                box.xyxy[0].tolist()
            )

            detections.append({
                "bbox": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
            })

        return detections

    def get_players(self, detections):
        return [
            detection
            for detection in detections
            if detection["class_name"]
            in PLAYER_CLASSES
        ]

    def get_referees(self, detections):
        return [
            detection
            for detection in detections
            if detection["class_name"]
            == REFEREE_CLASS
        ]

    def get_ball(self, detections):
        balls = [
            detection
            for detection in detections
            if detection["class_name"]
            == BALL_CLASS
        ]

        if not balls:
            return None

        return max(
            balls,
            key=lambda detection: (
                detection["confidence"]
            ),
        )
