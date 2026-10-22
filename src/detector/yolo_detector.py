"""YOLO detector with occasional overlapping-crop recovery.

Retains the existing FootballIQ detection dictionary interface.
"""

import cv2
from ultralytics import YOLO

from config import MODEL_PATH
from detection_settings import (
    SETTINGS, CLASS_MIN_CONFIDENCE,
    RECOVERY_CROP_FRACTION, RECOVERY_MIN_CONFIDENCE,
    RECOVERY_DUPLICATE_IOU,
)


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


class YOLODetector:
    def __init__(self, model_path=MODEL_PATH, confidence=None,
                 enable_recovery=None):
        self.model_path = model_path
        self.confidence = (
            float(SETTINGS["confidence"])
            if confidence is None else float(confidence)
        )
        self.enable_recovery = (
            SETTINGS["enable_recovery"]
            if enable_recovery is None else bool(enable_recovery)
        )
        self.frame_counter = 0
        print(f"Loading FootballIQ YOLO: {model_path}")
        self.model = YOLO(str(model_path))
        print(f"YOLO ready | conf={self.confidence} | "
              f"size={SETTINGS['image_size']} | recovery={self.enable_recovery}")

    def _predict(self, image, offset=(0, 0), conf=None):
        results = self.model.predict(
            source=image,
            conf=self.confidence if conf is None else float(conf),
            iou=float(SETTINGS["iou"]),
            imgsz=int(SETTINGS["image_size"]),
            max_det=int(SETTINGS["max_detections"]),
            device="cpu",
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        out = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(result.names[class_id]).lower()
            score = float(box.conf[0])
            if score < CLASS_MIN_CONFIDENCE.get(name, self.confidence):
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            dx, dy = offset
            out.append({
                "bbox": [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
                "class_id": class_id,
                "class_name": name,
                "confidence": score,
            })
        return out

    @staticmethod
    def _deduplicate(items):
        # Merge a player/goalkeeper overlap as the same physical object.
        # Do not merge unrelated classes such as ball and player.
        def group(name):
            return "person" if name in ("player", "goalkeeper") else name

        accepted = []
        for candidate in sorted(
            items, key=lambda d: d["confidence"], reverse=True
        ):
            if any(
                group(candidate["class_name"]) == group(kept["class_name"])
                and _iou(candidate["bbox"], kept["bbox"])
                >= RECOVERY_DUPLICATE_IOU
                for kept in accepted
            ):
                continue
            accepted.append(candidate)
        return accepted

    def detect(self, frame, frame_index=None):
        if frame_index is None:
            frame_index = self.frame_counter
        self.frame_counter += 1
        results = self._predict(frame)
        if not self.enable_recovery:
            return results

        interval = max(1, int(SETTINGS["recovery_every"]))
        if frame_index % interval != 0:
            return results

        height, width = frame.shape[:2]
        if width < 320 or height < 180:
            return results
        crop_width = max(1, int(width * RECOVERY_CROP_FRACTION))
        crops = ((0, 0, crop_width, height),
                 (width - crop_width, 0, width, height))
        for x1, y1, x2, y2 in crops:
            crop = frame[y1:y2, x1:x2]
            # Second pass is for players, not penalty-spot ball detections.
            rescues = self._predict(
                crop, offset=(x1, y1),
                conf=min(self.confidence, RECOVERY_MIN_CONFIDENCE),
            )
            for item in rescues:
                if item["class_name"] not in ("player", "goalkeeper"):
                    continue
                if item["confidence"] < RECOVERY_MIN_CONFIDENCE:
                    continue
                results.append(item)
        return self._deduplicate(results)

    def get_players(self, detections):
        return [d for d in detections if d["class_name"] in ("player", "goalkeeper")]

    def get_referees(self, detections):
        return [d for d in detections if d["class_name"] == "referee"]

    def get_ball(self, detections):
        balls = [d for d in detections if d["class_name"] == "ball"]
        return max(balls, key=lambda d: d["confidence"], default=None)
