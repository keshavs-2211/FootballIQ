
import cv2
import numpy as np


class AutomaticFieldDetector:
    """Estimate the playing surface from grass-color regions."""

    def detect(self, frame):
        if frame is None or frame.size == 0:
            return None, None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Broad grass-color range. This is a starting heuristic,
        # not a trained semantic segmentation model.
        lower_green = np.array([25, 25, 25], dtype=np.uint8)
        upper_green = np.array([100, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Remove small noisy regions and fill small gaps.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (9, 9)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=3
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        height, width = frame.shape[:2]
        minimum_area = height * width * 0.08

        candidates = [
            contour for contour in contours
            if cv2.contourArea(contour) >= minimum_area
        ]

        if not candidates:
            return None, mask

        # Initial estimate: largest connected green region.
        largest = max(candidates, key=cv2.contourArea)

        # Approximate its outline without assuming fixed coordinates.
        perimeter = cv2.arcLength(largest, True)
        polygon = cv2.approxPolyDP(
            largest, 0.01 * perimeter, True
        )

        return polygon, mask

    def draw(self, frame, polygon):
        result = frame.copy()

        if polygon is not None:
            cv2.polylines(
                result, [polygon], True, (0, 255, 255), 3
            )

        return result
