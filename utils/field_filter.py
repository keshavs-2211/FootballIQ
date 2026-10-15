import cv2
import numpy as np


class FieldFilter:
    """Filter detections using a polygon in video-frame coordinates."""

    def __init__(self, polygon=None):
        # Coordinates are normalized to frame width and height (0 to 1).
        # Replace these example points with the actual visible pitch boundary.
        self.normalized_polygon = polygon or [
            (0.00, 0.22),   # Far touchline, left
            (1.00, 0.17),   # Far touchline, right
            (1.00, 0.735),  # Near touchline, right
            (0.00, 0.84),   # Near touchline, left
        ]

    def get_polygon(self, frame_width, frame_height):
        return np.array(
            [
                (int(x * frame_width), int(y * frame_height))
                for x, y in self.normalized_polygon
            ],
            dtype=np.int32,
        )

    def contains_bbox(self, bbox, frame_width, frame_height):
        x1, y1, x2, y2 = map(float, bbox)
        foot_x = (x1 + x2) / 2.0
        foot_y = y2

        polygon = self.get_polygon(frame_width, frame_height)
        return cv2.pointPolygonTest(
            polygon,
            (float(foot_x), float(foot_y)),
            False,
        ) >= 0

    def draw_boundary(self, frame):
        height, width = frame.shape[:2]
        polygon = self.get_polygon(width, height)
        cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)
        return frame
