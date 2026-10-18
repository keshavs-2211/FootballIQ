
import cv2
import numpy as np

from utils.automatic_field_detector import AutomaticFieldDetector


class AutomaticPitchProjector:
    """Estimate a top-down projection from the detected playing surface."""

    def __init__(self, pitch_width=680, pitch_height=1050):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.field_detector = AutomaticFieldDetector()
        self.matrix = None
        self.source_corners = None

    @staticmethod
    def order_corners(points):
        """Order four points as top-left, top-right, bottom-right, bottom-left."""
        points = np.asarray(points, dtype=np.float32).reshape(4, 2)
        sums = points.sum(axis=1)
        differences = points[:, 1] - points[:, 0]

        return np.array([
            points[np.argmin(sums)],
            points[np.argmin(differences)],
            points[np.argmax(sums)],
            points[np.argmax(differences)],
        ], dtype=np.float32)

    def calibrate(self, frame):
        polygon, mask = self.field_detector.detect(frame)

        if polygon is None:
            return False, mask

        perimeter = cv2.arcLength(polygon, True)
        corners = None

        # Try progressively simpler approximations of the detected boundary.
        for epsilon in (0.01, 0.02, 0.03, 0.05, 0.08):
            approximation = cv2.approxPolyDP(
                polygon, epsilon * perimeter, True
            )
            if len(approximation) == 4 and cv2.isContourConvex(approximation):
                corners = approximation.reshape(4, 2)
                break

        # Fallback: use the detected boundary's rotated bounding rectangle.
        # This is approximate and may not align with the actual pitch corners.
        if corners is None:
            rectangle = cv2.minAreaRect(polygon)
            corners = cv2.boxPoints(rectangle)

        corners = self.order_corners(corners)

        width = self.pitch_width - 1
        height = self.pitch_height - 1

        destination = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(corners, destination)

        if not np.isfinite(matrix).all():
            return False, mask

        self.matrix = matrix
        self.source_corners = corners
        return True, mask

    def project_point(self, x, y):
        if self.matrix is None:
            return None

        point = np.array([[[x, y]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, self.matrix)[0, 0]

        px, py = float(projected[0]), float(projected[1])

        if not (0 <= px < self.pitch_width and
                0 <= py < self.pitch_height):
            return None

        return int(px), int(py)

    def draw_pitch(self):
        pitch = np.full(
            (self.pitch_height, self.pitch_width, 3),
            (45, 115, 45),
            dtype=np.uint8,
        )

        white = (235, 235, 235)
        margin = 20
        cv2.rectangle(
            pitch,
            (margin, margin),
            (self.pitch_width - margin, self.pitch_height - margin),
            white, 2,
        )
        cv2.line(
            pitch,
            (margin, self.pitch_height // 2),
            (self.pitch_width - margin, self.pitch_height // 2),
            white, 2,
        )
        cv2.circle(
            pitch,
            (self.pitch_width // 2, self.pitch_height // 2),
            75, white, 2,
        )

        return pitch
