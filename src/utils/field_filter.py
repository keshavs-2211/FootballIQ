
import cv2
import numpy as np


class FieldFilter:
    """
    Filter detections using a polygon in video-frame coordinates.

    By default, the legacy polygon filter is disabled because
    its hardcoded coordinates may exclude valid players.

    The old polygon remains available for debugging.
    """

    def __init__(self, polygon=None, enabled=False):
        self.enabled = enabled

        # Legacy normalized polygon coordinates.
        # These are NOT automatically detected pitch boundaries.
        self.normalized_polygon = (
            polygon
            if polygon is not None
            else [
                (0.00, 0.22),
                (1.00, 0.17),
                (1.00, 0.735),
                (0.00, 0.84),
            ]
        )

    def get_polygon(self, frame_width, frame_height):
        """Convert normalized polygon coordinates to pixels."""

        return np.array(
            [
                (
                    int(x * frame_width),
                    int(y * frame_height),
                )
                for x, y in self.normalized_polygon
            ],
            dtype=np.int32,
        )

    def contains_bbox(self, bbox, frame_width, frame_height):
        """
        Check whether the bottom-center of a bounding box
        lies inside the configured polygon.

        When filtering is disabled, accept all detections.
        """

        if not self.enabled:
            return True

        x1, y1, x2, y2 = map(float, bbox)

        foot_x = (x1 + x2) / 2.0
        foot_y = y2

        polygon = self.get_polygon(
            frame_width,
            frame_height,
        )

        return (
            cv2.pointPolygonTest(
                polygon,
                (float(foot_x), float(foot_y)),
                False,
            )
            >= 0
        )

    def contains_point(self, x, y, frame_width, frame_height):
        """
        Check whether a point is inside the polygon.

        When filtering is disabled, accept the point.
        """

        if not self.enabled:
            return True

        polygon = self.get_polygon(
            frame_width,
            frame_height,
        )

        return (
            cv2.pointPolygonTest(
                polygon,
                (float(x), float(y)),
                False,
            )
            >= 0
        )

    def draw_boundary(self, frame):
        """
        Draw the legacy polygon for debugging.

        This method is only called when debug
        visualization is enabled.
        """

        height, width = frame.shape[:2]

        polygon = self.get_polygon(width, height)

        result = frame.copy()

        cv2.polylines(
            result,
            [polygon],
            True,
            (0, 255, 255),
            2,
        )

        return result
