
import cv2
import numpy as np


class AutomaticFieldDetector:
    """
    Lightweight automatic football pitch-line detector.

    Features:
    - Detects green playing surface
    - Detects candidate white pitch markings
    - Uses Hough Line Transform
    - Processes frames at reduced resolution
    - Draws cyan pitch-line overlays
    - Compatible with FootballIQ process_video.py

    Note:
    This detects candidate pitch markings, not
    a fully calibrated pitch boundary.
    """

    def __init__(self, max_width=640):
        self.max_width = max_width

    # ==========================================
    # DETECT PITCH LINES
    # ==========================================

    def detect(self, frame):

        if frame is None or frame.size == 0:
            return [], None

        original_height, original_width = frame.shape[:2]

        # --------------------------------------
        # RESIZE FOR LOWER CPU / RAM USAGE
        # --------------------------------------

        scale = min(
            1.0,
            self.max_width / original_width,
        )

        small_width = max(
            1,
            int(original_width * scale),
        )

        small_height = max(
            1,
            int(original_height * scale),
        )

        if scale < 1.0:

            small_frame = cv2.resize(
                frame,
                (small_width, small_height),
                interpolation=cv2.INTER_AREA,
            )

        else:
            small_frame = frame

        # --------------------------------------
        # CONVERT IMAGE TO HSV
        # --------------------------------------

        hsv = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2HSV,
        )

        # --------------------------------------
        # DETECT GREEN GRASS
        # --------------------------------------

        lower_green = np.array(
            [25, 35, 35],
            dtype=np.uint8,
        )

        upper_green = np.array(
            [100, 255, 255],
            dtype=np.uint8,
        )

        grass_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green,
        )

        # Remove small gaps in grass detection.
        grass_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )

        grass_mask = cv2.morphologyEx(
            grass_mask,
            cv2.MORPH_CLOSE,
            grass_kernel,
        )

        # Expand grass regions slightly so
        # white lines surrounded by grass
        # remain inside the detection area.
        grass_region = cv2.dilate(
            grass_mask,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (9, 9),
            ),
            iterations=1,
        )

        # --------------------------------------
        # DETECT WHITE FIELD MARKINGS
        # --------------------------------------

        lower_white = np.array(
            [0, 0, 150],
            dtype=np.uint8,
        )

        upper_white = np.array(
            [179, 100, 255],
            dtype=np.uint8,
        )

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white,
        )

        # Keep white markings near grass.
        line_mask = cv2.bitwise_and(
            white_mask,
            grass_region,
        )

        # --------------------------------------
        # REMOVE SMALL NOISE
        # --------------------------------------

        noise_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (2, 2),
        )

        line_mask = cv2.morphologyEx(
            line_mask,
            cv2.MORPH_OPEN,
            noise_kernel,
        )

        # --------------------------------------
        # HOUGH LINE DETECTION
        # --------------------------------------

        detected_lines = cv2.HoughLinesP(
            line_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=45,
            minLineLength=55,
            maxLineGap=20,
        )

        segments = []

        if detected_lines is None:
            return segments, line_mask

        # --------------------------------------
        # FIX ARRAY-SHAPE ERROR
        # --------------------------------------

        # HoughLinesP may return shape (N, 1, 4)
        # or (N, 4). Convert safely to (N, 4).
        lines = np.asarray(
            detected_lines
        ).reshape(-1, 4)

        # --------------------------------------
        # SCALE LINES TO ORIGINAL RESOLUTION
        # --------------------------------------

        scale_x = original_width / small_width
        scale_y = original_height / small_height

        for x1, y1, x2, y2 in lines:

            length = np.hypot(
                float(x2 - x1),
                float(y2 - y1),
            )

            # Ignore very short line segments.
            if length < 55:
                continue

            segments.append(
                (
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y),
                )
            )

        return segments, line_mask

    # ==========================================
    # DRAW DETECTED PITCH LINES
    # ==========================================

    def draw(self, frame, segments):
        """
        Draw candidate pitch markings in cyan.

        OpenCV uses BGR:
        (255, 255, 0) = Cyan
        """

        result = frame.copy()

        if not segments:
            return result

        for x1, y1, x2, y2 in segments:

            cv2.line(
                result,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return result
