import json
from pathlib import Path
import cv2
import numpy as np


class PitchProjector:
    def __init__(self, calibration_path):
        data = json.loads(Path(calibration_path).read_text(encoding='utf-8'))
        self.H = np.asarray(data['homography'], dtype=np.float64)
        self.length = float(data.get('pitch_length', 105.0))
        self.width = float(data.get('pitch_width', 68.0))

    def project(self, x, y, current_to_first=None):
        p = np.array([[[float(x), float(y)]]], dtype=np.float64)
        if current_to_first is not None:
            p = cv2.perspectiveTransform(p, np.asarray(current_to_first, dtype=np.float64))
        result = cv2.perspectiveTransform(p, self.H)[0, 0]
        X, Y = map(float, result)
        if not np.isfinite([X, Y]).all():
            return None
        # Do not clamp: clipping would pile out-of-pitch detections on borders.
        if not (-3 <= X <= self.length + 3 and -3 <= Y <= self.width + 3):
            return None
        return (X, Y)
