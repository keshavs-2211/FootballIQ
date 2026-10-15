
import cv2
import numpy as np

from config import TEAM_COLOR_MIN_CONFIDENCE


class JerseyClassifier:
    """
    Estimates jersey color from the upper torso using HSV analysis.

    Team A / Team B assignment is handled separately.
    """

    def __init__(self):
        self.min_confidence = TEAM_COLOR_MIN_CONFIDENCE

    def _get_torso_crop(self, frame, bbox):
        """Extract a narrower upper-torso crop to avoid shorts and legs."""
        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame.shape[:2]

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        box_width = x2 - x1
        box_height = y2 - y1

        # Focus on the chest and shirt, avoiding the head and lower body.
        torso_x1 = x1 + int(box_width * 0.22)
        torso_x2 = x1 + int(box_width * 0.78)
        torso_y1 = y1 + int(box_height * 0.22)
        torso_y2 = y1 + int(box_height * 0.48)

        crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None

        return crop

    def _get_dominant_color(self, crop):
        """Estimate shirt color while rejecting ambiguous crops."""
        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        total_pixels = hue.size
        if total_pixels == 0:
            return "Unknown", 0.0

        # Ignore dark pixels and weakly colored pixels for color detection.
        valid = (value > 45) & (saturation > 55)

        # White is considered separately, with a stricter brightness test.
        white_mask = (saturation < 55) & (value > 145)
        white_ratio = np.count_nonzero(white_mask) / total_pixels

        color_masks = {
            "Red": valid & ((hue < 10) | (hue >= 170)),
            "Yellow": valid & (hue >= 10) & (hue < 30),
            "Green": valid & (hue >= 38) & (hue < 82),
            "Blue": valid & (hue >= 90) & (hue < 130),
            "Purple": valid & (hue >= 130) & (hue < 170),
        }

        counts = {
            name: int(np.count_nonzero(mask))
            for name, mask in color_masks.items()
        }

        dominant_color = max(counts, key=counts.get)
        dominant_count = counts[dominant_color]
        colored_ratio = dominant_count / total_pixels

        # A white shirt should occupy a substantial part of the crop.
        # Do not label white if a strong saturated color is more prevalent.
        if (
            white_ratio >= 0.38
            and white_ratio > colored_ratio * 1.15
        ):
            confidence = min(0.95, 0.55 + white_ratio * 0.40)
            return "White", confidence

        # Too little evidence of any reliable shirt color.
        if dominant_count < max(8, int(total_pixels * 0.12)):
            return "Unknown", 0.0

        # Reject mixed-color crops instead of guessing.
        second_count = sorted(counts.values(), reverse=True)[1]
        purity = dominant_count / max(1, sum(counts.values()))

        if purity < 0.55 or dominant_count < second_count * 1.20:
            return "Unknown", 0.0

        confidence = min(
            0.95,
            0.50 + 0.35 * purity
            + 0.10 * min(
                1.0,
                float(np.mean(saturation[color_masks[dominant_color]]))
                / 255.0,
            ),
        )

        return dominant_color, confidence

    def classify(self, frame, bbox):
        """Return jersey color, confidence, and an unassigned team."""
        crop = self._get_torso_crop(frame, bbox)

        if crop is None:
            return {
                "team": "Unknown",
                "jersey_color": "Unknown",
                "confidence": 0.0,
            }

        jersey_color, confidence = self._get_dominant_color(crop)

        if confidence < self.min_confidence:
            jersey_color = "Unknown"
            confidence = 0.0

        return {
            "team": "Unknown",
            "jersey_color": jersey_color,
            "confidence": float(confidence),
        }
