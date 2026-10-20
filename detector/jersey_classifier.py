
from pathlib import Path

import cv2
import numpy as np


REFERENCE_NAMES = {
    "Team A": "team_a.png",
    "Team B": "team_b.png",
    "Team A GK": "team_a_gk.png",
    "Team B GK": "team_b_gk.png",
}


def select_reference_images(video_path, reference_dir="references"):
    """
    Select four jersey patches from the input video.

    Each reference can come from a different timestamp.
    Existing saved references can be reused.
    """
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        name: reference_dir / filename
        for name, filename in REFERENCE_NAMES.items()
    }

    if all(path.exists() for path in paths.values()):
        answer = input(
            "\nReuse existing jersey references? (y/n): "
        ).strip().lower()

        if answer == "y":
            return paths

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = count / fps if fps > 0 else 0

    print("\n========== JERSEY REFERENCES ==========")
    print("Select only the SHIRT, not the whole player.")
    print("Avoid face, skin, shorts and green grass.")
    print("Drag a rectangle, then press ENTER.")
    print("You can choose a different time for each shirt.")
    print(f"Video duration: {duration:.1f} seconds")

    try:
        for name, path in paths.items():
            while True:
                raw = input(
                    f"\nTimestamp for {name} in seconds [0]: "
                ).strip()

                try:
                    seconds = float(raw) if raw else 0.0
                except ValueError:
                    print("Please enter a valid number.")
                    continue

                seconds = max(
                    0.0,
                    min(seconds, max(0.0, duration - 0.1)),
                )

                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    seconds * 1000,
                )

                success, frame = capture.read()

                if not success:
                    print("Unable to read that frame. Try again.")
                    continue

                # Resize the selection preview without
                # changing its original coordinates.
                height, width = frame.shape[:2]
                scale = min(1.0, 960 / width)

                preview = (
                    cv2.resize(
                        frame,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                    if scale < 1.0
                    else frame.copy()
                )

                window = f"Select SHIRT: {name}"

                roi = cv2.selectROI(
                    window,
                    preview,
                    showCrosshair=True,
                    fromCenter=False,
                )

                cv2.destroyWindow(window)

                x, y, w, h = map(int, roi)

                if w < 5 or h < 5:
                    print(
                        "Selection too small or cancelled. "
                        "Try another timestamp."
                    )
                    continue

                original_x = int(x / scale)
                original_y = int(y / scale)
                original_w = int(w / scale)
                original_h = int(h / scale)

                patch = frame[
                    original_y:original_y + original_h,
                    original_x:original_x + original_w,
                ]

                if patch.size == 0:
                    print("Invalid selection. Try again.")
                    continue

                cv2.imwrite(str(path), patch)
                print(f"Saved {name}: {path}")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return paths


class JerseyClassifier:
    """
    Reference-based jersey classification.

    The user supplies four cropped shirt references.
    Each tracked torso is compared with all references.

    Return values retain the existing FootballIQ format.
    """

    def __init__(
        self,
        reference_paths,
        min_similarity=0.58,
        min_margin=0.055,
    ):
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.references = {}

        for name, path in reference_paths.items():
            image = cv2.imread(str(path))

            if image is None:
                raise FileNotFoundError(
                    f"Missing jersey reference: {path}"
                )

            self.references[name] = self._profile(image)

        print("\nLoaded jersey references:")
        for name in self.references:
            print(f" - {name}")

    @staticmethod
    def _get_torso_crop(frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        bw = x2 - x1
        bh = y2 - y1

        # Central torso region to reduce background.
        tx1 = x1 + int(bw * 0.30)
        tx2 = x1 + int(bw * 0.70)
        ty1 = y1 + int(bh * 0.23)
        ty2 = y1 + int(bh * 0.46)

        crop = frame[ty1:ty2, tx1:tx2]

        if crop.size == 0 or min(crop.shape[:2]) < 3:
            return None

        return crop

    @staticmethod
    def _profile(image):
        """
        Compact colour profile:
        - HSV hue/saturation distribution
        - Lab chromaticity distribution
        - Mean brightness and saturation
        """
        h, w = image.shape[:2]

        if max(h, w) > 64:
            scale = 64 / max(h, w)
            image = cv2.resize(
                image,
                (
                    max(3, int(w * scale)),
                    max(3, int(h * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        hsv_hist = cv2.calcHist(
            [hsv],
            [0, 1],
            None,
            [36, 8],
            [0, 180, 0, 256],
        )

        lab_hist = cv2.calcHist(
            [lab],
            [1, 2],
            None,
            [16, 16],
            [0, 256, 0, 256],
        )

        cv2.normalize(
            hsv_hist,
            hsv_hist,
            alpha=1,
            norm_type=cv2.NORM_L1,
        )

        cv2.normalize(
            lab_hist,
            lab_hist,
            alpha=1,
            norm_type=cv2.NORM_L1,
        )

        return {
            "hsv": hsv_hist,
            "lab": lab_hist,
            "brightness": float(np.mean(hsv[:, :, 2])),
            "saturation": float(np.mean(hsv[:, :, 1])),
        }

    @staticmethod
    def _similarity(first, second):
        hsv_distance = cv2.compareHist(
            first["hsv"],
            second["hsv"],
            cv2.HISTCMP_BHATTACHARYYA,
        )

        lab_distance = cv2.compareHist(
            first["lab"],
            second["lab"],
            cv2.HISTCMP_BHATTACHARYYA,
        )

        brightness_difference = abs(
            first["brightness"] - second["brightness"]
        ) / 255.0

        saturation_difference = abs(
            first["saturation"] - second["saturation"]
        ) / 255.0

        score = (
            0.45 * (1.0 - hsv_distance)
            + 0.40 * (1.0 - lab_distance)
            + 0.08 * (1.0 - brightness_difference)
            + 0.07 * (1.0 - saturation_difference)
        )

        return float(np.clip(score, 0, 1))

    def classify(self, frame, bbox):
        unknown = {
            "team": "Unknown",
            "jersey_color": "Unknown",
            "confidence": 0.0,
        }

        crop = self._get_torso_crop(frame, bbox)

        if crop is None:
            return unknown

        profile = self._profile(crop)

        scores = {
            name: self._similarity(profile, reference)
            for name, reference in self.references.items()
        }

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_name, best_score = ranked[0]
        second_score = ranked[1][1]

        # Reject weak or ambiguous matches.
        if best_score < self.min_similarity:
            return unknown

        if best_score - second_score < self.min_margin:
            return unknown

        return {
            "team": "Unknown",
            "jersey_color": best_name,
            "confidence": best_score,
        }
