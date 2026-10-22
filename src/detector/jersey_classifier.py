"""Reference jersey selection and lightweight multi-reference comparison.

For existing matches, the primary files remain team_a.png / team_b.png,
so the bird's-eye renderer can reuse them as circle-colour references.
"""

from pathlib import Path
import cv2
import numpy as np
from detection_settings import SETTINGS, REFERENCE_COUNT_PER_ROLE

ROLE_FILES = {
    "Team A": "team_a",
    "Team B": "team_b",
    "Team A GK": "team_a_gk",
    "Team B GK": "team_b_gk",
    "Referee": "referee",
}


def _saved_references(directory):
    output = {}
    for name, stem in ROLE_FILES.items():
        files = []
        primary = directory / (stem + ".png")
        if primary.is_file():
            files.append(primary)
        for i in range(2, REFERENCE_COUNT_PER_ROLE + 1):
            extra = directory / f"{stem}_{i}.png"
            if extra.is_file():
                files.append(extra)
        if files:
            output[name] = files
    return output


def select_reference_images(video_path, reference_dir="references"):
    """Interactive crop selection. Every jersey role requires three images."""
    directory = Path(reference_dir)
    directory.mkdir(parents=True, exist_ok=True)
    current = _saved_references(directory)
    if all(len(current.get(name, [])) >= 3 for name in ROLE_FILES):
        answer = input("\nReuse existing jersey references? [Y/n]: ").strip().lower()
        if answer not in ("n", "no"):
            return current

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    duration = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    selected = {}
    print("\nReference selection: drag around the SHIRT fabric only, press Enter.")
    print("Choose different video timestamps if necessary.")
    print("Three samples required for both teams, goalkeepers and referees.")
    print(f"Clip duration: {duration:.1f} seconds")
    try:
        for name, stem in ROLE_FILES.items():
            paths = []
            for index in range(1, REFERENCE_COUNT_PER_ROLE + 1):
                while True:
                    raw = input(f"Timestamp (seconds) for {name} image {index} [0]: ").strip()
                    try:
                        time_s = float(raw) if raw else 0.0
                    except ValueError:
                        print("Enter a number.")
                        continue
                    time_s = max(0.0, min(time_s, max(0.0, duration - 0.1)))
                    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * time_s)
                    ok, frame = cap.read()
                    if not ok:
                        print("Cannot read frame at this timestamp.")
                        continue
                    h, w = frame.shape[:2]
                    scale = min(1.0, 960.0 / w)
                    preview = (cv2.resize(frame, None, fx=scale, fy=scale)
                               if scale < 1 else frame.copy())
                    window_name = f"SELECT SHIRT: {name} #{index}"
                    rect = cv2.selectROI(window_name, preview, False, False)
                    cv2.destroyWindow(window_name)
                    x, y, rw, rh = (int(v) for v in rect)
                    if rw < 4 or rh < 4:
                        print("Cancelled or too small. Please retry.")
                        continue
                    x1, y1 = int(x / scale), int(y / scale)
                    x2, y2 = int((x + rw) / scale), int((y + rh) / scale)
                    patch = frame[y1:y2, x1:x2]
                    if patch.size == 0:
                        continue
                    filename = stem + ("" if index == 1 else f"_{index}") + ".png"
                    out_path = directory / filename
                    if not cv2.imwrite(str(out_path), patch):
                        raise RuntimeError(f"Failed to save {out_path}")
                    paths.append(out_path)
                    print(f"Saved: {out_path}")
                    break
            if paths:
                selected[name] = paths
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return selected


class JerseyClassifier:
    """Returns {'team':'Unknown', 'jersey_color':'Team A', 'confidence':0..1}."""

    def __init__(self, reference_paths, min_similarity=None, min_margin=None):
        self.min_similarity = (SETTINGS["jersey_min_similarity"]
                               if min_similarity is None else float(min_similarity))
        self.min_margin = (SETTINGS["jersey_min_margin"]
                           if min_margin is None else float(min_margin))
        self.references = {}
        for label, path_or_paths in reference_paths.items():
            paths = (path_or_paths if isinstance(path_or_paths, (tuple, list))
                     else [path_or_paths])
            profiles = []
            for path in paths:
                if path is None:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    print(f"Warning: missing reference {path}")
                    continue
                profiles.append(self._profile(image))
            if profiles:
                self.references[label] = profiles
        for required in ROLE_FILES:
            if len(self.references.get(required, [])) < REFERENCE_COUNT_PER_ROLE:
                raise ValueError(
                    f"Expected {REFERENCE_COUNT_PER_ROLE} references "
                    f"for {required}"
                )
        print("Loaded reference images:",
              {k: len(v) for k, v in self.references.items()})

    @staticmethod
    def _profile(image):
        """Central shirt-pixel histograms plus Lab median, with elliptical mask."""
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        yy, xx = np.ogrid[:48, :48]
        # Weight the centre chest, limiting background and arms.
        mask = (((xx - 23.5) / 22.0) ** 2 +
                ((yy - 23.5) / 22.0) ** 2 <= 1.0).astype(np.uint8) * 255
        hsv_hist = cv2.calcHist([hsv], [0, 1], mask,
                                [24, 8], [0, 180, 0, 256])
        lab_hist = cv2.calcHist([lab], [1, 2], mask,
                                [12, 12], [0, 256, 0, 256])
        cv2.normalize(hsv_hist, hsv_hist, alpha=1, norm_type=cv2.NORM_L1)
        cv2.normalize(lab_hist, lab_hist, alpha=1, norm_type=cv2.NORM_L1)
        valid = mask > 0
        return (hsv_hist, lab_hist,
                np.median(lab[valid], axis=0).astype(np.float32))

    @staticmethod
    def _score(a, b):
        hsv = 1.0 - cv2.compareHist(a[0], b[0], cv2.HISTCMP_BHATTACHARYYA)
        lab = 1.0 - cv2.compareHist(a[1], b[1], cv2.HISTCMP_BHATTACHARYYA)
        med_distance = float(np.linalg.norm(a[2] - b[2])) / 441.7
        score = 0.40 * hsv + 0.42 * lab + 0.18 * (1.0 - med_distance)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _torso_crops(frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = map(float, bbox)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        bw, bh = x2-x1, y2-y1
        if bw < 5 or bh < 12:
            return []
        crops = []
        # Two different torso placements: one close to chest, one wider.
        for xa, xb, ya, yb in (
            (0.31, 0.69, 0.22, 0.44),
            (0.24, 0.76, 0.26, 0.52),
        ):
            sx1, sx2 = int(x1 + xa*bw), int(x1 + xb*bw)
            sy1, sy2 = int(y1 + ya*bh), int(y1 + yb*bh)
            patch = frame[sy1:sy2, sx1:sx2]
            if patch.size and min(patch.shape[:2]) >= 3:
                crops.append(patch)
        return crops

    def _role_scores(self, frame, bbox):
        crops = self._torso_crops(frame, bbox)

        if not crops:
            return {}

        profiles = [
            self._profile(crop)
            for crop in crops
        ]

        scores = {}

        for role, references in self.references.items():
            scores[role] = max(
                self._score(profile, reference)
                for profile in profiles
                for reference in references
            )

        return scores

    def is_referee(self, frame, bbox):
        """
        Identify referees using three jersey references.

        Ambiguous candidates are retained rather than
        incorrectly deleting real players.
        """

        scores = self._role_scores(frame, bbox)

        referee_score = scores.get("Referee", 0.0)

        other_score = max(
            (
                score
                for role, score in scores.items()
                if role != "Referee"
            ),
            default=0.0
        )

        minimum_similarity = 0.63
        minimum_margin = 0.07

        return (
            referee_score >= minimum_similarity
            and referee_score - other_score >= minimum_margin
        )

    def classify(self, frame, bbox):
        unknown = {"team": "Unknown", "jersey_color": "Unknown",
                   "confidence": 0.0}
        scores = self._role_scores(frame, bbox)

        if not scores:
            return unknown

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if (
            best == "Referee"
            or top < self.min_similarity
            or top - runner_up < self.min_margin
        ):
            return unknown
        return {"team": "Unknown", "jersey_color": best,
                "confidence": float(top)}
