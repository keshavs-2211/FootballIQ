
"""FootballIQ - Reference-Aware Ball Tracking.

Requires:
    references/<clip>/ball.png
    references/<clip>/ball_2.png
    references/<clip>/ball_3.png

Uses three ball reference crops to filter and rank YOLO
ball candidates, followed by Kalman filtering and short-term
optical flow tracking.

Existing professional ball CSV format is preserved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

import offline_settings as S
from camera.camera_movement import CameraMotionCompensator

cv2.setNumThreads(2)


# ============================================================
# 1. REFERENCE MATCHING SETTINGS
# ============================================================

# Lower values accept more candidates but may introduce
# more false positives.
MIN_REFERENCE_SIMILARITY = 0.39

# 65% appearance similarity, 35% YOLO confidence.
REFERENCE_WEIGHT = 0.65

MIN_YOLO_CONFIDENCE = 0.13
MIN_CROP_PIXELS = 4
BALL_PATCH_SIZE = 40


def setting(name, default):
    return getattr(S, name, default)


def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


# ============================================================
# 2. BALL REFERENCE MATCHER
# ============================================================

class BallReferenceMatcher:

    def __init__(self, reference_directory):

        folder = Path(reference_directory)

        filenames = [
            "ball.png",
            "ball_2.png",
            "ball_3.png",
        ]

        missing = [
            name for name in filenames
            if not (folder / name).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"Missing ball references in {folder}: {missing}. "
                "Capture all three references on the website."
            )

        self.references = []

        for name in filenames:

            path = folder / name

            image = cv2.imread(str(path))

            if image is None or min(image.shape[:2]) < MIN_CROP_PIXELS:
                raise ValueError(
                    f"Invalid ball reference: {path}"
                )

            descriptor = self.describe(image)
            self.references.append(descriptor)

        print(
            f"Loaded {len(self.references)} ball references "
            f"from {folder}"
        )

    @staticmethod
    def describe(image):

        interpolation = (
            cv2.INTER_AREA
            if max(image.shape[:2]) > BALL_PATCH_SIZE
            else cv2.INTER_LINEAR
        )

        patch = cv2.resize(
            image,
            (BALL_PATCH_SIZE, BALL_PATCH_SIZE),
            interpolation=interpolation,
        )

        hsv = cv2.cvtColor(
            patch, cv2.COLOR_BGR2HSV
        )

        lab = cv2.cvtColor(
            patch, cv2.COLOR_BGR2LAB
        )

        mask = np.zeros(
            (BALL_PATCH_SIZE, BALL_PATCH_SIZE),
            dtype=np.uint8,
        )

        center = (
            BALL_PATCH_SIZE // 2,
            BALL_PATCH_SIZE // 2,
        )

        radius = BALL_PATCH_SIZE * 39 // 100

        cv2.ellipse(
            mask,
            center,
            (radius, radius),
            0,
            0,
            360,
            255,
            -1,
        )

        color_hist = cv2.calcHist(
            [hsv],
            [0, 1],
            mask,
            [12, 8],
            [0, 180, 0, 256],
        )

        light_hist = cv2.calcHist(
            [lab],
            [0],
            mask,
            [16],
            [0, 256],
        )

        cv2.normalize(
            color_hist,
            color_hist,
            alpha=1.0,
            norm_type=cv2.NORM_L1,
        )

        cv2.normalize(
            light_hist,
            light_hist,
            alpha=1.0,
            norm_type=cv2.NORM_L1,
        )

        mean_lab = np.array(
            cv2.mean(lab, mask=mask)[:3],
            dtype=np.float32,
        )

        return color_hist, light_hist, mean_lab

    @staticmethod
    def compare(first, second):

        hsv_a, light_a, mean_a = first
        hsv_b, light_b, mean_b = second

        hsv_similarity = 1.0 - cv2.compareHist(
            hsv_a,
            hsv_b,
            cv2.HISTCMP_BHATTACHARYYA,
        )

        light_similarity = 1.0 - cv2.compareHist(
            light_a,
            light_b,
            cv2.HISTCMP_BHATTACHARYYA,
        )

        color_similarity = max(
            0.0,
            1.0 - np.linalg.norm(mean_a - mean_b) / 180.0,
        )

        score = (
            0.45 * hsv_similarity
            + 0.35 * light_similarity
            + 0.20 * color_similarity
        )

        return float(np.clip(score, 0.0, 1.0))

    def score(self, frame, bbox):

        height, width = frame.shape[:2]

        x1, y1, x2, y2 = [
            int(round(v)) for v in bbox
        ]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        if (
            x2 - x1 < MIN_CROP_PIXELS
            or y2 - y1 < MIN_CROP_PIXELS
        ):
            return 0.0

        crop = frame[y1:y2, x1:x2]

        candidate_descriptor = self.describe(crop)

        scores = [
            self.compare(candidate_descriptor, reference)
            for reference in self.references
        ]

        scores.sort(reverse=True)

        # Best matching reference plus supporting evidence
        # from the second-best reference.
        final_score = (
            0.75 * scores[0]
            + 0.25 * scores[1]
        )

        return round(final_score, 4)


# ============================================================
# 3. PITCH PROJECTION
# ============================================================

def project(pixel, homography, frame_to_first):

    try:
        point = np.asarray(
            pixel, dtype=np.float64
        ).reshape(1, 1, 2)

        compensated = cv2.perspectiveTransform(
            point, frame_to_first
        )

        projected = cv2.perspectiveTransform(
            compensated, homography
        )[0, 0]

        if np.isfinite(projected).all():
            return np.asarray(projected, dtype=float)

    except (cv2.error, TypeError, ValueError):
        pass

    return None


def permitted(point):

    if point is None:
        return False

    if not np.isfinite(point).all():
        return False

    if not (
        0 <= point[0] <= 105
        and 0 <= point[1] <= 68
    ):
        return False

    return True


# ============================================================
# 4. KALMAN BALL TRACKER
# ============================================================

class MotionBallFilter:

    def __init__(self, fps):

        self.fps = fps
        self.state = None

        self.P = np.eye(4) * 6.0

        self.last_measured = -10000
        self.seed = None
        self.seed_frame = -10000

        dt = 1.0 / fps

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        self.M = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        g = np.array([
            [dt * dt / 2, 0],
            [0, dt * dt / 2],
            [dt, 0],
            [0, dt],
        ])

        self.Q = (
            g @ g.T * 18 ** 2
            + np.eye(4) * 1e-7
        )

    def predict(self, frame_idx):

        if self.state is None:
            return None

        self.state = self.F @ self.state

        self.P = (
            self.F @ self.P @ self.F.T + self.Q
        )

        max_gap = setting(
            "BALL_MAX_PREDICTION_GAP", 11
        )

        if frame_idx - self.last_measured > max_gap:
            self.state = None
            self.seed = None
            return None

        return self.state[:2].copy()

    def candidate(
        self,
        frame_idx,
        point,
        source,
        confidence,
    ):

        if not permitted(point):
            return False, "outside_pitch_or_penalty_spot"

        if self.state is None:

            if source != "yolo":
                return False, "bootstrap_requires_yolo"

            window = setting(
                "BALL_CONFIRM_WINDOW", 7
            )

            if (
                self.seed is None
                or frame_idx - self.seed_frame > window
            ):

                self.seed = np.array(point)
                self.seed_frame = frame_idx

                return False, "awaiting_second_yolo"

            # Two hits must come from different frames.
            if frame_idx == self.seed_frame:
                return False, "awaiting_next_frame_yolo"

            gap = max(
                1, frame_idx - self.seed_frame
            )

            max_speed = setting(
                "BALL_MAX_SPEED_MPS", 42.0
            )

            max_distance = (
                max_speed * gap / self.fps + 2.0
            )

            if np.linalg.norm(
                point - self.seed
            ) > max_distance:

                self.seed = np.array(point)
                self.seed_frame = frame_idx

                return False, "failed_second_hit_gate"

            velocity = (
                (np.array(point) - self.seed)
                * self.fps / gap
            )

            self.state = np.array([
                point[0],
                point[1],
                velocity[0],
                velocity[1],
            ])

            self.P = np.diag([
                1.5, 1.5, 65.0, 65.0
            ])

            self.last_measured = frame_idx
            self.seed = None

            return True, "confirmed_two_yolo_hits"

        distance = float(
            np.linalg.norm(
                point - self.state[:2]
            )
        )

        since = max(
            1, frame_idx - self.last_measured
        )

        max_speed = setting(
            "BALL_MAX_SPEED_MPS", 42.0
        )

        limit = min(
            14.0,
            2.0 + max_speed * since / self.fps,
        )

        if distance > limit:
            return False, "motion_gate_failed"

        if source == "yolo":
            base = 0.42
        elif source == "flow":
            base = 1.3
        else:
            base = 1.8

        R = (
            np.eye(2)
            * (base / max(confidence, 0.25)) ** 2
        )

        K = (
            self.P @ self.M.T
            @ np.linalg.inv(
                self.M @ self.P @ self.M.T + R
            )
        )

        innovation = (
            np.array(point)
            - self.M @ self.state
        )

        self.state = (
            self.state + K @ innovation
        )

        IK = np.eye(4) - K @ self.M

        self.P = (
            IK @ self.P @ IK.T
            + K @ R @ K.T
        )

        # Only YOLO measurements refresh the track.
        if source == "yolo":
            self.last_measured = frame_idx

        return True, "accepted"


# ============================================================
# 5. SHORT-TERM OPTICAL FLOW
# ============================================================

class OpticalBallFollow:

    def __init__(self):

        self.prev_gray = None
        self.last_pixel = None
        self.template = None

    def update_frame(self, gray):

        self.prev_gray = gray.copy()

    def anchor(self, gray, pixel):

        x, y = np.round(pixel).astype(int)

        if not (
            8 <= x < gray.shape[1] - 8
            and 8 <= y < gray.shape[0] - 8
        ):
            return

        self.last_pixel = np.array(
            [x, y], dtype=np.float32
        )

        self.template = gray[
            y - 6:y + 7,
            x - 6:x + 7
        ].copy()

    def follow(self, gray):

        if (
            self.prev_gray is None
            or self.last_pixel is None
        ):
            return None

        try:

            next_points, status, _ = (
                cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    gray,
                    self.last_pixel.reshape(1, 1, 2),
                    None,
                    winSize=(17, 17),
                    maxLevel=3,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS
                        | cv2.TERM_CRITERIA_COUNT,
                        25,
                        0.015,
                    ),
                )
            )

            if (
                next_points is not None
                and status is not None
                and int(status[0, 0]) == 1
            ):

                pixel = next_points.reshape(2)

                max_step = setting(
                    "BALL_LK_MAX_PIXEL_STEP", 55.0
                )

                if (
                    np.isfinite(pixel).all()
                    and np.linalg.norm(
                        pixel - self.last_pixel
                    ) <= max_step
                ):
                    return pixel, "flow", 0.65

        except cv2.error:
            pass

        if (
            not setting("BALL_TEMPLATE_MATCH", True)
            or self.template is None
        ):
            return None

        x, y = np.round(
            self.last_pixel
        ).astype(int)

        x1 = max(0, x - 40)
        y1 = max(0, y - 40)

        x2 = min(gray.shape[1], x + 41)
        y2 = min(gray.shape[0], y + 41)

        roi = gray[y1:y2, x1:x2]

        if (
            min(roi.shape[:2]) < 14
            or np.std(self.template) < 8
        ):
            return None

        result = cv2.matchTemplate(
            roi,
            self.template,
            cv2.TM_CCOEFF_NORMED,
        )

        _, similarity, _, location = (
            cv2.minMaxLoc(result)
        )

        threshold = setting(
            "BALL_TEMPLATE_THRESHOLD", 0.78
        )

        if similarity < threshold:
            return None

        pixel = np.array([
            x1 + location[0] + 6,
            y1 + location[1] + 6,
        ], dtype=float)

        return pixel, "template", similarity


# ============================================================
# 6. YOLO BALL CANDIDATES
# ============================================================

class BallOnlyDetector:

    def __init__(self):

        from ultralytics import YOLO
        from config import MODEL_PATH

        self.model = YOLO(
            str(MODEL_PATH)
        )

        print(
            "Loaded YOLO model:",
            MODEL_PATH,
        )

    def detect(self, frame):

        results = self.model.predict(
            source=frame,
            conf=setting(
                "BALL_YOLO_CONF",
                MIN_YOLO_CONFIDENCE,
            ),
            iou=0.55,
            imgsz=setting(
                "BALL_YOLO_IMAGE_SIZE",
                768,
            ),
            device="cpu",
            max_det=100,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        candidates = []

        for box in result.boxes:

            class_id = int(box.cls[0])

            label = str(
                result.names[class_id]
            ).strip().lower()

            if label != "ball":
                continue

            candidates.append({
                "bbox": list(
                    map(
                        float,
                        box.xyxy[0].tolist(),
                    )
                ),
                "confidence": float(
                    box.conf[0]
                ),
            })

        return candidates


# ============================================================
# 7. COMPLETE TRACKING PIPELINE
# ============================================================

def run():

    for file in (
        S.VIDEO,
        S.CALIBRATION_JSON,
    ):

        if not Path(file).is_file():
            raise FileNotFoundError(
                f"Ball tracking input missing: {file}"
            )

    reference_directory = setting(
        "REFERENCE_DIR",
        f"references/{S.CLIP}",
    )

    matcher = BallReferenceMatcher(
        reference_directory
    )

    calibration = json.loads(
        Path(S.CALIBRATION_JSON).read_text(
            encoding="utf-8"
        )
    )

    H = np.asarray(
        calibration["homography"],
        dtype=np.float64,
    )

    if H.shape != (3, 3):
        raise ValueError(
            "Invalid pitch calibration homography."
        )

    detector = BallOnlyDetector()

    capture = cv2.VideoCapture(
        str(S.VIDEO)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {S.VIDEO}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
        or 25.0
    )

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    kalman = MotionBallFilter(fps)
    follower = OpticalBallFollow()

    motion = None
    frame_idx = 0

    counts = Counter()
    reasons = Counter()

    last_reference_score = 0.0

    Path(S.BALL_CSV).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(S.BALL_AUDIT_CSV).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_columns = [
        "frame", "timestamp_seconds", "pitch_x_m", "pitch_y_m",
        "source", "confidence", "pixel_x", "pixel_y", "reference_similarity",
    ]

    audit_columns = [
        "frame",
        "source",
        "confidence",
        "appearance_similarity",
        "combined_score",
        "pitch_x_m",
        "pitch_y_m",
        "accepted",
        "reason",
    ]

    try:

        with open(
            S.BALL_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as output, open(
            S.BALL_AUDIT_CSV,
            "w",
            newline="",
            encoding="utf-8",
        ) as audit_file:

            writer = csv.DictWriter(
                output,
                fieldnames=output_columns,
            )

            audit = csv.DictWriter(
                audit_file,
                fieldnames=audit_columns,
            )

            writer.writeheader()
            audit.writeheader()

            while True:

                ok, frame = capture.read()

                if not ok:
                    break

                if motion is None:

                    motion = CameraMotionCompensator(
                        frame
                    )

                    frame_to_first = np.eye(
                        3, dtype=float
                    )

                else:

                    frame_to_first = motion.update(
                        frame
                    )

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

                kalman.predict(frame_idx)

                options = []

                # --------------------------------
                # A. Optical flow candidates
                # --------------------------------

                weak = follower.follow(gray)

                if (
                    weak is not None
                    and kalman.state is not None
                ):

                    pixel, source, confidence = weak

                    point = project(
                        pixel,
                        H,
                        frame_to_first,
                    )

                    if permitted(point):

                        options.append({
                            "point": point,
                            "pixel": pixel,
                            "source": source,
                            "confidence": confidence,
                            "similarity": last_reference_score,
                            "score": confidence * 0.25,
                        })

                # --------------------------------
                # B. YOLO + reference verification
                # --------------------------------

                detection_interval = max(
                    1,
                    int(
                        setting(
                            "BALL_YOLO_EVERY", 2
                        )
                    ),
                )

                if frame_idx % detection_interval == 0:

                    detections = detector.detect(
                        frame
                    )

                    for detection in detections:

                        bbox = detection["bbox"]

                        confidence = safe_float(
                            detection["confidence"]
                        )

                        similarity = matcher.score(
                            frame,
                            bbox,
                        )

                        pixel = np.array([
                            (bbox[0] + bbox[2]) / 2,
                            (bbox[1] + bbox[3]) / 2,
                        ], dtype=float)

                        point = project(
                            pixel,
                            H,
                            frame_to_first,
                        )

                        combined = (
                            REFERENCE_WEIGHT * similarity
                            + (1 - REFERENCE_WEIGHT)
                            * confidence
                        )

                        if not permitted(point):

                            reason = (
                                "outside_pitch_or_penalty_spot"
                            )

                        elif similarity < MIN_REFERENCE_SIMILARITY:

                            reason = (
                                "rejected_reference_mismatch"
                            )

                        else:

                            reason = None

                        if reason is not None:

                            reasons[reason] += 1

                            audit.writerow({
                                "frame": frame_idx,
                                "source": "yolo",
                                "confidence": round(
                                    confidence, 3
                                ),
                                "appearance_similarity": similarity,
                                "combined_score": round(
                                    combined, 4
                                ),
                                "pitch_x_m": (
                                    round(float(point[0]), 3)
                                    if point is not None
                                    else ""
                                ),
                                "pitch_y_m": (
                                    round(float(point[1]), 3)
                                    if point is not None
                                    else ""
                                ),
                                "accepted": 0,
                                "reason": reason,
                            })

                            continue

                        options.append({
                            "point": point,
                            "pixel": pixel,
                            "source": "yolo",
                            "confidence": confidence,
                            "similarity": similarity,
                            "score": combined,
                        })

                # --------------------------------
                # C. Rank candidate locations
                # --------------------------------

                if kalman.state is None:

                    options.sort(
                        key=lambda item: (
                            item["source"] == "yolo",
                            item["score"],
                        ),
                        reverse=True,
                    )

                else:

                    predicted = kalman.state[:2]

                    options.sort(
                        key=lambda item: (
                            float(
                                np.linalg.norm(
                                    item["point"] - predicted
                                )
                            )
                            - 3.0 * item["score"]
                            - (
                                0.6
                                if item["source"] == "yolo"
                                else 0.0
                            )
                        )
                    )

                # --------------------------------
                # D. Kalman acceptance
                # --------------------------------

                measured_source = None
                measured_confidence = 0.0
                measured_pixel = None
                measured_similarity = None

                for candidate in options:

                    point = candidate["point"]

                    accepted, reason = kalman.candidate(
                        frame_idx,
                        point,
                        candidate["source"],
                        candidate["confidence"],
                    )

                    reasons[reason] += 1

                    audit.writerow({
                        "frame": frame_idx,
                        "source": candidate["source"],
                        "confidence": round(
                            candidate["confidence"], 3
                        ),
                        "appearance_similarity": (
                            candidate["similarity"]
                            if candidate["source"] == "yolo"
                            else ""
                        ),
                        "combined_score": round(
                            candidate["score"], 4
                        ),
                        "pitch_x_m": round(
                            float(point[0]), 3
                        ),
                        "pitch_y_m": round(
                            float(point[1]), 3
                        ),
                        "accepted": int(accepted),
                        "reason": reason,
                    })

                    if accepted:

                        measured_source = candidate[
                            "source"
                        ]

                        measured_confidence = candidate[
                            "confidence"
                        ]
                        measured_pixel = candidate["pixel"]
                        measured_similarity = candidate["similarity"]

                        if measured_source == "yolo":

                            last_reference_score = (
                                candidate["similarity"]
                            )

                        follower.anchor(
                            gray,
                            candidate["pixel"],
                        )

                        break

                # --------------------------------
                # E. Export compatible coordinates
                # --------------------------------

                max_prediction_gap = setting(
                    "BALL_MAX_PREDICTION_GAP", 11
                )

                if (
                    kalman.state is not None
                    and frame_idx - kalman.last_measured
                    <= max_prediction_gap
                ):

                    if measured_source:
                        source = (
                            f"measured_{measured_source}"
                        )
                    else:
                        source = "predicted"

                    writer.writerow({
                        "frame": frame_idx,
                        "timestamp_seconds": round(
                            frame_idx / fps, 4
                        ),
                        "pitch_x_m": round(
                            float(kalman.state[0]), 4
                        ),
                        "pitch_y_m": round(
                            float(kalman.state[1]), 4
                        ),
                        "source": source,
                        "confidence": round(measured_confidence, 3),
                        "pixel_x": round(float(measured_pixel[0]), 2) if measured_pixel is not None else "",
                        "pixel_y": round(float(measured_pixel[1]), 2) if measured_pixel is not None else "",
                        "reference_similarity": round(float(measured_similarity), 3) if measured_similarity is not None else "",
                    })

                    counts[source] += 1

                follower.update_frame(gray)

                frame_idx += 1

                if frame_idx % 100 == 0:

                    print(
                        f"Ball tracking: {frame_idx} frames "
                        f"| {dict(counts)}",
                        flush=True,
                    )

    finally:
        capture.release()

    # --------------------------------
    # F. Diagnostic report
    # --------------------------------

    report = {
        "clip": S.CLIP,
        "frames": frame_idx,
        "fps": round(fps, 4),
        "reference_images": [
            "ball.png",
            "ball_2.png",
            "ball_3.png",
        ],
        "reference_similarity_minimum": (
            MIN_REFERENCE_SIMILARITY
        ),
        "counts": dict(counts),
        "audit_reasons": dict(reasons),
        "warning": (
            "Appearance matching cannot recover "
            "a completely occluded ball."
        ),
    }

    report_path = Path(
        S.BALL_CSV
    ).with_name(
        f"{S.CLIP}_ball_reference_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== FootballIQ Ball Tracking Complete ===")
    print("Ball coordinates:", S.BALL_CSV)
    print("Candidate audit:", S.BALL_AUDIT_CSV)
    print("Reference report:", report_path)
    print("Output counts:", dict(counts))


if __name__ == "__main__":
    run()
