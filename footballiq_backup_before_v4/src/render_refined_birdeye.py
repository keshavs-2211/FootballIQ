"""FootballIQ refined bird's-eye renderer + optional Kalman ball tracking.

Run refine_coordinates.py first, then:
    python render_refined_birdeye.py

No YOLO reprocessing for *players*. Ball uses existing best.pt on sampled frames.
"""
from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd
import offline_settings as S

VIDEO = S.VIDEO
SIDE_SOURCE = S.ANNOTATED_VIDEO if Path(S.ANNOTATED_VIDEO).is_file() else S.VIDEO
REFINED_CSV = S.OUT_CSV
REFERENCE_FOLDER = S.REFERENCE_DIR
OUTPUT_VIDEO = S.OUT_VIDEO
OUTPUT_SIDE = S.OUT_SIDE
BALL_COORDS_CSV = S.BALL_CSV

WIDTH = 960
HEIGHT = 650
MARGIN = 46
cv2.setNumThreads(2)


def reference_bgr(filename, fallback):
    image = cv2.imread(str(Path(REFERENCE_FOLDER) / filename))
    if image is None:
        return fallback
    h, w = image.shape[:2]
    crop = image[max(0, h // 5):max(1, h * 4 // 5),
                 max(0, w // 5):max(1, w * 4 // 5)]
    if crop.size == 0:
        return fallback
    pixels = cv2.resize(crop, (48, 48)).reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(42)
    _, label, centers = cv2.kmeans(
        pixels, 3, None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 16, 0.2),
        3, cv2.KMEANS_PP_CENTERS,
    )
    largest = centers[np.bincount(label.ravel(), minlength=3).argmax()]
    return tuple(int(x) for x in np.clip(largest, 0, 255))


class PitchPainter:
    def __init__(self):
        self.colors = {
            "Team A": reference_bgr("team_a.png", (245, 245, 245)),
            "Team B": reference_bgr("team_b.png", (40, 40, 230)),
            "Team A GK": reference_bgr("team_a_gk.png", (245, 245, 245)),
            "Team B GK": reference_bgr("team_b_gk.png", (40, 40, 230)),
            "Unknown": (156, 156, 156),
        }
        self.base = self.pitch()

    @staticmethod
    def pt(x, y):
        return (round(MARGIN + x * (WIDTH - 2 * MARGIN) / 105),
                round(MARGIN + y * (HEIGHT - 2 * MARGIN) / 68))

    def pitch(self):
        out = np.full((HEIGHT, WIDTH, 3), (43, 151, 47), dtype=np.uint8)
        p = self.pt
        for i in range(10):
            if i % 2:
                cv2.rectangle(out, p(10.5 * i, 0), p(10.5 * (i + 1), 68),
                              (43, 141, 44), -1)
        white = (240, 245, 240)
        cv2.rectangle(out, p(0, 0), p(105, 68), white, 2)
        cv2.line(out, p(52.5, 0), p(52.5, 68), white, 2)
        sx = (WIDTH - 2 * MARGIN) / 105
        sy = (HEIGHT - 2 * MARGIN) / 68
        cv2.ellipse(out, p(52.5, 34), (round(9.15 * sx), round(9.15 * sy)),
                    0, 0, 360, white, 2)
        cv2.circle(out, p(52.5, 34), 3, white, -1)
        for is_right in (False, True):
            x1, x2 = (88.5, 105) if is_right else (0, 16.5)
            cv2.rectangle(out, p(x1, 13.84), p(x2, 54.16), white, 2)
            x1, x2 = (99.5, 105) if is_right else (0, 5.5)
            cv2.rectangle(out, p(x1, 24.84), p(x2, 43.16), white, 2)
            penalty_x = 94 if is_right else 11
            cv2.circle(out, p(penalty_x, 34), 3, white, -1)
            start, end = (127, 233) if is_right else (-53, 53)
            cv2.ellipse(out, p(penalty_x, 34),
                        (round(9.15 * sx), round(9.15 * sy)),
                        0, start, end, white, 2)
        return out

    def render(self, frame_idx, fps, players, ball):
        out = self.base.copy()
        for item in players:
            x, y = float(item["pitch_x_m"]), float(item["pitch_y_m"])
            if not (0 <= x <= 105 and 0 <= y <= 68):
                continue
            role = str(item.get("class_name", "player"))
            team = str(item.get("team", "Unknown"))
            key = team + " GK" if role == "goalkeeper" and team != "Unknown" else team
            color = self.colors.get(key, self.colors["Unknown"])
            px, py = self.pt(x, y)
            radius = 13 if role == "goalkeeper" else 11
            cv2.circle(out, (px + 2, py + 3), radius + 2, (28, 80, 27), -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), radius, (20, 20, 20), 1, cv2.LINE_AA)
            if S.DRAW_ESTIMATE_OUTLINES and str(item.get("position_source", "")).startswith(("interpolated", "held", "rejected")):
                cv2.circle(out, (px, py), radius + 4, (210, 210, 210), 1, cv2.LINE_AA)
            
        if ball is not None:
            x, y, state = ball
            if 0 <= x <= 105 and 0 <= y <= 68:
                pt = self.pt(x, y)
                cv2.circle(out, pt, 8, (20, 20, 20), -1, cv2.LINE_AA)
                cv2.circle(out, pt, 6, (250, 250, 250), -1, cv2.LINE_AA)
                cv2.circle(out, pt, 3, (20, 20, 20), -1, cv2.LINE_AA)
                if str(state) == "predicted":
                    cv2.circle(out, pt, 11, (150, 150, 150), 1, cv2.LINE_AA)
        return out



def writer(path, fps, size):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not out.isOpened():
        raise RuntimeError(f"Cannot write video {path}")
    return out


def run():
    for path in (VIDEO, SIDE_SOURCE, REFINED_CSV):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {path}")
    players = pd.read_csv(REFINED_CSV).fillna({"team": "Unknown"})
    frame_players = {int(f): g.to_dict("records") for f, g in players.groupby("frame")}
    frame_ball = {}
    if Path(BALL_COORDS_CSV).is_file():
        for row in pd.read_csv(BALL_COORDS_CSV).itertuples(index=False):
            frame_ball[int(row.frame)] = (float(row.pitch_x_m), float(row.pitch_y_m), str(row.source))
    else:
        print("Ball CSV not found; rendering players without ball. Run track_ball.py to include ball.")
    video = cv2.VideoCapture(VIDEO)
    ball_annotated = Path(f'outputs/videos/{S.CLIP}_ball_annotated.mp4')
    side_source = str(ball_annotated) if ball_annotated.is_file() else SIDE_SOURCE
    side = cv2.VideoCapture(side_source)
    if not video.isOpened() or not side.isOpened():
        raise RuntimeError("Cannot open video sources")
    fps = float(video.get(cv2.CAP_PROP_FPS) or 25.)
    painter = PitchPainter()
    tactical = writer(OUTPUT_VIDEO, fps, (WIDTH, HEIGHT))
    compared = writer(OUTPUT_SIDE, fps, (WIDTH * 2, HEIGHT))
    i = 0
    try:
        while True:
            ok, frame = video.read()
            ok2, side_frame = side.read()
            if not ok or not ok2:
                break
            image = painter.render(i, fps, frame_players.get(i, []), frame_ball.get(i))
            tactical.write(image)
            side_scaled = cv2.resize(side_frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            compared.write(np.hstack((side_scaled, image)))
            i += 1
            if i % 100 == 0:
                print(f"Rendered {i} frames")
    finally:
        tactical.release(); compared.release()
        video.release(); side.release()
    print("Refined tactical video:", OUTPUT_VIDEO)
    print("Refined side-by-side video:", OUTPUT_SIDE)
    print("Player sources: observed_smoothed / interpolated_same_id / held_first_frame")
    print("Ball statuses: measured_yolo / measured_flow / measured_template / predicted")


if __name__ == "__main__":
    run()
