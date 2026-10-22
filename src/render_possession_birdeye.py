"""Render offline possession states on top of FootballIQ's existing pitch painter.

Run after run_possession.py:
    python render_possession_birdeye.py

New output filenames: existing refined/other videos are never overwritten.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

import possession_settings as P
from render_refined_birdeye import PitchPainter, WIDTH, HEIGHT, writer
from run_possession import FrameCursor

cv2.setNumThreads(2)

HUD_COLORS = {
    "Team A": (225, 225, 245),
    "Team B": (80, 90, 245),
    "Contested": (0, 190, 255),
    "Free ball": (235, 210, 100),
    "Unknown": (175, 175, 175),
}


def draw_hud(image, state, ball_status, owner, elapsed, a_frames, b_frames):
    """A discreet display, not official match possession."""
    overlay = image.copy()
    cv2.rectangle(overlay, (12, HEIGHT-94), (WIDTH-12, HEIGHT-11), (20, 40, 24), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    color = HUD_COLORS.get(state, HUD_COLORS["Unknown"])
    cv2.circle(image, (31, HEIGHT-71), 9, color, -1, cv2.LINE_AA)
    label = f"Estimated possession: {state}"
    if owner and state in ("Team A", "Team B"):
        label += f" | ID {owner}"
    cv2.putText(image, label, (50, HEIGHT-66), cv2.FONT_HERSHEY_SIMPLEX, .62,
                (245,245,245), 2, cv2.LINE_AA)
    total = a_frames + b_frames
    if total:
        cv2.putText(image, f"Team-only: A {100*a_frames/total:.1f}% | B {100*b_frames/total:.1f}%  (check coverage in report)",
                    (24, HEIGHT-27), cv2.FONT_HERSHEY_SIMPLEX, .43, (232,232,232), 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "Assigned time: insufficient ball evidence",
                    (24, HEIGHT-27), cv2.FONT_HERSHEY_SIMPLEX, .49, (232,232,232), 1, cv2.LINE_AA)
    if state == "Unknown":
        cv2.putText(image, "LOW CONFIDENCE", (WIDTH-190, HEIGHT-67),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (185,185,245), 1, cv2.LINE_AA)
    return image


def parse_ball(rows):
    if not rows:
        return None
    item = rows[0]
    try:
        return (float(item["pitch_x_m"]), float(item["pitch_y_m"]), str(item["source"]))
    except (KeyError, ValueError, TypeError):
        return None


def run():
    for path in (P.VIDEO, P.PLAYER_CSV, P.BALL_CSV, P.TIMELINE_CSV):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing: {path}. Run tracking and run_possession.py first.")
    side_path = P.SIDE_VIDEO if Path(P.SIDE_VIDEO).is_file() else P.VIDEO
    source = cv2.VideoCapture(P.VIDEO)
    companion = cv2.VideoCapture(side_path)
    if not source.isOpened() or not companion.isOpened():
        source.release(); companion.release()
        raise RuntimeError("Could not open input/annotated videos")
    fps = float(source.get(cv2.CAP_PROP_FPS) or 25)
    players = FrameCursor(P.PLAYER_CSV)
    balls = FrameCursor(P.BALL_CSV)
    states = FrameCursor(P.TIMELINE_CSV)
    painter = PitchPainter()
    Path(P.OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
    tactical = writer(P.OUTPUT_VIDEO, fps, (WIDTH, HEIGHT))
    compared = writer(P.OUTPUT_SIDE, fps, (WIDTH*2, HEIGHT))
    counts = {"Team A": 0, "Team B": 0}
    i = 0
    try:
        while True:
            ok, frame = source.read()
            ok2, annotated = companion.read()
            if not ok or not ok2:
                break
            p_rows = players.get(i)
            ball = parse_ball(balls.get(i))
            state_rows = states.get(i)
            state_row = state_rows[0] if state_rows else {}
            state = state_row.get("state", "Unknown")
            owner = state_row.get("possessing_player_id", "")
            if state in counts:
                counts[state] += 1
            image = painter.render(i, fps, p_rows, ball)
            if P.DRAW_POSSESSION_HUD:
                image = draw_hud(image, state, state_row.get("ball_source", ""),
                                 owner, i/fps, counts["Team A"], counts["Team B"])
            tactical.write(image)
            left = cv2.resize(annotated, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            compared.write(np.hstack((left, image)))
            i += 1
            if i % 300 == 0:
                print(f"Rendered {i} frames")
    finally:
        tactical.release(); compared.release()
        source.release(); companion.release()
    print("Possession tactical:", P.OUTPUT_VIDEO)
    print("Possession side by side:", P.OUTPUT_SIDE)


if __name__ == "__main__":
    run()
