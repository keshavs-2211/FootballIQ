"""Overlay reference-verified ball observations on the annotated player footage.

Output: outputs/videos/<clip>_ball_annotated.mp4.
The tracker exports measured pixel coordinates; predicted ball positions are
intentionally NOT drawn as detections, to avoid implying measured evidence.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import cv2
import offline_settings as S


def load_observations(path):
    positions = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if not str(row.get('source', '')).startswith('measured_'):
                continue
            try:
                number = int(row['frame'])
                x, y = float(row['pixel_x']), float(row['pixel_y'])
            except (ValueError, TypeError, KeyError):
                continue
            if not all(map(math.isfinite, (x, y))):
                continue
            positions[number] = (round(x), round(y), row['source'])
    return positions


def overlay(frame, observation):
    if observation is None:
        return frame
    x, y, source = observation
    h, w = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return frame
    color = (0, 234, 255)
    cv2.circle(frame, (x, y), 12, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 12, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)
    label = 'BALL • verified' if source == 'measured_yolo' else 'BALL • image-tracked'
    cv2.putText(frame, label.replace(' • ', ' - '),
                (max(4, x - 20), max(20, y - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label.replace(' • ', ' - '),
                (max(4, x - 20), max(20, y - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)
    return frame


def render_video(source, ball_csv, output):
    observations = load_observations(ball_csv)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f'Annotated player video cannot be opened: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f'Cannot create video: {output}')
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(overlay(frame, observations.get(n)))
            n += 1
    finally:
        writer.release()
        cap.release()
    if not n:
        raise RuntimeError('Source video has zero readable frames.')
    print(f'Ball annotated video: {output} ({n} frames, {len(observations)} measured ball positions)')
    return n


def run():
    render_video(S.ANNOTATED_VIDEO, S.BALL_CSV,
                 f'outputs/videos/{S.CLIP}_ball_annotated.mp4')


if __name__ == '__main__':
    run()
