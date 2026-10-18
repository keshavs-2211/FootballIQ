import cv2
import pandas as pd
from pathlib import Path
from utils.pitch_projector import AutomaticPitchProjector

video_path = Path("sample_data/input_videos/08fd33_4.mp4")
csv_path = Path("outputs/csv/03d06da60d324789bfbf469e642049d8_annotated_coordinates.csv")
output_path = Path("outputs/videos/footballiq_pitch_projection_test.mp4")

df = pd.read_csv(csv_path)
cap = cv2.VideoCapture(str(video_path))

ok, first_frame = cap.read()
if not ok:
    raise RuntimeError(f"Cannot read video: {video_path}")

projector = AutomaticPitchProjector()
success, _ = projector.calibrate(first_frame)
if not success:
    cap.release()
    raise RuntimeError("Automatic field calibration failed.")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
pitch = projector.draw_pitch()
height, width = pitch.shape[:2]

output_path.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

# CSV frame numbers are 1-based; video frame indices are 0-based.
by_frame = {int(n): group for n, group in df.groupby("frame")}
frame_index = 0
projected_count = 0

try:
    while True:
        if frame_index == 0:
            frame = first_frame
            ok = True
        else:
            ok, frame = cap.read()

        if not ok:
            break

        canvas = projector.draw_pitch()
        rows = by_frame.get(frame_index + 1)

        if rows is not None:
            for row in rows.itertuples(index=False):
                if row.class_name not in ("player", "goalkeeper"):
                    continue

                point = projector.project_point(row.foot_x, row.foot_y)
                if point is None:
                    continue

                team = str(row.team)
                if team == "Team A":
                    color = (255, 80, 30)
                elif team == "Team B":
                    color = (30, 80, 255)
                else:
                    color = (0, 220, 255)

                cv2.circle(canvas, point, 7, color, -1)
                cv2.putText(
                    canvas, str(row.track_id),
                    (point[0] + 7, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                )
                projected_count += 1

        cv2.putText(
            canvas, f"Frame {frame_index + 1}",
            (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 255), 2,
        )
        writer.write(canvas)
        frame_index += 1
finally:
    cap.release()
    writer.release()

print(f"Frames written: {frame_index}")
print(f"Player positions projected: {projected_count}")
print(f"Output: {output_path.resolve()}")
print("Note: this test uses the existing CSV, which covers frames 1-345.")
