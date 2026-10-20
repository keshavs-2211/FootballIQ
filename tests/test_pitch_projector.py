
import cv2
from utils.pitch_projector import AutomaticPitchProjector

video_path = "sample_data/input_videos/08fd33_4.mp4"

cap = cv2.VideoCapture(video_path)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError(f"Could not read {video_path}")

projector = AutomaticPitchProjector()
success, _ = projector.calibrate(frame)

if not success:
    raise RuntimeError("Field calibration failed")

pitch = projector.draw_pitch()

# Test a grid of points in the original frame.
height, width = frame.shape[:2]
test_points = [
    (width // 2, height // 2),
    (width // 4, height // 2),
    (3 * width // 4, height // 2),
    (width // 2, height // 4),
    (width // 2, 3 * height // 4),
]

for x, y in test_points:
    projected = projector.project_point(x, y)
    print(f"Frame point ({x}, {y}) -> pitch point {projected}")

    if projected is not None:
        cv2.circle(pitch, projected, 7, (0, 0, 255), -1)

cv2.imwrite("outputs/pitch_calibration_test.jpg", pitch)
print("Saved outputs/pitch_calibration_test.jpg")
