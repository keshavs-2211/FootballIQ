
import cv2
from pathlib import Path

from utils.automatic_field_detector import AutomaticFieldDetector


video_path = "sample_data/input_videos/08fd33_4.mp4"
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

capture = cv2.VideoCapture(video_path)
ok, frame = capture.read()
capture.release()

if not ok:
    raise RuntimeError(f"Could not read video: {video_path}")

detector = AutomaticFieldDetector()
polygon, mask = detector.detect(frame)

if mask is None:
    raise RuntimeError("Field detector could not process the frame.")

cv2.imwrite(str(output_dir / "automatic_field_mask.jpg"), mask)

overlay = detector.draw(frame, polygon)
cv2.imwrite(str(output_dir / "automatic_field_overlay.jpg"), overlay)

if polygon is None:
    print("No plausible field region found.")
else:
    print("Candidate polygon points:")
    print(polygon.reshape(-1, 2).tolist())

print("Saved outputs/automatic_field_mask.jpg")
print("Saved outputs/automatic_field_overlay.jpg")
