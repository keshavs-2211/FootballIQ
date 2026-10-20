import cv2
from pathlib import Path
from utils.automatic_field_detector import AutomaticFieldDetector

video = r"sample_data/input_videos/08fd33_4.mp4"
out = Path("outputs/field_diagnostics")
out.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {video}")

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
detector = AutomaticFieldDetector()

try:
    for index in [0, 100, 250, 400, 600, 749]:
        if index >= total:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            print(f"Could not read frame {index}")
            continue

        polygon, mask = detector.detect(frame)
        overlay = detector.draw(frame, polygon)

        cv2.imwrite(str(out / f"frame_{index}_overlay.jpg"), overlay)
        cv2.imwrite(str(out / f"frame_{index}_mask.jpg"), mask)

        points = None if polygon is None else polygon.reshape(-1, 2).tolist()
        print(f"Frame {index}: polygon={points}")

finally:
    cap.release()

print(f"Diagnostics saved to: {out.resolve()}")
