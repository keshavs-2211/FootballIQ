
from collections import Counter

from config import MODEL_PATH, DEFAULT_CONFIDENCE
from detector.yolo_detector import YOLODetector
from utils.video_reader import VideoReader


VIDEO_PATH = "sample_data/input_videos/08fd33_4.mp4"
FRAMES_TO_TEST = 150


def main():
    print("Loading detector...")
    detector = YOLODetector(MODEL_PATH, confidence=0.10)
    reader = VideoReader(VIDEO_PATH)

    class_counts = Counter()
    ball_confidences = []
    frames_with_ball = 0
    frame_count = 0

    try:
        for frame_index, frame in reader.frames():
            detections = detector.detect(frame)
            frame_count += 1

            class_counts.update(
                detection["class_name"]
                for detection in detections
            )

            ball = detector.get_ball(detections)

            if ball is not None:
                frames_with_ball += 1
                ball_confidences.append(ball["confidence"])

            if frame_count >= FRAMES_TO_TEST:
                break
    finally:
        reader.release()

    print("\n" + "=" * 45)
    print("BALL DETECTION DIAGNOSTIC")
    print("=" * 45)
    print(f"Confidence threshold: {detector.confidence}")
    print(f"Frames tested: {frame_count}")
    print(f"Frames with ball: {frames_with_ball}")
    print(
        f"Ball detection rate: "
        f"{frames_with_ball / frame_count * 100:.1f}%"
        if frame_count else "Ball detection rate: N/A"
    )
    print(f"Total detections by class: {dict(class_counts)}")

    if ball_confidences:
        print(
            f"Ball confidence min: {min(ball_confidences):.3f}"
        )
        print(
            f"Ball confidence max: {max(ball_confidences):.3f}"
        )
        print(
            f"Ball confidence average: "
            f"{sum(ball_confidences) / len(ball_confidences):.3f}"
        )


if __name__ == "__main__":
    main()