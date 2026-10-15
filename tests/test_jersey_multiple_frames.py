import cv2
from collections import Counter

from detector.jersey_classifier import JerseyClassifier
from detector.yolo_detector import YOLODetector
from config import MODEL_PATH


VIDEO_PATH = "sample_data/input_videos/08fd33_4.mp4"


def main():
    print("Loading detector...")
    detector = YOLODetector(MODEL_PATH)

    print("Creating jersey classifier...")
    classifier = JerseyClassifier()

    print("Opening video...")
    cap = cv2.VideoCapture(VIDEO_PATH)

    frame_index = 0
    total_frames = 30

    all_colors = Counter()

    while frame_index < total_frames:

        success, frame = cap.read()

        if not success:
            break

        detections = detector.detect(frame)
        players = detector.get_players(detections)

        frame_colors = Counter()

        for player in players:

            result = classifier.classify(
                frame,
                player["bbox"]
            )

            color = result["jersey_color"]

            frame_colors[color] += 1
            all_colors[color] += 1

        print(
            f"Frame {frame_index}: "
            f"{dict(frame_colors)}"
        )

        frame_index += 1

    cap.release()

    print("\n==============================")
    print("Overall jersey colors")
    print("==============================")

    for color, count in all_colors.most_common():
        print(f"{color}: {count}")

    print("\nMulti-frame jersey test completed.")


if __name__ == "__main__":
    main()