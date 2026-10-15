import cv2

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

    success, frame = cap.read()

    if not success:
        raise RuntimeError("Could not read the first frame.")

    print("\nRunning YOLO detection...")
    detections = detector.detect(frame)

    players = detector.get_players(detections)

    print(f"Total detections: {len(detections)}")
    print(f"Player/goalkeeper detections: {len(players)}")

    print("\nJersey classification results:")

    for index, player in enumerate(players):
        result = classifier.classify(
            frame,
            player["bbox"],
        )

        print(
            f"Player {index + 1}: "
            f"class={player['class_name']} "
            f"jersey_color={result['jersey_color']} "
            f"confidence={result['confidence']:.2f} "
            f"team={result['team']}"
        )

    cap.release()

    print("\nJersey classifier test completed.")


if __name__ == "__main__":
    main()