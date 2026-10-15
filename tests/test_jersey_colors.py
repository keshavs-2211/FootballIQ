import cv2
import numpy as np

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

    detections = detector.detect(frame)
    players = detector.get_players(detections)

    print(f"\nPlayer detections: {len(players)}")
    print("\nJersey color diagnostics:")

    for index, player in enumerate(players):

        crop = classifier._get_torso_crop(
            frame,
            player["bbox"]
        )

        if crop is None:
            print(f"Player {index + 1}: invalid crop")
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        mask = (saturation > 40) & (value > 40)

        pixels = hsv[mask]

        if len(pixels) == 0:
            print(f"Player {index + 1}: no valid pixels")
            continue

        mean_hue = np.mean(pixels[:, 0])
        mean_saturation = np.mean(pixels[:, 1])
        mean_value = np.mean(pixels[:, 2])

        result = classifier.classify(
            frame,
            player["bbox"]
        )

        print(
            f"Player {index + 1}: "
            f"class={player['class_name']} | "
            f"H={mean_hue:.1f} | "
            f"S={mean_saturation:.1f} | "
            f"V={mean_value:.1f} | "
            f"detected={result['jersey_color']}"
        )

    cap.release()

    print("\nJersey color diagnostic completed.")


if __name__ == "__main__":
    main()