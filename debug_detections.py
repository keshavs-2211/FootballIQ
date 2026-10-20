
from pathlib import Path
import argparse
import cv2

from detector.yolo_detector import YOLODetector
from detector.jersey_classifier import (
    JerseyClassifier,
    SUPPORTED_COLORS,
)


# ==========================================
# DEBUG SETTINGS
# ==========================================

# Lower than normal to expose weak YOLO detections.
DEBUG_CONFIDENCE = 0.15

# Detections below this are shown in red.
LOW_CONFIDENCE_THRESHOLD = 0.25

# OpenCV uses BGR colours.
GREEN = (70, 220, 70)
YELLOW = (0, 220, 255)
RED = (60, 60, 255)
PURPLE = (220, 80, 200)
BLUE = (255, 180, 40)
WHITE = (255, 255, 255)

cv2.setNumThreads(2)


# ==========================================
# INPUT HELPERS
# ==========================================

def ask_color(message):
    while True:
        value = input(message).strip().title()

        if value in SUPPORTED_COLORS:
            return value

        print(
            "Choose from: "
            + ", ".join(SUPPORTED_COLORS)
        )


def get_match_colors():
    print("\n========== DETECTION DEBUG SETUP ==========")
    print(
        "Supported colours: "
        + ", ".join(SUPPORTED_COLORS)
    )

    team_a = ask_color("Team A jersey: ")
    team_b = ask_color("Team B jersey: ")
    gk_a = ask_color("Team A goalkeeper: ")
    gk_b = ask_color("Team B goalkeeper: ")

    return {
        "Team A": team_a,
        "Team B": team_b,
        "Team A GK": gk_a,
        "Team B GK": gk_b,
    }


def match_team(color, role, config):
    if color == "Unknown":
        return "Unknown"

    if role == "goalkeeper":
        choices = ("Team A GK", "Team B GK")
    else:
        choices = ("Team A", "Team B")

    matches = [
        name
        for name in choices
        if config[name] == color
    ]

    if len(matches) == 1:
        return matches[0]

    return "Unknown"


# ==========================================
# DRAW BOX AND LABEL
# ==========================================

def draw_detection(frame, bbox, label, color):
    height, width = frame.shape[:2]

    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1

    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    top = max(0, y1 - text_h - baseline - 8)
    bottom = min(
        height - 1,
        top + text_h + baseline + 8,
    )

    right = min(
        width - 1,
        x1 + text_w + 8,
    )

    cv2.rectangle(
        frame,
        (x1, top),
        (right, bottom),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, min(height - 1, top + text_h + 4)),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


# ==========================================
# MAIN DEBUG PROCESSING
# ==========================================

def debug_video(input_path, output_path, config):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    video = cv2.VideoCapture(str(input_path))

    if not video.isOpened():
        raise RuntimeError(
            f"Cannot open video: {input_path}"
        )

    writer = None

    try:
        fps = video.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25.0

        width = int(
            video.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        frame_count = int(
            video.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Could not create output video."
            )

        # Load the same YOLO model as FootballIQ,
        # using a lower confidence threshold.
        detector = YOLODetector(
            confidence=DEBUG_CONFIDENCE
        )

        jersey_classifier = JerseyClassifier(
            allowed_colors=set(config.values())
        )

        counters = {
            "confirmed": 0,
            "uncertain": 0,
            "low_confidence": 0,
            "referee": 0,
            "ball": 0,
        }

        processed = 0

        print("\n========== DEBUG PROCESSING ==========")
        print(f"Input: {input_path}")
        print(f"Frames: {frame_count}")
        print(f"Resolution: {width}x{height}")
        print(f"Debug YOLO confidence: {DEBUG_CONFIDENCE}")

        while True:
            success, frame = video.read()

            if not success:
                break

            detections = detector.detect(frame)

            annotated = frame.copy()

            frame_counts = {
                "confirmed": 0,
                "uncertain": 0,
                "low_confidence": 0,
                "referee": 0,
                "ball": 0,
            }

            for detection in detections:
                role = str(
                    detection["class_name"]
                ).lower()

                confidence = float(
                    detection["confidence"]
                )

                bbox = detection["bbox"]

                # --------------------------------
                # REFEREES
                # --------------------------------

                if role == "referee":
                    color = PURPLE
                    label = f"REFEREE {confidence:.2f}"
                    category = "referee"

                # --------------------------------
                # BALL
                # --------------------------------

                elif role == "ball":
                    color = BLUE
                    label = f"BALL {confidence:.2f}"
                    category = "ball"

                # --------------------------------
                # PLAYERS / GOALKEEPERS
                # --------------------------------

                elif role in ("player", "goalkeeper"):
                    if confidence < LOW_CONFIDENCE_THRESHOLD:
                        color = RED
                        label = (
                            f"LOW CONF {role} "
                            f"{confidence:.2f}"
                        )
                        category = "low_confidence"

                    else:
                        result = jersey_classifier.classify(
                            frame,
                            bbox,
                        )

                        jersey_color = result[
                            "jersey_color"
                        ]

                        team = match_team(
                            jersey_color,
                            role,
                            config,
                        )

                        if team != "Unknown":
                            color = GREEN
                            label = (
                                f"{team} {role} "
                                f"{jersey_color} "
                                f"{confidence:.2f}"
                            )
                            category = "confirmed"
                        else:
                            color = YELLOW
                            label = (
                                f"UNKNOWN {role} "
                                f"{jersey_color} "
                                f"{confidence:.2f}"
                            )
                            category = "uncertain"

                else:
                    continue

                frame_counts[category] += 1
                counters[category] += 1

                draw_detection(
                    annotated,
                    bbox,
                    label,
                    color,
                )

            # --------------------------------
            # DEBUG SUMMARY
            # --------------------------------

            overlay = (
                f"Frame {processed} | "
                f"Confirmed {frame_counts['confirmed']} | "
                f"Unknown {frame_counts['uncertain']} | "
                f"Low conf {frame_counts['low_confidence']}"
            )

            cv2.rectangle(
                annotated,
                (0, 0),
                (min(width, 760), 58),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                annotated,
                overlay,
                (12, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                WHITE,
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated)

            processed += 1

            if processed % 25 == 0:
                print(
                    f"Processed {processed}/{frame_count}"
                )

        print("\n========== DEBUG RESULTS ==========")
        print(f"Frames processed: {processed}")

        for name, count in counters.items():
            print(f"{name}: {count}")

        print(
            "\nCounts are detection instances "
            "across frames, not unique people."
        )

        print(f"\nOutput saved: {output_path}")

    finally:
        video.release()

        if writer is not None:
            writer.release()


# ==========================================
# COMMAND LINE
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_video",
        help="Path to the input MP4",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional debug MP4 output path",
    )

    args = parser.parse_args()

    input_file = Path(args.input_video)

    output_file = (
        Path(args.output)
        if args.output
        else Path("outputs/videos") / (
            input_file.stem + "_detection_debug.mp4"
        )
    )

    colors = get_match_colors()

    debug_video(
        input_file,
        output_file,
        colors,
    )
