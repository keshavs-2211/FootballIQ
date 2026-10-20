from config import MODEL_PATH
from detector.yolo_detector import YOLODetector
from tracker.bytetrack_tracker import ByteTrackTracker
from utils.video_reader import VideoReader

from analytics.possession import PossessionTracker


VIDEO_PATH = "sample_data/input_videos/08fd33_4.mp4"


def main():
    print("Loading detector...")
    detector = YOLODetector(MODEL_PATH)

    print("Creating tracker...")
    tracker = ByteTrackTracker()

    print("Opening video...")
    reader = VideoReader(VIDEO_PATH)

    possession_tracker = PossessionTracker()

    print("\nInspecting nearby player candidates...")
    print("=" * 80)

    for frame_index, frame in reader.frames():

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        ball_detection = detector.get_ball(detections)

        if ball_detection is None:
            continue

        bx1, by1, bx2, by2 = ball_detection["bbox"]

        ball_x = (bx1 + bx2) / 2.0
        ball_y = (by1 + by2) / 2.0

        candidates = []

        for obj in tracked_objects:

            if obj["class_name"] not in {
                "player",
                "goalkeeper",
            }:
                continue

            if obj["track_id"] < 0:
                continue

            x1, y1, x2, y2 = obj["bbox"]

            foot_x = (x1 + x2) / 2.0
            foot_y = y2

            distance = possession_tracker._distance(
                (ball_x, ball_y),
                (foot_x, foot_y),
            )

            if distance <= 150:
                candidates.append(
                    (
                        distance,
                        obj["track_id"],
                        obj["class_name"],
                        x1,
                        y1,
                        x2,
                        y2,
                    )
                )

        candidates.sort(key=lambda item: item[0])

        print(
            f"\nFrame {frame_index} "
            f"Ball=({ball_x:.1f}, {ball_y:.1f})"
        )

        if not candidates:
            print("  No players within 150 px")
        else:
            for (
                distance,
                track_id,
                class_name,
                x1,
                y1,
                x2,
                y2,
            ) in candidates:
                print(
                    f"  ID={track_id} "
                    f"class={class_name} "
                    f"distance={distance:.1f}px "
                    f"bbox=({x1:.1f}, {y1:.1f}, "
                    f"{x2:.1f}, {y2:.1f})"
                )

        if frame_index >= 29:
            break

    reader.release()

    print("\n" + "=" * 80)
    print("Possession candidate diagnostic completed.")


if __name__ == "__main__":
    main()