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

    print("\nChecking ball-to-player distances...")
    print("=" * 70)

    possession_tracker = PossessionTracker()

    detected_ball_frames = 0
    frames_with_tracked_players = 0

    for frame_index, frame in reader.frames():

        detections = detector.detect(frame)

        tracked_objects = tracker.update(
            detections
        )

        ball_detection = detector.get_ball(
            detections
        )

        if ball_detection is None:
            if frame_index >= 29:
                break
            continue

        detected_ball_frames += 1

        x1, y1, x2, y2 = ball_detection["bbox"]

        ball_x = (x1 + x2) / 2.0
        ball_y = (y1 + y2) / 2.0

        closest_id = None
        closest_distance = None

        valid_player_count = 0

        for obj in tracked_objects:

            if obj["class_name"] not in {
                "player",
                "goalkeeper",
            }:
                continue

            if obj["track_id"] < 0:
                continue

            valid_player_count += 1

            px1, py1, px2, py2 = obj["bbox"]

            foot_x = (px1 + px2) / 2.0
            foot_y = py2

            distance = possession_tracker._distance(
                (ball_x, ball_y),
                (foot_x, foot_y),
            )

            if (
                closest_distance is None
                or distance < closest_distance
            ):
                closest_distance = distance
                closest_id = obj["track_id"]

        if valid_player_count > 0:
            frames_with_tracked_players += 1

        if closest_distance is None:
            print(
                f"Frame {frame_index}: "
                f"ball=({ball_x:.1f}, {ball_y:.1f}) "
                f"no valid tracked players"
            )
        else:
            print(
                f"Frame {frame_index}: "
                f"ball=({ball_x:.1f}, {ball_y:.1f}) "
                f"closest_player={closest_id} "
                f"distance={closest_distance:.1f}px"
            )

        if frame_index >= 29:
            break

    reader.release()

    print("\n" + "=" * 70)
    print(
        f"Frames with ball detection: "
        f"{detected_ball_frames}"
    )

    print(
        f"Ball-detected frames with tracked players: "
        f"{frames_with_tracked_players}"
    )

    print("Possession distance diagnostic completed.")


if __name__ == "__main__":
    main()