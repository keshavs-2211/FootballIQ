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

    print("\nInspecting possession state...")
    print("=" * 80)

    for frame_index, frame in reader.frames():

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        ball_detection = detector.get_ball(detections)

        ball_state = possession_tracker_ball_state(
            ball_detection,
            frame_index,
        )

        players = []

        for obj in tracked_objects:
            if obj["class_name"] not in {
                "player",
                "goalkeeper",
            }:
                continue

            if obj["track_id"] < 0:
                continue

            players.append(obj)

        result = possession_tracker.update(
            ball_state,
            players,
        )

        if ball_detection is not None:
            print(
                f"Frame {frame_index}: "
                f"candidate={possession_tracker.candidate_player_id}, "
                f"candidate_frames={possession_tracker.candidate_frames}, "
                f"current={possession_tracker.current_player_id}, "
                f"possession_frames={possession_tracker.possession_frames}, "
                f"result={result['possession_player_id']}"
            )

        if frame_index >= 29:
            break

    reader.release()

    print("\n" + "=" * 80)
    print("Possession state diagnostic completed.")


def possession_tracker_ball_state(ball_detection, frame_index):
    if ball_detection is None:
        return {
            "ball_x": None,
            "ball_y": None,
            "status": "unavailable",
            "confidence": None,
        }

    x1, y1, x2, y2 = ball_detection["bbox"]

    return {
        "ball_x": (x1 + x2) / 2.0,
        "ball_y": (y1 + y2) / 2.0,
        "status": "detected",
        "confidence": ball_detection["confidence"],
    }


if __name__ == "__main__":
    main()