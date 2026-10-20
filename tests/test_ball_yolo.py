from config import MODEL_PATH
from detector.yolo_detector import YOLODetector
from tracker.ball_tracker import BallTracker
from utils.video_reader import VideoReader


VIDEO_PATH = "sample_data/input_videos/08fd33_4.mp4"


def main():
    print("Loading detector...")
    detector = YOLODetector(MODEL_PATH)

    print("Creating BallTracker...")
    ball_tracker = BallTracker()

    print("Opening video...")
    reader = VideoReader(VIDEO_PATH)

    print("\nTesting YOLO ball detection...")
    print("=" * 60)

    detected_count = 0
    unavailable_count = 0

    for frame_index, frame in reader.frames():

        detections = detector.detect(frame)

        ball_detection = detector.get_ball(detections)

        ball_state = ball_tracker.update(
            ball_detection,
            frame_index=frame_index,
        )

        if ball_detection is not None:
            detected_count += 1
        else:
            unavailable_count += 1

        print(
            f"Frame {frame_index}: "
            f"status={ball_state['status']} "
            f"ball=({ball_state['ball_x']}, "
            f"{ball_state['ball_y']}) "
            f"confidence={ball_state['confidence']} "
            f"frames_since_detection="
            f"{ball_state['frames_since_detection']}"
        )

        if frame_index >= 29:
            break

    reader.release()

    print("\n" + "=" * 60)
    print("Ball YOLO integration test completed.")
    print(f"Frames with ball detected: {detected_count}")
    print(f"Frames without ball: {unavailable_count}")


if __name__ == "__main__":
    main()