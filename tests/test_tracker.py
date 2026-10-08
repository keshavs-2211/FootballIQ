from config import MODEL_PATH
from detector.yolo_detector import YOLODetector
from tracker.bytetrack_tracker import ByteTrackTracker
from utils.video_reader import VideoReader


VIDEO_PATH = "sample_data/input_videos/08fd33_4.mp4"


def main():
    print("Loading detector...")
    detector = YOLODetector(MODEL_PATH)

    print("Creating ByteTrack tracker...")
    tracker = ByteTrackTracker()

    print("Opening video...")
    reader = VideoReader(VIDEO_PATH)

    for frame_index, frame in reader.frames():

        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)

        print(f"\nFrame {frame_index}")
        print(f"YOLO detections: {len(detections)}")
        print(f"Tracked objects: {len(tracked_objects)}")

        for obj in tracked_objects:
            print(
                f"  ID={obj['track_id']} "
                f"class={obj['class_name']} "
                f"confidence={obj['confidence']:.2f} "
                f"bbox={obj['bbox']}"
            )

        # Test only the first 5 frames.
        if frame_index >= 4:
            break

    reader.release()

    print("\nByteTrack test completed.")


if __name__ == "__main__":
    main()