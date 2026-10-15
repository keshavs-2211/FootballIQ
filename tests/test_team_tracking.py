from config import MODEL_PATH
from detector.yolo_detector import YOLODetector
from detector.jersey_classifier import JerseyClassifier
from tracker.bytetrack_tracker import ByteTrackTracker
from analytics.team import TeamCalibrator
from analytics.team_persistence import TeamPersistence
from utils.video_reader import VideoReader


VIDEO_PATH = "sample_data/input_videos/08fd33_4.mp4"


def main():
    print("Loading detector...")
    detector = YOLODetector(MODEL_PATH)

    print("Creating tracker...")
    tracker = ByteTrackTracker()

    print("Creating jersey classifier...")
    classifier = JerseyClassifier()

    print("Creating team calibrator...")
    calibrator = TeamCalibrator(
        minimum_observations=20
    )

    print("Creating team persistence...")
    persistence = TeamPersistence()

    print("Opening video...")
    reader = VideoReader(VIDEO_PATH)

    # ---------------------------------------------------------
    # STEP 1: Collect team-color observations
    # ---------------------------------------------------------
    print("\nCollecting team-color observations...")

    for frame_index, frame in reader.frames():

        detections = detector.detect(frame)

        players = detector.get_players(detections)

        for player in players:

            result = classifier.classify(
                frame,
                player["bbox"]
            )

            calibrator.observe(
                result["jersey_color"],
                result["confidence"]
            )

        if frame_index >= 29:
            break

    # ---------------------------------------------------------
    # STEP 2: Calibrate Team A / Team B
    # ---------------------------------------------------------
    mapping = calibrator.calibrate()

    print("\nTeam calibration result:")

    if mapping is None:
        print("Team calibration failed.")
        reader.release()
        return

    print(mapping)

    # ---------------------------------------------------------
    # STEP 3: Tracking + team classification + persistence
    # ---------------------------------------------------------
    print("\nRunning tracking + team persistence...")

    reader.release()
    reader = VideoReader(VIDEO_PATH)

    for frame_index, frame in reader.frames():

        detections = detector.detect(frame)

        tracked_objects = tracker.update(detections)

        print(f"\nFrame {frame_index}")

        for obj in tracked_objects:

            # Only players and goalkeepers need team classification.
            if obj["class_name"] not in {
                "player",
                "goalkeeper",
            }:
                continue

            track_id = obj["track_id"]

            # Ignore detections without a persistent ID.
            if track_id < 0:
                continue

            # -------------------------------------------------
            # Jersey classification
            # -------------------------------------------------
            result = classifier.classify(
                frame,
                obj["bbox"]
            )

            # -------------------------------------------------
            # Instantaneous team classification
            # -------------------------------------------------
            detected_team = calibrator.get_team(
                result["jersey_color"],
                result["confidence"]
            )

            # -------------------------------------------------
            # Persistent team classification
            # -------------------------------------------------
            persistent_team = persistence.update(
                track_id=track_id,
                team=detected_team,
                confidence=result["confidence"],
            )

            print(
                f"  ID={track_id} "
                f"class={obj['class_name']} "
                f"color={result['jersey_color']} "
                f"detected_team={detected_team} "
                f"persistent_team={persistent_team}"
            )

        # Test first 10 frames.
        if frame_index >= 9:
            break

    reader.release()

    # ---------------------------------------------------------
    # STEP 4: Display final persisted team for each track
    # ---------------------------------------------------------
    print("\n==============================")
    print("Final persistent team assignments")
    print("==============================")

    for track_id in sorted(persistence.history.keys()):

        history = persistence.get_history(track_id)

        persistent_team = persistence.get_team(track_id)

        print(
            f"ID={track_id}: "
            f"team={persistent_team} "
            f"history={history}"
        )

    print("\nTeam tracking + persistence test completed.")


if __name__ == "__main__":
    main()