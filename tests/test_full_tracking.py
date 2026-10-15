
from config import MODEL_PATH

from detector.yolo_detector import YOLODetector
from detector.jersey_classifier import JerseyClassifier

from tracker.bytetrack_tracker import ByteTrackTracker
from tracker.ball_tracker import BallTracker

from analytics.team import TeamCalibrator
from analytics.team_persistence import TeamPersistence
from analytics.possession import PossessionTracker

from utils.video_reader import VideoReader


VIDEO_PATH = "sample_data/input_videos/08fd33_4.mp4"


def main():
    print("Loading detector...")
    detector = YOLODetector(MODEL_PATH, confidence=0.15)

    print("Creating ByteTrack tracker...")
    tracker = ByteTrackTracker()

    print("Creating jersey classifier...")
    classifier = JerseyClassifier()

    print("Creating team calibrator...")
    calibrator = TeamCalibrator(minimum_observations=20)

    print("Creating team persistence...")
    team_persistence = TeamPersistence()

    print("Creating BallTracker...")
    ball_tracker = BallTracker()

    print("Creating PossessionTracker...")
    possession_tracker = PossessionTracker()

    # --------------------------------------------------
    # STEP 1: Team calibration
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 1: TEAM CALIBRATION")
    print("=" * 60)

    reader = VideoReader(VIDEO_PATH)

    for frame_index, frame in reader.frames():
        detections = detector.detect(frame)
        players = detector.get_players(detections)

        for player in players:
            result = classifier.classify(
                frame,
                player["bbox"],
            )

            calibrator.observe(
                result["jersey_color"],
                result["confidence"],
            )

        if frame_index >= 149:
            break

    reader.release()

    mapping = calibrator.calibrate()

    print(f"Team mapping: {mapping}")

    if mapping is None:
        print("ERROR: Team calibration failed.")
        return

    # --------------------------------------------------
    # STEP 2: Full tracking integration
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 2: FULL TRACKING INTEGRATION")
    print("=" * 60)

    reader = VideoReader(VIDEO_PATH)

    ball_detected_count = 0
    possession_count = 0

    for frame_index, frame in reader.frames():

        # ----------------------------------------------
        # YOLO detection
        # ----------------------------------------------

        detections = detector.detect(frame)

        # ----------------------------------------------
        # ByteTrack
        # ----------------------------------------------

        raw_player_detections = [
            d for d in detections
            if d["class_name"] in {"player", "goalkeeper"}
        ]

        print(
            f"[DETECTION DEBUG] Frame {frame_index}: "
            f"raw_players={len(raw_player_detections)}, "
            f"total_detections={len(detections)}"
        )

        tracked_objects = tracker.update(detections)

        # DEBUG ONLY: inspect raw tracking IDs
        # before team classification.
        player_objects = [
            obj
            for obj in tracked_objects
            if obj["class_name"] in {"player", "goalkeeper"}
            and obj["track_id"] >= 0
        ]

        player_ids = sorted(
            obj["track_id"] for obj in player_objects
        )

        print(
            f"[TRACKING DEBUG] Frame {frame_index}: "
            f"players={len(player_objects)}, "
            f"track_ids={player_ids}"
        )

        # ----------------------------------------------
        # Team classification + persistence
        # ----------------------------------------------

        tracked_players = []

        for obj in tracked_objects:

            if obj["class_name"] not in {
                "player",
                "goalkeeper",
            }:
                continue

            track_id = obj["track_id"]

            if track_id < 0:
                continue

            result = classifier.classify(
                frame,
                obj["bbox"],
            )


            if track_id == 5 and frame_index in (17, 18, 19, 22):
                import cv2

                x1, y1, x2, y2 = map(int, obj["bbox"])
                crop = frame[
                    max(0, y1):min(frame.shape[0], y2),
                    max(0, x1):min(frame.shape[1], x2)
                ]

                if crop.size > 0:
                    cv2.imwrite(
                        f"debug_player5_frame{frame_index}.jpg",
                        crop
                    )


            if track_id == 5 and frame_index in (14, 17, 18, 19, 22):
                print(
                    f"[CROP DEBUG] frame={frame_index}, "
                    f"bbox={obj['bbox']}, "
                    f"jersey_color={result['jersey_color']}, "
                    f"confidence={result['confidence']:.3f}"
                )

            detected_team = calibrator.get_team(
                result["jersey_color"],
                result["confidence"],
            )

            persistent_team = team_persistence.update(
                track_id=track_id,
                team=detected_team,
                confidence=result["confidence"],
            )

            if track_id == 5:
                print(
                    f"[TEAM DEBUG] Frame={frame_index} "
                    f"detected_team={detected_team} "
                    f"jersey_color={result['jersey_color']} "
                    f"confidence={result['confidence']:.3f} "
                    f"persistent_team={persistent_team} "
                    f"history={team_persistence.get_history(track_id)}"
                )


            player = {
                **obj,
                "team": persistent_team,
                "jersey_color": result["jersey_color"],
            }

            tracked_players.append(player)

        # ----------------------------------------------
        # Ball detection + BallTracker
        # ----------------------------------------------

        ball_detection = detector.get_ball(detections)

        ball_state = ball_tracker.update(
            ball_detection,
            frame_index=frame_index,
        )

        if ball_detection is not None:
            ball_detected_count += 1

        # ----------------------------------------------
        # Possession
        # ----------------------------------------------

        # DEBUG ONLY: capture state before possession update.
        previous_candidate = (
            possession_tracker.candidate_player_id
        )
        previous_current = (
            possession_tracker.current_player_id
        )

        tracked_ids = {
            player["track_id"]
            for player in tracked_players
        }

        ball_available = (
            ball_state.get("ball_x") is not None
            and ball_state.get("ball_y") is not None
        )

        # Run the original algorithm unchanged.
        possession = possession_tracker.update(
            ball_state=ball_state,
            players=tracked_players,
        )

        # Report candidate transitions for diagnosis.
        if (
            previous_candidate is not None
            and previous_candidate
            != possession_tracker.candidate_player_id
        ):
            print(
                "\n[POSSESSION TRANSITION]"
                f"\n  Frame: {frame_index}"
                f"\n  Ball available: {ball_available}"
                f"\n  Previous candidate: {previous_candidate}"
                f"\n  Previous candidate still tracked: "
                f"{previous_candidate in tracked_ids}"
                f"\n  Previous current player: {previous_current}"
                f"\n  Current player still tracked: "
                f"{previous_current in tracked_ids}"
                f"\n  New candidate: "
                f"{possession_tracker.candidate_player_id}"
                f"\n  Candidate frames: "
                f"{possession_tracker.candidate_frames}"
                f"\n  Missed frames: "
                f"{possession_tracker.candidate_missed_frames}"
                f"\n  Tracked IDs: {sorted(tracked_ids)}"
            )

        # Existing possession debug output.
        print(
            f"  Possession debug: "
            f"candidate={possession_tracker.candidate_player_id}, "
            f"candidate_frames={possession_tracker.candidate_frames}, "
            f"missed={possession_tracker.candidate_missed_frames}, "
            f"current={possession_tracker.current_player_id}, "
            f"possession_frames={possession_tracker.possession_frames}"
        )

        # ----------------------------------------------
        # Closest player distances
        # ----------------------------------------------

        if (
            ball_state.get("ball_x") is not None
            and ball_state.get("ball_y") is not None
            and tracked_players
        ):
            ball_point = (
                ball_state["ball_x"],
                ball_state["ball_y"],
            )

            distances = []

            for player in tracked_players:
                x1, y1, x2, y2 = player["bbox"]
                foot_point = (
                    (x1 + x2) / 2.0,
                    y2,
                )

                distance = possession_tracker._distance(
                    ball_point,
                    foot_point,
                )

                distances.append(
                    (
                        player["track_id"],
                        round(distance, 2),
                    )
                )

            distances.sort(key=lambda item: item[1])

            print("  Closest player distances:", distances[:3])

        if possession["possession_player_id"] is not None:
            possession_count += 1

        # ----------------------------------------------
        # Print frame summary
        # ----------------------------------------------

        print(f"\nFrame {frame_index}")

        print(f"  YOLO detections: {len(detections)}")

        print(f"  Tracked players: {len(tracked_players)}")

        print(
            f"  Ball: {ball_state['status']} "
            f"position=({ball_state['ball_x']}, "
            f"{ball_state['ball_y']})"
        )

        print(
            f"  Ball confidence: {ball_state['confidence']}"
        )

        print(
            f"  Possession player: "
            f"{possession['possession_player_id']}"
        )

        print(
            f"  Possession team: "
            f"{possession['possession_team']}"
        )

        # Print players with possession.

        for player in tracked_players:
            track_id = player["track_id"]

            has_ball = possession["has_ball"].get(
                track_id,
                False,
            )

            if has_ball:
                print(
                    f"  HAS BALL -> "
                    f"ID={track_id} "
                    f"player_team={player['team']} "
                    f"possession_team={possession['possession_team']} "
                    f"teams_match="
                    f"{player['team'] == possession['possession_team']}"
                )


        # Test only the first 150 frames.
        if frame_index >= 149:
            break

    reader.release()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FULL INTEGRATION TEST SUMMARY")
    print("=" * 60)

    print(f"Frames tested: {frame_index + 1}")

    print(
        f"Frames with ball detection: {ball_detected_count}"
    )

    print(
        f"Frames with possession: {possession_count}"
    )

    print(f"Team mapping: {mapping}")

    print("\nFull tracking integration test completed.")


if __name__ == "__main__":
    main()
