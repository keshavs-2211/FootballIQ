
from utils.field_filter import FieldFilter
from collections import Counter, defaultdict, deque
from pathlib import Path
import csv
import cv2

from config import (
    BALL_CLASS,
    PLAYER_CLASSES,
    REFEREE_CLASS,
    CSV_OUTPUT_DIR,
)
from analytics.team import TeamCalibrator
from analytics.team_persistence import TeamPersistence
from utils.video_reader import VideoReader
from utils.video_writer import VideoWriter
from detector.yolo_detector import YOLODetector
from detector.jersey_classifier import JerseyClassifier
from tracker.bytetrack_tracker import ByteTrackTracker
from tracker.ball_tracker import BallTracker
from visualization.annotator import annotate_frame


def process_video(input_path: str, output_path: str):
    """Detect, track, classify jerseys, assign teams, and export coordinates."""

    input_path = str(Path(input_path).resolve())
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = Path(CSV_OUTPUT_DIR) / (
        f"{output_path.stem}_coordinates.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(input_path)
    fps = reader.fps if reader.fps > 0 else 25.0

    print("Video information:")
    print(f"Frames: {reader.frame_count}")
    print(f"FPS: {fps}")
    print(f"Resolution: {reader.width}x{reader.height}")

    writer = None
    detector = None

    tracker = ByteTrackTracker(frame_rate=max(1, round(fps)))
    ball_tracker = BallTracker()
    jersey_classifier = JerseyClassifier()
    field_filter = FieldFilter()

    team_calibrator = TeamCalibrator(minimum_observations=20)
    team_persistence = TeamPersistence()

    # Stabilize jersey color independently for each tracked player.
    jersey_history = defaultdict(lambda: deque(maxlen=10))

    processed_frames = 0
    total_detections = 0
    player_count = 0
    goalkeeper_count = 0
    referee_count = 0
    ball_count = 0

    fieldnames = [
        "frame",
        "timestamp_seconds",
        "track_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "foot_x",
        "foot_y",
        "normalized_x",
        "normalized_y",
        "jersey_color",
        "team",
    ]

    try:
        detector = YOLODetector()

        writer = VideoWriter(
            output_path=str(output_path),
            fps=fps,
            width=reader.width,
            height=reader.height,
        )

        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )
            csv_writer.writeheader()

            for frame_index, frame in reader.frames():
                detections = detector.detect(frame)
                total_detections += len(detections)

                frame_height, frame_width = frame.shape[:2]

                # Filter objects before tracking.
                trackable = [
                    detection
                    for detection in detections
                    if detection["class_name"] != BALL_CLASS
                    and field_filter.contains_bbox(
                        detection["bbox"],
                        frame_width,
                        frame_height,
                    )
                ]

                # Handle the ball separately because it may be airborne.
                ball_detection = detector.get_ball(detections)

                if ball_detection is not None:
                    bx1, by1, bx2, by2 = map(
                        float,
                        ball_detection["bbox"],
                    )
                    ball_x = (bx1 + bx2) / 2
                    ball_y = (by1 + by2) / 2

                    polygon = field_filter.get_polygon(
                        frame_width,
                        frame_height,
                    )

                    if cv2.pointPolygonTest(
                        polygon,
                        (float(ball_x), float(ball_y)),
                        False,
                    ) < 0:
                        ball_detection = None

                ball_state = ball_tracker.update(
                    ball_detection,
                    frame_index,
                )
                tracked_objects = tracker.update(trackable)

                for obj in tracked_objects:
                    class_name = obj["class_name"]
                    track_id = obj["track_id"]

                    x1, y1, x2, y2 = map(
                        float,
                        obj["bbox"],
                    )

                    # Bottom-center approximates ground contact.
                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2

                    jersey_color = "Unknown"
                    team = "Unknown"

                    # Only outfield players contribute to team calibration.
                    if class_name == "player":
                        result = jersey_classifier.classify(
                            frame,
                            obj["bbox"],
                        )

                        observed_color = result["jersey_color"]
                        confidence = float(result["confidence"])

                        if (
                            observed_color != "Unknown"
                            and confidence >= 0.55
                        ):
                            jersey_history[track_id].append(
                                observed_color
                            )

                        if jersey_history[track_id]:
                            jersey_color = Counter(
                                jersey_history[track_id]
                            ).most_common(1)[0][0]

                        obj["jersey_color"] = jersey_color

                        # Add reliable observations to calibration.
                        if (
                            jersey_color != "Unknown"
                            and confidence >= 0.55
                        ):
                            team_calibrator.observe(
                                jersey_color,
                                confidence,
                            )

                        # Calibration remains unavailable until enough
                        # observations and at least two colors are present.
                        team_calibrator.calibrate()

                        team_candidate = team_calibrator.get_team(
                            jersey_color,
                            confidence,
                        )

                        team = team_persistence.update(
                            track_id,
                            team_candidate,
                            confidence,
                        )

                        obj["team"] = team
                        player_count += 1

                    elif class_name == "goalkeeper":
                        goalkeeper_count += 1
                        obj["team"] = "Unknown"
                        obj["jersey_color"] = "Unknown"

                    elif class_name == REFEREE_CLASS:
                        referee_count += 1
                        obj["team"] = "Unknown"
                        obj["jersey_color"] = "Unknown"

                    else:
                        obj["team"] = "Unknown"
                        obj["jersey_color"] = "Unknown"

                    csv_writer.writerow({
                        "frame": frame_index,
                        "timestamp_seconds": round(
                            frame_index / fps,
                            3,
                        ),
                        "track_id": track_id,
                        "class_name": class_name,
                        "confidence": round(
                            float(obj["confidence"]),
                            4,
                        ),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "foot_x": round(foot_x, 2),
                        "foot_y": round(foot_y, 2),
                        "normalized_x": round(
                            foot_x / frame_width,
                            6,
                        ),
                        "normalized_y": round(
                            foot_y / frame_height,
                            6,
                        ),
                        "jersey_color": jersey_color,
                        "team": team,
                    })

                if ball_detection is not None:
                    ball_count += 1

                frame_with_boundary = field_filter.draw_boundary(
                    frame.copy()
                )

                annotated = annotate_frame(
                    frame_with_boundary,
                    tracked_objects,
                    ball_state,
                )
                writer.write(annotated)
                processed_frames += 1

                if processed_frames % 25 == 0:
                    print(
                        f"Processed {processed_frames}/"
                        f"{reader.frame_count} frames"
                    )

    finally:
        reader.release()

        if writer is not None:
            writer.release()

    summary = {
        "frames_processed": processed_frames,
        "total_detections": total_detections,
        "tracked_player_detections": player_count,
        "tracked_goalkeeper_detections": goalkeeper_count,
        "tracked_referee_detections": referee_count,
        "frames_with_ball_detection": ball_count,
        "team_color_mapping": team_calibrator.get_mapping(),
        "output_path": str(output_path),
        "coordinates_csv": str(csv_path),
    }

    print("\nProcessing complete.")

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


if __name__ == "__main__":
    process_video(
        "sample_data/input_videos/08fd33_4.mp4",
        "outputs/videos/footballiq_annotated.mp4",
    )
