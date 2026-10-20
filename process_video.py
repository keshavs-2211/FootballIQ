
from collections import Counter, defaultdict, deque
from pathlib import Path
import csv

import cv2
import pyarrow as pa
import pyarrow.parquet as pq

from config import (
    CSV_OUTPUT_DIR,
    TEAM_A_NAME,
    TEAM_B_NAME,
)

from detector.yolo_detector import YOLODetector
from detector.jersey_classifier import (
    JerseyClassifier,
    select_reference_images,
)

from tracker.bytetrack_tracker import ByteTrackTracker
from tracker.ball_tracker import BallTracker

from utils.video_reader import VideoReader
from utils.video_writer import VideoWriter
from utils.field_filter import FieldFilter
from utils.automatic_field_detector import (
    AutomaticFieldDetector,
)

from visualization.annotator import annotate_frame


# ==========================================
# LOW-RAM SETTINGS
# ==========================================

cv2.setNumThreads(2)

JERSEY_INTERVAL = 5
HISTORY_LENGTH = 10
PARQUET_BATCH_SIZE = 1000

GOALKEEPER_MIN_VOTES = 3
GOALKEEPER_TIMEOUT = 45

PITCH_INTERVAL = 10

SHOW_PITCH_DEBUG = False
USE_LEGACY_PITCH_FILTER = False

# Set to True to include Unknown players
# in CSV and Parquet, not just the video.
EXPORT_UNKNOWN_PLAYERS = False


# ==========================================
# TEAM REFERENCE MAPPING
# ==========================================

OUTFIELD_TEAMS = {
    "Team A": TEAM_A_NAME,
    "Team B": TEAM_B_NAME,
}

GOALKEEPER_TEAMS = {
    "Team A GK": TEAM_A_NAME,
    "Team B GK": TEAM_B_NAME,
}


# ==========================================
# PARQUET SCHEMA
# ==========================================

SCHEMA = pa.schema([
    ("frame", pa.int64()),
    ("timestamp_seconds", pa.float64()),
    ("track_id", pa.int64()),
    ("class_name", pa.string()),
    ("confidence", pa.float64()),
    ("x1", pa.float64()),
    ("y1", pa.float64()),
    ("x2", pa.float64()),
    ("y2", pa.float64()),
    ("foot_x", pa.float64()),
    ("foot_y", pa.float64()),
    ("normalized_x", pa.float64()),
    ("normalized_y", pa.float64()),
    ("jersey_color", pa.string()),
    ("team", pa.string()),
])


def write_batch(writer, rows):
    """
    Export a small batch to Parquet.

    Avoids storing all match coordinates in RAM.
    """
    if not rows:
        return

    table = pa.Table.from_pylist(
        rows,
        schema=SCHEMA,
    )

    writer.write_table(table)


# ==========================================
# GOALKEEPER MANAGER
# ==========================================

class GoalkeeperManager:
    """
    Assign at most one active goalkeeper ID
    to each team.

    Requires repeated matching goalkeeper
    reference observations and a YOLO
    goalkeeper detection.

    Skipped goalkeeper references do not
    produce automatic team assignments.
    """

    def __init__(self, available_references):
        self.available_references = set(
            available_references
        )

        self.slots = {
            TEAM_A_NAME: None,
            TEAM_B_NAME: None,
        }

        self.last_seen = {}
        self.votes = defaultdict(Counter)

    def observe(
        self,
        track_id,
        original_class,
        reference_name,
        fresh,
    ):
        if not fresh:
            return

        if original_class != "goalkeeper":
            return

        if reference_name not in self.available_references:
            return

        if reference_name not in GOALKEEPER_TEAMS:
            return

        team = GOALKEEPER_TEAMS[reference_name]
        self.votes[track_id][team] += 1

    def assign(self, objects, frame_index):
        visible = {
            int(obj["track_id"]): obj
            for obj in objects
        }

        # Update existing assignments.
        for team, track_id in self.slots.items():
            if track_id is None:
                continue

            if track_id in visible:
                self.last_seen[track_id] = frame_index

            elif (
                frame_index
                - self.last_seen.get(
                    track_id,
                    frame_index,
                )
                > GOALKEEPER_TIMEOUT
            ):
                self.slots[team] = None

        # Fill currently vacant goalkeeper slots.
        for team in (TEAM_A_NAME, TEAM_B_NAME):

            if self.slots[team] is not None:
                continue

            reference_name = (
                "Team A GK"
                if team == TEAM_A_NAME
                else "Team B GK"
            )

            # Optional goalkeeper reference.
            if reference_name not in self.available_references:
                continue

            candidates = []

            for track_id, obj in visible.items():

                if track_id in self.slots.values():
                    continue

                votes = self.votes[track_id][team]

                if votes < GOALKEEPER_MIN_VOTES:
                    continue

                most_common = self.votes[
                    track_id
                ].most_common(1)

                if (
                    not most_common
                    or most_common[0][0] != team
                ):
                    continue

                candidates.append((
                    votes,
                    float(
                        obj.get("confidence", 0.0)
                    ),
                    track_id,
                ))

            if not candidates:
                continue

            candidates.sort(
                key=lambda item: (
                    -item[0],
                    -item[1],
                    item[2],
                )
            )

            chosen_id = candidates[0][2]

            self.slots[team] = chosen_id
            self.last_seen[chosen_id] = frame_index

        # Only return goalkeepers visible now.
        return {
            track_id: team
            for team, track_id in self.slots.items()
            if track_id is not None
            and track_id in visible
        }

    def cleanup(self, valid_ids):
        reserved_ids = {
            track_id
            for track_id in self.slots.values()
            if track_id is not None
        }

        for track_id in list(self.votes):

            if (
                track_id not in valid_ids
                and track_id not in reserved_ids
            ):
                del self.votes[track_id]


# ==========================================
# VIDEO PROCESSING
# ==========================================

def process_video(
    input_path,
    output_path,
    show_pitch_debug=SHOW_PITCH_DEBUG,
    use_legacy_pitch_filter=USE_LEGACY_PITCH_FILTER,
):

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input video not found: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = Path(CSV_OUTPUT_DIR) / (
        f"{output_path.stem}_coordinates.csv"
    )

    parquet_path = Path(CSV_OUTPUT_DIR) / (
        f"{output_path.stem}_coordinates.parquet"
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ======================================
    # STEP 1: SELECT REFERENCE JERSEYS
    # ======================================

    reference_dir = (
        Path("references") / input_path.stem
    )

    reference_paths = select_reference_images(
        str(input_path),
        reference_dir=reference_dir,
    )

    # Handle optional goalkeeper references.
    usable_references = {
        name: path
        for name, path in reference_paths.items()
        if path is not None and Path(path).is_file()
    }

    for required in ("Team A", "Team B"):
        if required not in usable_references:
            raise ValueError(
                f"Required reference missing: {required}"
            )

    jersey_classifier = JerseyClassifier(
        reference_paths=usable_references,
    )

    # ======================================
    # STEP 2: INITIALIZE VIDEO
    # ======================================

    reader = VideoReader(str(input_path))

    fps = (
        reader.fps
        if reader.fps > 0
        else 25.0
    )

    print("\n========== FOOTBALLIQ ==========")
    print(f"Input: {input_path}")
    print(f"Frames: {reader.frame_count}")
    print(f"FPS: {fps}")
    print(
        f"Resolution: "
        f"{reader.width}x{reader.height}"
    )

    print("\nProcessing configuration:")
    print("YOLO detection: Enabled")
    print("ByteTrack: Standard")
    print("Custom predictor: Disabled")
    print("Referee tracking: Disabled")
    print("Reference jersey classification: Enabled")
    print("Unknown player visualization: Enabled")
    print(f"Pitch debug: {show_pitch_debug}")
    print(f"Legacy field filter: {use_legacy_pitch_filter}")
    print(f"Parquet batch size: {PARQUET_BATCH_SIZE}")

    writer = None
    parquet_writer = None

    # ======================================
    # STEP 3: INITIALIZE COMPONENTS
    # ======================================

    tracker = ByteTrackTracker(
        frame_rate=max(1, round(fps))
    )

    ball_tracker = BallTracker()

    field_filter = FieldFilter(
        enabled=use_legacy_pitch_filter,
    )

    pitch_detector = (
        AutomaticFieldDetector(max_width=640)
        if show_pitch_debug
        else None
    )

    pitch_segments = []

    goalkeeper_manager = GoalkeeperManager(
        available_references=usable_references.keys()
    )

    # ======================================
    # TRACKING HISTORY
    # ======================================

    reference_history = defaultdict(
        lambda: deque(maxlen=HISTORY_LENGTH)
    )

    last_classified = {}
    last_seen = {}

    # ======================================
    # STATISTICS
    # ======================================

    processed_frames = 0
    total_detections = 0

    player_count = 0
    goalkeeper_count = 0
    unknown_count = 0

    ball_count = 0
    excluded_referee_count = 0

    coordinate_batch = []

    try:

        # ==================================
        # STEP 4: START YOLO
        # ==================================

        detector = YOLODetector()

        writer = VideoWriter(
            output_path=str(output_path),
            fps=fps,
            width=reader.width,
            height=reader.height,
        )

        parquet_writer = pq.ParquetWriter(
            str(parquet_path),
            schema=SCHEMA,
            compression="snappy",
        )

        # ==================================
        # STEP 5: STREAM COORDINATE CSV
        # ==================================

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=SCHEMA.names,
            )

            csv_writer.writeheader()

            # ==============================
            # FRAME LOOP
            # ==============================

            for frame_index, frame in reader.frames():

                # --------------------------
                # YOLO DETECTION
                # --------------------------

                detections = detector.detect(frame)

                total_detections += len(detections)

                frame_height, frame_width = (
                    frame.shape[:2]
                )

                # --------------------------
                # REFEREE FILTERING
                # --------------------------

                excluded_referee_count += sum(
                    1
                    for detection in detections
                    if str(
                        detection["class_name"]
                    ).lower() == "referee"
                )

                trackable = [
                    detection
                    for detection in detections
                    if str(
                        detection["class_name"]
                    ).lower() in (
                        "player",
                        "goalkeeper",
                    )
                    and field_filter.contains_bbox(
                        detection["bbox"],
                        frame_width,
                        frame_height,
                    )
                ]

                # --------------------------
                # BALL TRACKING
                # --------------------------

                ball_detection = detector.get_ball(
                    detections
                )

                if ball_detection is not None:

                    bx1, by1, bx2, by2 = map(
                        float,
                        ball_detection["bbox"],
                    )

                    ball_x = (bx1 + bx2) / 2.0
                    ball_y = (by1 + by2) / 2.0

                    if not field_filter.contains_point(
                        ball_x,
                        ball_y,
                        frame_width,
                        frame_height,
                    ):
                        ball_detection = None

                ball_state = ball_tracker.update(
                    ball_detection,
                    frame_index,
                )

                if ball_detection is not None:
                    ball_count += 1

                # --------------------------
                # STANDARD BYTETRACK
                # --------------------------

                tracked_objects = tracker.update(
                    trackable
                )

                # --------------------------
                # JERSEY CLASSIFICATION
                # --------------------------

                stable_references = {}

                for obj in tracked_objects:

                    track_id = int(
                        obj["track_id"]
                    )

                    last_seen[track_id] = frame_index

                    original_class = str(
                        obj["class_name"]
                    ).lower()

                    fresh = (
                        track_id not in last_classified
                        or frame_index
                        - last_classified[track_id]
                        >= JERSEY_INTERVAL
                    )

                    observed_reference = "Unknown"

                    if fresh:

                        result = jersey_classifier.classify(
                            frame,
                            obj["bbox"],
                        )

                        last_classified[track_id] = (
                            frame_index
                        )

                        observed_reference = result.get(
                            "jersey_color",
                            "Unknown",
                        )

                        if (
                            observed_reference
                            in usable_references
                        ):
                            reference_history[
                                track_id
                            ].append(
                                observed_reference
                            )

                    history = reference_history[
                        track_id
                    ]

                    stable_reference = (
                        Counter(
                            history
                        ).most_common(1)[0][0]
                        if history
                        else "Unknown"
                    )

                    stable_references[track_id] = (
                        stable_reference
                    )

                    goalkeeper_manager.observe(
                        track_id=track_id,
                        original_class=original_class,
                        reference_name=observed_reference,
                        fresh=fresh,
                    )

                # --------------------------
                # GOALKEEPER ASSIGNMENT
                # --------------------------

                confirmed_goalkeepers = (
                    goalkeeper_manager.assign(
                        tracked_objects,
                        frame_index,
                    )
                )

                # --------------------------
                # TEAM AND ROLE ASSIGNMENT
                # --------------------------

                visible_objects = []

                for obj in tracked_objects:

                    track_id = int(
                        obj["track_id"]
                    )

                    reference_name = (
                        stable_references.get(
                            track_id,
                            "Unknown",
                        )
                    )

                    # Always initialize both role
                    # and team before using them.
                    role = "player"
                    team = "Unknown"

                    if track_id in confirmed_goalkeepers:

                        role = "goalkeeper"

                        team = (
                            confirmed_goalkeepers[
                                track_id
                            ]
                        )

                        reference_name = (
                            "Team A GK"
                            if team == TEAM_A_NAME
                            else "Team B GK"
                        )

                    else:

                        # Unconfirmed goalkeeper-class
                        # objects remain player candidates.
                        role = "player"

                        team = OUTFIELD_TEAMS.get(
                            reference_name,
                            "Unknown",
                        )

                    # ----------------------
                    # ADD ALL TRACKED OBJECTS
                    # TO THE ANNOTATED VIDEO
                    # ----------------------

                    obj["class_name"] = role
                    obj["team"] = team
                    obj["jersey_color"] = (
                        reference_name
                    )

                    visible_objects.append(obj)

                    # ----------------------
                    # HANDLE UNKNOWN PLAYERS
                    # ----------------------

                    if team == "Unknown":
                        unknown_count += 1

                        # Draw grey boxes but exclude
                        # uncertain identities from
                        # confirmed coordinate exports.
                        if not EXPORT_UNKNOWN_PLAYERS:
                            continue

                    # ----------------------
                    # STATISTICS
                    # ----------------------

                    if team != "Unknown":

                        if role == "goalkeeper":
                            goalkeeper_count += 1
                        else:
                            player_count += 1

                    # ----------------------
                    # COORDINATE EXTRACTION
                    # ----------------------

                    x1, y1, x2, y2 = map(
                        float,
                        obj["bbox"],
                    )

                    foot_x = (x1 + x2) / 2.0
                    foot_y = y2

                    coordinate_row = {
                        "frame": frame_index,

                        "timestamp_seconds": round(
                            frame_index / fps,
                            3,
                        ),

                        "track_id": track_id,
                        "class_name": role,

                        "confidence": round(
                            float(
                                obj["confidence"]
                            ),
                            4,
                        ),

                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),

                        "foot_x": round(
                            foot_x,
                            2,
                        ),

                        "foot_y": round(
                            foot_y,
                            2,
                        ),

                        "normalized_x": round(
                            foot_x / frame_width,
                            6,
                        ),

                        "normalized_y": round(
                            foot_y / frame_height,
                            6,
                        ),

                        "jersey_color": reference_name,
                        "team": team,
                    }

                    # Stream to CSV.
                    csv_writer.writerow(
                        coordinate_row
                    )

                    # Keep only a small Parquet batch.
                    coordinate_batch.append(
                        coordinate_row
                    )

                    if (
                        len(coordinate_batch)
                        >= PARQUET_BATCH_SIZE
                    ):

                        write_batch(
                            parquet_writer,
                            coordinate_batch,
                        )

                        coordinate_batch.clear()

                # --------------------------
                # MEMORY CLEANUP
                # --------------------------

                if (
                    frame_index > 0
                    and frame_index % 500 == 0
                ):

                    stale_ids = [
                        track_id
                        for track_id, seen
                        in last_seen.items()
                        if frame_index - seen > 250
                    ]

                    for track_id in stale_ids:

                        last_seen.pop(
                            track_id,
                            None,
                        )

                        last_classified.pop(
                            track_id,
                            None,
                        )

                        reference_history.pop(
                            track_id,
                            None,
                        )

                    goalkeeper_manager.cleanup(
                        set(last_seen)
                    )

                # --------------------------
                # OPTIONAL PITCH DEBUG
                # --------------------------

                display_frame = frame.copy()

                if show_pitch_debug:

                    if frame_index % PITCH_INTERVAL == 0:

                        pitch_segments, _ = (
                            pitch_detector.detect(frame)
                        )

                    display_frame = (
                        pitch_detector.draw(
                            display_frame,
                            pitch_segments,
                        )
                    )

                # --------------------------
                # ANNOTATE VIDEO
                # --------------------------

                annotated = annotate_frame(
                    display_frame,
                    visible_objects,
                    ball_state,
                )

                writer.write(annotated)

                processed_frames += 1

                if processed_frames % 25 == 0:

                    print(
                        f"Processed "
                        f"{processed_frames}/"
                        f"{reader.frame_count} frames"
                    )

            # ------------------------------
            # FINAL PARQUET BATCH
            # ------------------------------

            if coordinate_batch:

                write_batch(
                    parquet_writer,
                    coordinate_batch,
                )

                coordinate_batch.clear()

    finally:

        reader.release()

        if writer is not None:
            writer.release()

        if parquet_writer is not None:
            parquet_writer.close()

    # ======================================
    # PROCESSING SUMMARY
    # ======================================

    summary = {
        "frames_processed": processed_frames,
        "total_detections": total_detections,

        "tracked_player_detections": (
            player_count
        ),

        "tracked_goalkeeper_detections": (
            goalkeeper_count
        ),

        "unknown_team_observations": (
            unknown_count
        ),

        "frames_with_ball_detection": (
            ball_count
        ),

        "referee_detections_excluded": (
            excluded_referee_count
        ),

        "goalkeeper_assignments": (
            goalkeeper_manager.slots
        ),

        "output_video": str(output_path),
        "coordinates_csv": str(csv_path),
        "coordinates_parquet": str(parquet_path),
    }

    print("\n========== PROCESSING COMPLETE ==========")

    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    process_video(
        "sample_data/input_videos/whitevsred.mp4",
        "outputs/videos/whitevsred_reference_debug.mp4",

        # True: show candidate pitch lines.
        # False: no pitch overlay.
        show_pitch_debug=True,

        # Keep the inaccurate old polygon off.
        use_legacy_pitch_filter=False,
    )
