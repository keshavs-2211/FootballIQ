"""FootballIQ reference-based player detection and tracking (CPU friendly).

Run from project root:
    python process_video.py
"""

from collections import Counter, defaultdict, deque
from pathlib import Path
import csv

import cv2
import pyarrow as pa
import pyarrow.parquet as pq

from config import CSV_OUTPUT_DIR, TEAM_A_NAME, TEAM_B_NAME
from detection_settings import (
    ACTIVE_PRESET, SETTINGS, TEAM_HISTORY_LENGTH, TEAM_MIN_VOTES,
    TEAM_DECISION_RATIO, EXPORT_UNKNOWN_PLAYERS,
    SHOW_PITCH_DEBUG, USE_LEGACY_PITCH_FILTER,
)
from detector.yolo_detector import YOLODetector
from detector.jersey_classifier import JerseyClassifier, select_reference_images
from tracker.bytetrack_tracker import ByteTrackTracker
from tracker.ball_tracker import BallTracker
from player_identity import ActivePlayerManager
from utils.video_reader import VideoReader
from utils.video_writer import VideoWriter
from visualization.annotator import annotate_frame

cv2.setNumThreads(2)
CSV_COLUMNS = [
    "frame", "timestamp_seconds", "track_id", "raw_track_id", "class_name",
    "confidence", "x1", "y1", "x2", "y2", "foot_x", "foot_y",
    "normalized_x", "normalized_y", "jersey_color", "team",
]
SCHEMA = pa.schema([
    ("frame", pa.int64()), ("timestamp_seconds", pa.float64()),
    ("track_id", pa.int64()), ("raw_track_id", pa.int64()), ("class_name", pa.string()),
    ("confidence", pa.float64()),
    ("x1", pa.float64()), ("y1", pa.float64()),
    ("x2", pa.float64()), ("y2", pa.float64()),
    ("foot_x", pa.float64()), ("foot_y", pa.float64()),
    ("normalized_x", pa.float64()), ("normalized_y", pa.float64()),
    ("jersey_color", pa.string()), ("team", pa.string()),
])
PARQUET_BATCH_SIZE = 1000
GK_MIN_VOTES = 2
GK_TIMEOUT_FRAMES = 45


def _field_filter(use_legacy):
    if not use_legacy:
        return None
    from utils.field_filter import FieldFilter
    try:
        return FieldFilter(enabled=True)
    except TypeError:
        return FieldFilter()


def _pitch_debug_overlay(image):
    """Compatible with both versions of AutomaticFieldDetector."""
    try:
        from utils.automatic_field_detector import AutomaticFieldDetector
        try:
            detector = AutomaticFieldDetector(max_width=640)
        except TypeError:
            detector = AutomaticFieldDetector()
        candidate, _ = detector.detect(image)
        if candidate is not None:
            image = detector.draw(image, candidate)
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        print(f"Pitch debug unavailable for this frame: {error}")
    return image


class StableJerseyVotes:
    """Confidence-weighted, rolling vote per track ID."""

    def __init__(self):
        self.history = defaultdict(lambda: deque(maxlen=TEAM_HISTORY_LENGTH))
        self.last_label = {}

    def observe(self, track_id, classification):
        label = classification.get("jersey_color", "Unknown")
        score = float(classification.get("confidence", 0.0))
        if label not in ("Unknown", "") and score > 0:
            self.history[track_id].append((label, score))

    def label(self, track_id):
        votes = self.history.get(track_id, ())
        if not votes:
            return "Unknown"
        scores = defaultdict(float)
        counts = Counter()
        for label, confidence in votes:
            scores[label] += confidence
            counts[label] += 1
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        winner, score = ranked[0]
        next_score = ranked[1][1] if len(ranked) > 1 else 0.0
        previous = self.last_label.get(track_id)
        if counts[winner] < TEAM_MIN_VOTES:
            return previous or "Unknown"
        # Prevent team flicker when the top two references are ambiguous.
        if score < TEAM_DECISION_RATIO * next_score:
            return previous or "Unknown"
        # Switching teams requires stronger evidence than first assignment.
        if previous and previous != winner and score < 1.65 * next_score:
            return previous
        self.last_label[track_id] = winner
        return winner

    def forget(self, track_id):
        self.history.pop(track_id, None)
        self.last_label.pop(track_id, None)


class GoalkeeperManager:
    """At most one active, repeatedly confirmed goalkeeper ID per team."""

    def __init__(self, references):
        self.references = set(references)
        self.slots = {TEAM_A_NAME: None, TEAM_B_NAME: None}
        self.votes = defaultdict(Counter)
        self.last_seen = {}

    def observe(self, track_id, yolo_class, reference, was_classified):
        if (
            not was_classified
            or yolo_class not in ("goalkeeper", "player")
        ):
            return
        mapping = {"Team A GK": TEAM_A_NAME, "Team B GK": TEAM_B_NAME}
        if reference in mapping and reference in self.references:
            self.votes[track_id][mapping[reference]] += 1

    def resolve(self, objects, frame_index):
        visible = {int(item["track_id"]): item for item in objects}
        for team, track_id in self.slots.items():
            if track_id in visible:
                self.last_seen[track_id] = frame_index
            elif track_id is not None and frame_index - self.last_seen.get(
                track_id, frame_index
            ) > GK_TIMEOUT_FRAMES:
                self.slots[team] = None

        for name, reference in ((TEAM_A_NAME, "Team A GK"),
                                (TEAM_B_NAME, "Team B GK")):
            if reference not in self.references or self.slots[name] is not None:
                continue
            candidates = []
            for track_id, obj in visible.items():
                if track_id in self.slots.values():
                    continue
                count = self.votes[track_id][name]
                if count >= GK_MIN_VOTES and self.votes[track_id].most_common(1)[0][0] == name:
                    candidates.append((count, float(obj.get("confidence", 0)), track_id))
            if candidates:
                candidates.sort(key=lambda v: (-v[0], -v[1], v[2]))
                chosen = candidates[0][2]
                self.slots[name] = chosen
                self.last_seen[chosen] = frame_index
        return {tid: team for team, tid in self.slots.items()
                if tid is not None and tid in visible}

    def forget(self, track_id):
        if track_id not in self.slots.values():
            self.votes.pop(track_id, None)
            self.last_seen.pop(track_id, None)


def process_video(
    input_path,
    output_path,
    show_pitch_debug=SHOW_PITCH_DEBUG,
    use_legacy_pitch_filter=USE_LEGACY_PITCH_FILTER,
    reference_paths=None,
):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_dir = Path(CSV_OUTPUT_DIR)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{output_path.stem}_coordinates.csv"
    parquet_path = csv_dir / f"{output_path.stem}_coordinates.parquet"
    diagnostics_path = csv_dir / f"{output_path.stem}_detection_summary.csv"

    if reference_paths is None:
        # Both the terminal and website use saved references; no ROI terminal prompts.
        from detector.jersey_classifier import _saved_references
        folder = Path("references") / input_path.stem
        references = _saved_references(folder)
    else:
        references = reference_paths
    if len(references.get("Referee", [])) < 3:
        raise ValueError("Three referee references required for player processing")
    if not references.get("Team A") or not references.get("Team B"):
        print("Team jersey references incomplete: team assignment may be Unknown")
    classifier = JerseyClassifier(
        references,
        min_similarity=SETTINGS["jersey_min_similarity"],
        min_margin=SETTINGS["jersey_min_margin"],
    )
    reader = VideoReader(str(input_path))
    fps = float(reader.fps) if reader.fps > 0 else 25.0
    tracker = ByteTrackTracker(
        frame_rate=max(1, round(fps)),
        track_activation_threshold=SETTINGS["track_activation"],
        lost_track_buffer=SETTINGS["lost_buffer"],
        minimum_matching_threshold=SETTINGS["tracker_iou"],
    )
    ball_tracker = BallTracker()
    field_filter = _field_filter(use_legacy_pitch_filter)
    votes = StableJerseyVotes()
    goalkeepers = GoalkeeperManager(references.keys())
    active_players = ActivePlayerManager(fps=fps)
    last_classified = {}
    last_seen = {}
    stats = Counter()
    parquet_batch = []
    video_writer = None
    parquet_writer = None
    print(f"\nFootballIQ | preset={ACTIVE_PRESET} | FPS={fps:.2f} | "
          f"frames={reader.frame_count}")
    print("Standard ByteTrack; no custom ID recovery")
    print("Unknown players:", "exported" if EXPORT_UNKNOWN_PLAYERS else "video only")
    print("Recovery scan every", SETTINGS["recovery_every"], "frames" if SETTINGS["enable_recovery"] else "(OFF)")

    try:
        detector = YOLODetector(enable_recovery=SETTINGS["enable_recovery"])
        video_writer = VideoWriter(
            output_path=str(output_path), fps=fps,
            width=reader.width, height=reader.height,
        )
        parquet_writer = pq.ParquetWriter(
            str(parquet_path), SCHEMA, compression="snappy"
        )
        with (
            csv_path.open("w", encoding="utf-8", newline="") as csv_file,
            diagnostics_path.open("w", encoding="utf-8", newline="") as diag_file,
        ):
            coords = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            diag = csv.DictWriter(diag_file, fieldnames=[
                "frame", "yolo_players", "yolo_goalkeepers", "yolo_referees",
                "yolo_balls", "tracked", "classified", "unknown",
                "raw_tracks", "distinct_candidates", "duplicate_tracks_removed",
                "active_players", "cap_or_team_filtered", "reid_matches",
                "active_team_a", "active_team_b", "active_unknown",
            ])
            coords.writeheader()
            diag.writeheader()
            for frame_index, frame in reader.frames():
                detections = detector.detect(frame, frame_index=frame_index)
                stats["frames"] += 1
                for item in detections:
                    stats["yolo_" + item["class_name"]] += 1

                h, w = frame.shape[:2]
                trackable = []
                ignored_referees = 0

                for detection in detections:

                    label = str(
                        detection["class_name"]
                    ).lower()

                    # Direct YOLO referee detection.
                    if label == "referee":
                        ignored_referees += 1
                        continue

                    if label not in ("player", "goalkeeper"):
                        continue

                    if (
                        field_filter is not None
                        and not field_filter.contains_bbox(
                            detection["bbox"], w, h
                        )
                    ):
                        continue

                    # Reference-based referee verification.
                    if classifier.is_referee(
                        frame, detection["bbox"]
                    ):
                        ignored_referees += 1
                        continue

                    trackable.append(detection)

                stats["referee_exclusions"] += ignored_referees

                tracked = tracker.update(trackable)
                ball = detector.get_ball(detections)
                if ball is not None and field_filter is not None:
                    x1, y1, x2, y2 = ball["bbox"]
                    if not field_filter.contains_point((x1+x2)/2, (y1+y2)/2, w, h):
                        ball = None
                # Ball detection is handled by the
                # separate reference-aware tracking stage.
                # Do not draw unverified YOLO ball markers.
                ball_state = None
                classified_now = 0
                for obj in tracked:
                    tid = int(obj["track_id"])
                    last_seen[tid] = frame_index
                    fresh = (
                        tid not in last_classified
                        or frame_index - last_classified[tid] >= SETTINGS["jersey_every"]
                    )
                    result = {"jersey_color": "Unknown", "confidence": 0.0}
                    if fresh:
                        result = classifier.classify(frame, obj["bbox"])
                        votes.observe(tid, result)
                        last_classified[tid] = frame_index
                        classified_now += 1
                    goalkeepers.observe(
                        tid,
                        str(obj["class_name"]).lower(),
                        result["jersey_color"],
                        fresh,
                    )
                confirmed_gk = goalkeepers.resolve(tracked, frame_index)

                # Enrich raw ByteTrack objects before applying the 22-player
                # roster, deduplication and conservative identity continuity.
                enriched = []
                for obj in tracked:
                    tid = int(obj["track_id"])
                    ref = votes.label(tid)
                    original_role = str(obj.get("class_name", "player")).lower()
                    if tid in confirmed_gk:
                        role = "goalkeeper"
                        team = confirmed_gk[tid]
                        ref = "Team A GK" if team == TEAM_A_NAME else "Team B GK"
                    else:
                        role = "goalkeeper" if original_role == "goalkeeper" else "player"
                        team = TEAM_A_NAME if ref == "Team A" else TEAM_B_NAME if ref == "Team B" else "Unknown"
                        if ref in ("Team A GK", "Team B GK"):
                            team = "Unknown"
                    annotated_obj = dict(obj)
                    annotated_obj.update({"class_name": role, "team": team,
                                          "jersey_color": ref})
                    enriched.append(annotated_obj)

                draw_objects, roster_diag = active_players.select(frame_index, enriched, frame)
                known_in_frame = 0
                unknown_in_frame = 0
                for obj in draw_objects:
                    tid = int(obj["track_id"])  # Stable FootballIQ identity
                    team = obj["team"]
                    ref = obj["jersey_color"]
                    role = obj["class_name"]
                    if team == "Unknown":
                        unknown_in_frame += 1
                        stats["unknown"] += 1
                        if not EXPORT_UNKNOWN_PLAYERS:
                            continue
                    else:
                        known_in_frame += 1
                        stats["confirmed"] += 1
                    x1, y1, x2, y2 = map(float, obj["bbox"])
                    foot_x, foot_y = (x1+x2)/2.0, y2
                    row = {
                        "frame": frame_index,
                        "timestamp_seconds": round(frame_index/fps, 3),
                        "track_id": tid, "raw_track_id": int(obj["raw_track_id"]), "class_name": role,
                        "confidence": round(float(obj["confidence"]), 4),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "foot_x": round(foot_x, 2), "foot_y": round(foot_y, 2),
                        "normalized_x": round(foot_x/max(w, 1), 6),
                        "normalized_y": round(foot_y/max(h, 1), 6),
                        "jersey_color": ref, "team": team,
                    }
                    coords.writerow(row)
                    parquet_batch.append(row)
                    if len(parquet_batch) >= PARQUET_BATCH_SIZE:
                        parquet_writer.write_table(pa.Table.from_pylist(parquet_batch, schema=SCHEMA))
                        parquet_batch.clear()

                diag.writerow({
                    "frame": frame_index,
                    "yolo_players": sum(d["class_name"] == "player" for d in detections),
                    "yolo_goalkeepers": sum(d["class_name"] == "goalkeeper" for d in detections),
                    "yolo_referees": sum(d["class_name"] == "referee" for d in detections),
                    "yolo_balls": sum(d["class_name"] == "ball" for d in detections),
                    "tracked": len(tracked),
                    "classified": known_in_frame,
                    "unknown": unknown_in_frame,
                    **roster_diag,
                })

                display = frame.copy()
                if show_pitch_debug and frame_index % 10 == 0:
                    display = _pitch_debug_overlay(display)
                annotated = annotate_frame(display, draw_objects, ball_state)
                video_writer.write(annotated)
                if frame_index and frame_index % 50 == 0:
                    print(f"{frame_index}/{reader.frame_count} | "
                          f"tracked={len(tracked)}, confirmed={known_in_frame}, "
                          f"unknown={unknown_in_frame}")
                if frame_index and frame_index % 500 == 0:
                    for tid in list(last_seen):
                        if frame_index - last_seen[tid] > 250:
                            votes.forget(tid)
                            goalkeepers.forget(tid)
                            last_classified.pop(tid, None)
                            last_seen.pop(tid, None)
            if parquet_batch:
                parquet_writer.write_table(pa.Table.from_pylist(parquet_batch, schema=SCHEMA))
                parquet_batch.clear()
    finally:
        reader.release()
        if video_writer is not None:
            video_writer.release()
        if parquet_writer is not None:
            parquet_writer.close()

    print("\nProcessed frames:", stats["frames"])
    print("Confirmed track observations:", stats["confirmed"])
    print("Unknown track observations:", stats["unknown"])
    print("Annotated video:", output_path)
    print("Tracking CSV:", csv_path)
    print("Tracking Parquet:", parquet_path)
    print("Detection counts CSV:", diagnostics_path)
    return dict(stats)


if __name__ == "__main__":
    process_video(
        "sample_data/input_videos/bluevsred.mp4",
        "outputs/videos/bluevsred_improved.mp4",
        show_pitch_debug=False,
        use_legacy_pitch_filter=False,
    )
