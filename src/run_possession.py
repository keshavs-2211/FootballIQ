"""Generate per-frame possession timeline and summaries. Run after track_ball.py.

    python run_possession.py

Consumes outputs configured in offline_settings.py. Doesn't change original
tracking files, doesn't load YOLO, and uses constant-memory streaming.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import cv2

import possession_settings as P
from analytics.ball_possession import Config, PossessionEstimator, STATES, TEAM_A, TEAM_B


OUTPUT_COLUMNS = [
    "frame", "timestamp_seconds", "state", "possessing_player_id",
    "evidence_state", "reason", "ball_source", "ball_observed",
    "last_ball_observation_age_s", "ball_x_m", "ball_y_m",
    "nearest_player_distance_m",
]


def iter_csv_by_frame(path):
    """Yield (frame, list[dict]) assuming CSV is sorted by ascending frame."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        frame = None
        records = []
        for row in reader:
            try:
                number = int(row["frame"])
            except (ValueError, TypeError, KeyError):
                continue
            if frame is None:
                frame = number
            if number < frame:
                raise ValueError(f"CSV not sorted by frame: {path}")
            if number != frame:
                yield frame, records
                frame, records = number, []
            records.append(row)
        if frame is not None:
            yield frame, records


class FrameCursor:
    def __init__(self, path):
        self.it = iter(iter_csv_by_frame(path))
        self.next = next(self.it, None)

    def get(self, frame):
        while self.next is not None and self.next[0] < frame:
            self.next = next(self.it, None)
        if self.next is not None and self.next[0] == frame:
            data = self.next[1]
            self.next = next(self.it, None)
            return data
        return []


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if not (fps > 0 and frame_count > 0):
        raise RuntimeError("Unable to read FPS/length; verify VIDEO in offline_settings.py")
    return fps, frame_count


def run():
    for path, label in ((P.VIDEO, "source video"), (P.PLAYER_CSV, "player tracking CSV"),
                        (P.BALL_CSV, "ball tracking CSV")):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {label}: {path}. Run the earlier tracking stages first.")

    fps, frame_count = video_info(P.VIDEO)
    cfg = Config(
        fps=fps,
        control_radius=P.CONTROL_RADIUS_M,
        contested_margin=P.CONTESTED_DISTANCE_MARGIN_M,
        free_radius=P.FREE_BALL_RADIUS_M,
        confirm_frames=P.CONFIRM_FRAMES,
        max_occlusion_s=P.MAX_OCCLUSION_SECONDS,
        max_ball_control_speed=P.MAX_CONTROL_BALL_SPEED_MPS,
        estimated_radius_factor=P.ESTIMATED_PLAYER_RADIUS_FACTOR,
        ball_observation_sources=tuple(P.ACCEPT_BALL_SOURCES),
    )
    estimator = PossessionEstimator(cfg)
    player_reader = FrameCursor(P.PLAYER_CSV)
    ball_reader = FrameCursor(P.BALL_CSV)

    Path(P.TIMELINE_CSV).parent.mkdir(parents=True, exist_ok=True)
    Path(P.SUMMARY_CSV).parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    reasons = Counter()
    measured_ball_frames = 0
    ball_csv_frames = 0
    transitions = Counter()
    previous = "Unknown"

    with open(P.TIMELINE_CSV, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for frame in range(frame_count):
            players = player_reader.get(frame)
            ball_candidates = ball_reader.get(frame)
            ball = None
            if ball_candidates:
                ball_csv_frames += 1
                # The professional tracker usually emits one row per frame.
                # Prefer verified measurements when multiple rows exist.
                ball = next((b for b in ball_candidates
                             if str(b.get("source", "")).lower() in P.ACCEPT_BALL_SOURCES),
                            ball_candidates[0])
            result = estimator.update(frame, players, ball)
            writer.writerow(result)
            counts[result["state"]] += 1
            reasons[result["reason"]] += 1
            measured_ball_frames += result["ball_observed"]
            if result["state"] != previous:
                transitions[f"{previous} -> {result['state']}"] += 1
                previous = result["state"]
            if (frame + 1) % 3000 == 0:
                print(f"Possession frames processed: {frame+1}/{frame_count}")

    team_frames = counts[TEAM_A] + counts[TEAM_B]
    classified_frames = frame_count - counts["Unknown"]
    summary = {
        "clip": P.CLIP,
        "fps": round(fps, 4),
        "frames": frame_count,
        "duration_seconds": round(frame_count / fps, 2),
        "measured_ball_frames": measured_ball_frames,
        "ball_csv_frames": ball_csv_frames,
        "measured_ball_coverage_pct": round(100 * measured_ball_frames / frame_count, 2),
        "time_by_state_frames": {state: counts[state] for state in STATES},
        "time_by_state_seconds": {state: round(counts[state] / fps, 2) for state in STATES},
        "all_time_percentage": {state: round(100 * counts[state] / frame_count, 2) for state in STATES},
        "team_only_possession_pct": {
            TEAM_A: round(100 * counts[TEAM_A] / team_frames, 2) if team_frames else None,
            TEAM_B: round(100 * counts[TEAM_B] / team_frames, 2) if team_frames else None,
        },
        "two_team_assigned_coverage_pct": round(100 * team_frames / frame_count, 2),
        "non_unknown_coverage_pct": round(100 * classified_frames / frame_count, 2),
        "top_reasons": reasons.most_common(12),
        "state_transitions": dict(transitions),
        "warnings": [
            "Experimental proximity-based possession, not official verified match events.",
            "A predicted or missing ball never starts a new team possession.",
            "Unknown, Contested and Free ball are excluded from two-team percentages; check assigned coverage.",
        ],
    }
    if measured_ball_frames < P.MIN_REQUIRED_MEASURED_BALL_FRAMES:
        summary["warnings"].append("Very few ball measurements; possession results may be unusable.")
    if team_frames < 0.10 * frame_count:
        summary["warnings"].append("Under 10% of frames confidently assigned to teams; don't report team possession percentages as reliable.")

    Path(P.SUMMARY_JSON).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(P.SUMMARY_CSV, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["state", "frames", "seconds", "percent_of_all_frames"])
        for state in STATES:
            writer.writerow([state, counts[state], round(counts[state] / fps, 3),
                             round(100 * counts[state] / frame_count, 3)])
    print("\n=== FootballIQ Possession Estimation ===")
    print(f"Clip: {P.CLIP} | Duration: {frame_count/fps:.1f}s | Ball observations: {measured_ball_frames}/{frame_count}")
    for state in STATES:
        print(f"  {state:<11} {counts[state]:>6} frames ({100*counts[state]/frame_count:5.1f}%)")
    print("Team-only split:", summary["team_only_possession_pct"])
    print("Two-team assigned coverage:", summary["two_team_assigned_coverage_pct"], "%")
    for warning in summary["warnings"]:
        print("NOTE:", warning)
    print("Timeline:", P.TIMELINE_CSV)
    print("Summary:", P.SUMMARY_CSV)
    print("Full report:", P.SUMMARY_JSON)


if __name__ == "__main__":
    run()
