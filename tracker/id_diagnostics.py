
import argparse
import csv
import math
from collections import Counter
from pathlib import Path


# ==========================================
# SETTINGS
# ==========================================

# More than 1 frame between observations
# indicates a tracking gap.
MIN_GAP_FRAMES = 2

# Report longer disappearances separately.
LONG_GAP_FRAMES = 5

# Allowed displacement is measured relative
# to the player's bounding-box height.
# This helps account for perspective.
MAX_MOVEMENT_PER_FRAME = 0.65

# Additional tolerance for detection noise.
POSITION_TOLERANCE = 0.50

# Don't evaluate very small / invalid boxes.
MIN_BOX_HEIGHT = 8.0


# ==========================================
# HELPERS
# ==========================================

def get_position(row):
    """
    Return bottom-center foot coordinates
    and bounding-box height.
    """

    x1 = float(row["x1"])
    y1 = float(row["y1"])
    x2 = float(row["x2"])
    y2 = float(row["y2"])

    foot_x = (x1 + x2) / 2.0
    foot_y = y2

    box_height = max(0.0, y2 - y1)

    return foot_x, foot_y, box_height


def write_event(writer, event_type, track_id,
                start_frame, end_frame,
                distance, allowed_distance,
                details):
    writer.writerow({
        "event_type": event_type,
        "track_id": track_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frames_elapsed": end_frame - start_frame,
        "distance_pixels": round(distance, 2),
        "allowed_pixels": round(allowed_distance, 2),
        "details": details,
    })


# ==========================================
# ANALYZER
# ==========================================

def analyze_tracking(input_csv, output_csv):

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(input_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    last_observation = {}
    counters = Counter()

    total_records = 0

    with open(
        input_csv,
        "r",
        encoding="utf-8",
        newline="",
    ) as source, open(
        output_csv,
        "w",
        encoding="utf-8",
        newline="",
    ) as destination:

        reader = csv.DictReader(source)

        required_columns = {
            "frame",
            "track_id",
            "class_name",
            "team",
            "x1",
            "y1",
            "x2",
            "y2",
        }

        missing = required_columns - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}"
            )

        writer = csv.DictWriter(
            destination,
            fieldnames=[
                "event_type",
                "track_id",
                "start_frame",
                "end_frame",
                "frames_elapsed",
                "distance_pixels",
                "allowed_pixels",
                "details",
            ],
        )

        writer.writeheader()

        last_frame = -1
        seen_this_frame = set()

        for row in reader:

            total_records += 1

            frame = int(row["frame"])
            track_id = int(row["track_id"])

            team = row.get(
                "team",
                "Unknown",
            )

            role = row.get(
                "class_name",
                "Unknown",
            )

            if frame < last_frame:
                raise ValueError(
                    "Input CSV must be ordered "
                    "by frame number."
                )

            if frame != last_frame:
                seen_this_frame.clear()
                last_frame = frame

            if track_id in seen_this_frame:
                counters["duplicate_id"] += 1

                write_event(
                    writer,
                    "DUPLICATE_ID",
                    track_id,
                    frame,
                    frame,
                    0,
                    0,
                    "Same tracking ID appears "
                    "multiple times in one frame.",
                )

            seen_this_frame.add(track_id)

            x, y, height = get_position(row)

            previous = last_observation.get(
                track_id
            )

            if previous is not None:

                previous_frame = previous["frame"]
                elapsed = frame - previous_frame

                if elapsed > 0:

                    dx = x - previous["x"]
                    dy = y - previous["y"]

                    distance = math.hypot(
                        dx,
                        dy,
                    )

                    reference_height = max(
                        MIN_BOX_HEIGHT,
                        height,
                        previous["height"],
                    )

                    allowed_distance = (
                        reference_height
                        * POSITION_TOLERANCE
                        +
                        reference_height
                        * MAX_MOVEMENT_PER_FRAME
                        * elapsed
                    )

                    # ----------------------
                    # GAP / REAPPEARANCE
                    # ----------------------

                    if elapsed >= MIN_GAP_FRAMES:

                        counters["tracking_gap"] += 1

                        if elapsed >= LONG_GAP_FRAMES:
                            event = "LONG_GAP"
                        else:
                            event = "SHORT_GAP"

                        write_event(
                            writer,
                            event,
                            track_id,
                            previous_frame,
                            frame,
                            distance,
                            allowed_distance,
                            "Track was absent from "
                            "the exported observations.",
                        )

                    # ----------------------
                    # SUSPICIOUS MOVEMENT
                    # ----------------------

                    if (
                        reference_height >= MIN_BOX_HEIGHT
                        and distance > allowed_distance
                    ):

                        counters["suspicious_jump"] += 1

                        write_event(
                            writer,
                            "SUSPICIOUS_JUMP",
                            track_id,
                            previous_frame,
                            frame,
                            distance,
                            allowed_distance,
                            (
                                f"Previous position="
                                f"({previous['x']:.1f}, "
                                f"{previous['y']:.1f}); "
                                f"new position=({x:.1f}, "
                                f"{y:.1f})."
                            ),
                        )

                    # ----------------------
                    # TEAM CHANGE
                    # ----------------------

                    previous_team = previous["team"]

                    if (
                        team != "Unknown"
                        and previous_team != "Unknown"
                        and team != previous_team
                    ):

                        counters["team_change"] += 1

                        write_event(
                            writer,
                            "TEAM_CHANGE",
                            track_id,
                            previous_frame,
                            frame,
                            distance,
                            allowed_distance,
                            (
                                f"{previous_team} changed "
                                f"to {team}."
                            ),
                        )

                    # ----------------------
                    # ROLE CHANGE
                    # ----------------------

                    if (
                        role != previous["role"]
                        and role != "Unknown"
                        and previous["role"] != "Unknown"
                    ):

                        counters["role_change"] += 1

                        write_event(
                            writer,
                            "ROLE_CHANGE",
                            track_id,
                            previous_frame,
                            frame,
                            distance,
                            allowed_distance,
                            (
                                f"{previous['role']} changed "
                                f"to {role}."
                            ),
                        )

            last_observation[track_id] = {
                "frame": frame,
                "x": x,
                "y": y,
                "height": height,
                "team": team,
                "role": role,
            }

    print("\n========== FOOTBALLIQ ID ANALYSIS ==========")

    print(f"Records analyzed: {total_records}")
    print(f"Unique tracking IDs: {len(last_observation)}")

    print(
        f"Tracking gaps: "
        f"{counters['tracking_gap']}"
    )

    print(
        f"Suspicious position jumps: "
        f"{counters['suspicious_jump']}"
    )

    print(
        f"Team changes: "
        f"{counters['team_change']}"
    )

    print(
        f"Role changes: "
        f"{counters['role_change']}"
    )

    print(
        f"Duplicate IDs in same frame: "
        f"{counters['duplicate_id']}"
    )

    print(f"\nDetailed report saved: {output_csv}")

    print(
        "\nNote: These are diagnostic warnings, "
        "not confirmed identity switches."
    )


# ==========================================
# COMMAND LINE
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Analyze FootballIQ tracking IDs "
            "using exported coordinate CSV."
        )
    )

    parser.add_argument(
        "input_csv",
        help="FootballIQ coordinates CSV",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path for diagnostics report",
    )

    args = parser.parse_args()

    input_path = Path(args.input_csv)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(
            input_path.stem + "_id_diagnostics.csv"
        )
    )

    analyze_tracking(
        input_path,
        output_path,
    )
