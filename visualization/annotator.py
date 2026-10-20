
import cv2


TEAM_BOX_COLORS = {
    "Team A": (255, 180, 40),
    "Team B": (60, 70, 245),
    "Unknown": (165, 165, 165),
}


def annotate_frame(frame, detections, ball_state=None):
    """
    FootballIQ tracking visualization.

    Confirmed players:
        Team-coloured bounding boxes.

    Unclassified players:
        Grey bounding boxes with tracking IDs.

    Referees:
        Not drawn.

    Ball:
        Existing visualization retained.
    """

    annotated = frame.copy()

    for obj in detections:

        class_name = str(
            obj.get("class_name", "")
        ).lower()

        if class_name not in ("player", "goalkeeper"):
            continue

        team = obj.get("team", "Unknown")
        track_id = obj.get("track_id", "?")

        color = TEAM_BOX_COLORS.get(
            team,
            TEAM_BOX_COLORS["Unknown"],
        )

        x1, y1, x2, y2 = map(
            int,
            obj["bbox"],
        )

        if team == "Unknown":

            color = TEAM_BOX_COLORS["Unknown"]

            label = (
                f"ID #{track_id} | Unknown"
            )

            # Thinner grey box for uncertain tracks.
            thickness = 1

        else:

            role = (
                "GK"
                if class_name == "goalkeeper"
                else "Player"
            )

            label = (
                f"{role} #{track_id} | {team}"
            )

            thickness = 2

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

    # ======================================
    # BALL DISPLAY
    # ======================================

    if ball_state:

        status = ball_state.get(
            "status",
            "unavailable",
        )

        bx = ball_state.get("ball_x")
        by = ball_state.get("ball_y")

        if bx is not None and by is not None:

            center = (
                int(bx),
                int(by),
            )

            cv2.circle(
                annotated,
                center,
                7,
                (0, 0, 255),
                -1,
            )

            cv2.circle(
                annotated,
                center,
                11,
                (255, 255, 255),
                2,
            )

        cv2.putText(
            annotated,
            f"Ball: {status}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated
