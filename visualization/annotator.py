
import cv2


# OpenCV uses BGR, not RGB.
JERSEY_COLORS = {
    "White": (235, 235, 235),
    "Green": (50, 210, 70),
    "Red": (50, 50, 235),
    "Yellow": (30, 220, 255),
    "Blue": (235, 130, 40),
    "Purple": (200, 70, 190),
    "Unknown": (60, 220, 60),
}

ROLE_COLORS = {
    "goalkeeper": (255, 180, 40),
    "referee": (40, 200, 255),
    "ball": (0, 140, 255),
}


def annotate_frame(frame, detections, ball_state=None):
    """Draw tracked objects with jersey colors and team assignments."""
    annotated = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        class_name = detection.get("class_name", "object").lower()
        track_id = detection.get("track_id")
        jersey_color = detection.get("jersey_color", "Unknown")
        team = detection.get("team", "Unknown")

        if class_name == "player":
            color = JERSEY_COLORS.get(
                jersey_color,
                JERSEY_COLORS["Unknown"],
            )

            if team != "Unknown":
                label = (
                    f"Player #{track_id} | {team} ({jersey_color})"
                )
            else:
                label = f"Player #{track_id} | {jersey_color}"

        elif class_name in ROLE_COLORS:
            color = ROLE_COLORS[class_name]
            label = f"{class_name.title()} #{track_id}"

        else:
            color = (200, 200, 200)
            label = f"{class_name.title()} #{track_id}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        text_y = max(22, y1 - 8)

        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # Preserve the existing ball visualization.
    if ball_state:
        status = ball_state.get("status", "unavailable")
        ball_x = ball_state.get("ball_x")
        ball_y = ball_state.get("ball_y")

        if ball_x is not None and ball_y is not None:
            center = (int(ball_x), int(ball_y))
            cv2.circle(annotated, center, 7, (0, 0, 255), -1)
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
