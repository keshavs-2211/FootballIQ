import cv2
from config import TEAM_A_NAME, TEAM_B_NAME

TEAM_COLOURS = {
    TEAM_A_NAME: (245, 210, 80),
    TEAM_B_NAME: (60, 60, 235),
    "Unknown": (160, 160, 160),
}


def annotate_frame(frame, detections, ball_state=None):
    """Show known and Unknown tracked players, with IDs and team labels."""
    annotated = frame.copy()
    for obj in detections:
        role = str(obj.get("class_name", "")).lower()
        if role not in ("player", "goalkeeper"):
            continue
        team = str(obj.get("team", "Unknown"))
        color = TEAM_COLOURS.get(team, TEAM_COLOURS["Unknown"])
        x1, y1, x2, y2 = map(int, obj["bbox"])
        label = ("GK" if role == "goalkeeper" else "Player")
        label += f" #{obj.get('track_id', '?')} | {team}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color,
                      1 if team == "Unknown" else 2)
        cv2.putText(annotated, label, (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
    if ball_state:
        status = ball_state.get("status", "unavailable")
        x, y = ball_state.get("ball_x"), ball_state.get("ball_y")
        if x is not None and y is not None:
            cv2.circle(annotated, (int(x), int(y)), 7, (0, 0, 255), -1)
            cv2.circle(annotated, (int(x), int(y)), 10, (255, 255, 255), 2)
        cv2.putText(annotated, f"Ball: {status}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return annotated
