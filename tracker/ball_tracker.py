from math import hypot


class BallTracker:
    """
    Tracks the football using YOLO ball detections.

    This module does not perform:
    - YOLO detection
    - player detection
    - player tracking
    - team classification
    - analytics

    It maintains the current ball position and state.

    Ball states:
        detected
        player_possession
        unavailable
    """

    DETECTED = "detected"
    PLAYER_POSSESSION = "player_possession"
    UNAVAILABLE = "unavailable"

    def __init__(self):
        self.last_position = None
        self.last_frame = None
        self.frames_since_detection = 0

    def update(self, ball_detection, frame_index):
        """
        Update the ball state using the current YOLO detection.

        If the ball is detected:
            frames_since_detection = 0

        If the ball is not detected:
            frames_since_detection increments.

        No free-running prediction is performed when
        the ball is unavailable.
        """

        # Ball not detected
        if ball_detection is None:
            self.last_position = None
            self.last_frame = frame_index
            self.frames_since_detection += 1

            return {
                "ball_x": None,
                "ball_y": None,
                "velocity_x": None,
                "velocity_y": None,
                "status": self.UNAVAILABLE,
                "confidence": None,
                "frames_since_detection": self.frames_since_detection,
            }

        # Ball detected
        bbox = ball_detection["bbox"]
        confidence = float(ball_detection["confidence"])

        x1, y1, x2, y2 = bbox

        ball_x = (x1 + x2) / 2.0
        ball_y = (y1 + y2) / 2.0

        velocity_x = None
        velocity_y = None

        # Calculate movement only when the previous
        # frame also had a valid ball position.
        if self.last_position is not None:
            previous_x, previous_y = self.last_position

            velocity_x = ball_x - previous_x
            velocity_y = ball_y - previous_y

        self.last_position = (ball_x, ball_y)
        self.last_frame = frame_index

        # Successful detection resets the counter.
        self.frames_since_detection = 0

        return {
            "ball_x": ball_x,
            "ball_y": ball_y,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "status": self.DETECTED,
            "confidence": confidence,
            "frames_since_detection": 0,
        }

    def update_possession(self, ball_state, possession_player_id=None):
        """
        Change the ball state to player_possession when
        a player has been determined to possess the ball.

        The actual possession decision will be implemented
        separately using spatial and temporal evidence.
        """

        if ball_state is None:
            return None

        if (
            ball_state["status"] == self.DETECTED
            and possession_player_id is not None
        ):
            ball_state["status"] = self.PLAYER_POSSESSION

        return ball_state

    @staticmethod
    def get_ball_center(ball_detection):
        """
        Return the center point of a ball detection.
        """

        if ball_detection is None:
            return None

        x1, y1, x2, y2 = ball_detection["bbox"]

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    @staticmethod
    def distance(point_a, point_b):
        """
        Calculate Euclidean distance between two points.
        """

        if point_a is None or point_b is None:
            return None

        return hypot(
            point_a[0] - point_b[0],
            point_a[1] - point_b[1],
        )