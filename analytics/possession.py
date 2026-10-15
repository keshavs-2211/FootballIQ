
import math


class PossessionTracker:
    """Estimate ball possession using player distance and temporal evidence."""

    PROXIMITY_THRESHOLD = 90.0
    CONFIRMATION_THRESHOLD = 130.0
    SWITCH_CONFIRMATION_FRAMES = 3
    MAX_POSSESSION_FRAMES = 30
    MAX_CANDIDATE_MISSED_FRAMES = 2

    def __init__(self):
        self.current_player_id = None
        self.current_team = "Unknown"
        self.possession_frames = 0

        self.candidate_player_id = None
        self.candidate_team = "Unknown"
        self.candidate_frames = 0
        self.candidate_missed_frames = 0

    @staticmethod
    def _distance(point_a, point_b):
        """Calculate Euclidean distance between two points."""
        if point_a is None or point_b is None:
            return None

        return math.hypot(
            point_a[0] - point_b[0],
            point_a[1] - point_b[1],
        )

    @staticmethod
    def _get_foot_point(player):
        """Return the bottom-center point of a player's bounding box."""
        x1, y1, x2, y2 = player["bbox"]
        return (x1 + x2) / 2.0, y2

    def update(self, ball_state, players):
        """Update possession using the ball observation and tracked players."""
        result = {
            "possession_player_id": None,
            "possession_team": "Unknown",
            "has_ball": {},
        }



        # Handle a missing ball observation.
        if (
            ball_state is None
            or ball_state.get("ball_x") is None
            or ball_state.get("ball_y") is None
        ):
            self._handle_missing_observation()
            return self._build_result()

        # Clear stale possession if the current holder is no longer tracked.
        active_track_ids = {
            player.get("track_id")
            for player in players
            if player.get("track_id") is not None
            and player.get("track_id") >= 0
        }

        if (
            self.current_player_id is not None
            and self.current_player_id not in active_track_ids
        ):
            self._clear_possession()

        ball_point = (
            ball_state["ball_x"],
            ball_state["ball_y"],
        )

        # Find the closest tracked player to the ball.
        closest_player = None
        closest_distance = None

        for player in players:
            track_id = player.get("track_id")

            if track_id is None or track_id < 0:
                continue

            distance = self._distance(
                ball_point,
                self._get_foot_point(player),
            )

            if distance is None:
                continue

            if (
                closest_distance is None
                or distance < closest_distance
            ):
                closest_distance = distance
                closest_player = player

        # No tracked player is within the confirmation range.
        if (
            closest_player is None
            or closest_distance > self.CONFIRMATION_THRESHOLD
        ):
            self._handle_out_of_range_observation()
            return self._build_result()

        candidate_id = closest_player["track_id"]
        candidate_team = closest_player.get("team", "Unknown")

        # The current player remains closest to the ball.
        if candidate_id == self.current_player_id:
            # Keep the possession team synchronized with the
            # latest persisted team label.
            if candidate_team != "Unknown":
                self.current_team = candidate_team

            self.possession_frames += 1
            self._clear_candidate()
            self._expire_possession_if_needed()
            return self._build_result()

        # Continue tracking the same candidate, or start a new one.
        if candidate_id == self.candidate_player_id:
            self.candidate_frames += 1


    # Refresh the candidate's team as persistence updates it.
            if candidate_team != "Unknown":
                self.candidate_team = candidate_team

        else:
            self.candidate_player_id = candidate_id
            self.candidate_team = candidate_team
            self.candidate_frames = 1

        # A valid in-range observation resets the missed count.
        self.candidate_missed_frames = 0

        # Confirm possession after the required number of observations.
        if (
            self.candidate_frames
            >= self.SWITCH_CONFIRMATION_FRAMES
        ):
            self.current_player_id = self.candidate_player_id
            self.current_team = self.candidate_team
            self.possession_frames = 0
            self._clear_candidate()

        self._expire_possession_if_needed()
        return self._build_result()

    def _handle_missing_observation(self):
        """Preserve candidates briefly when the ball is not detected."""
        if self.candidate_player_id is not None:
            self.candidate_missed_frames += 1

            if (
                self.candidate_missed_frames
                > self.MAX_CANDIDATE_MISSED_FRAMES
            ):
                self._clear_candidate()

        if self.current_player_id is not None:
            self.possession_frames += 1
            self._expire_possession_if_needed()

    def _handle_out_of_range_observation(self):
        """Allow a pending candidate to survive brief out-of-range frames."""
        if self.candidate_player_id is not None:
            self.candidate_missed_frames += 1

            if (
                self.candidate_missed_frames
                > self.MAX_CANDIDATE_MISSED_FRAMES
            ):
                self._clear_candidate()

        if self.current_player_id is not None:
            self.possession_frames += 1
            self._expire_possession_if_needed()

    def _expire_possession_if_needed(self):
        """Clear possession when it exceeds the allowed duration."""
        if (
            self.current_player_id is not None
            and self.possession_frames > self.MAX_POSSESSION_FRAMES
        ):
            self._clear_possession()

    def _build_result(self):
        """Build a consistent possession result."""
        result = {
            "possession_player_id": None,
            "possession_team": "Unknown",
            "has_ball": {},
        }

        if self.current_player_id is not None:
            result["possession_player_id"] = self.current_player_id
            result["possession_team"] = self.current_team
            result["has_ball"][self.current_player_id] = True

        return result

    def _clear_candidate(self):
        """Reset the pending possession candidate."""
        self.candidate_player_id = None
        self.candidate_team = "Unknown"
        self.candidate_frames = 0
        self.candidate_missed_frames = 0

    def _clear_possession(self):
        """Reset confirmed possession and the pending candidate."""
        self.current_player_id = None
        self.current_team = "Unknown"
        self.possession_frames = 0
        self._clear_candidate()

    def reset(self):
        """Reset all possession tracking state."""
        self._clear_possession()
