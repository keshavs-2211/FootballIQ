from collections import defaultdict, deque

from config import TEAM_HISTORY_LENGTH, TEAM_A_NAME, TEAM_B_NAME


class TeamPersistence:
    """
    Stabilizes team assignments per track.

    The first valid team is retained unless the opposite team receives
    repeated, sufficiently confident observations. Unknown observations
    do not overwrite an established assignment.
    """

    def __init__(self, history_length=TEAM_HISTORY_LENGTH,
                 switch_observations=5, switch_confidence=0.80):
        self.history_length = history_length
        self.switch_observations = switch_observations
        self.switch_confidence = switch_confidence

        self.history = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )
        self.current_team = {}
        self.opposite_streak = defaultdict(int)

    def update(self, track_id, team, confidence):
        """Update a track's team while resisting isolated misclassifications."""
        if track_id is None or track_id < 0:
            return "Unknown"

        if team not in {TEAM_A_NAME, TEAM_B_NAME}:
            return self.get_team(track_id)

        if not 0.0 < confidence <= 1.0:
            return self.get_team(track_id)

        current = self.current_team.get(track_id)

        if current is None:
            self.current_team[track_id] = team
            self.history[track_id].append((team, confidence))
            self.opposite_streak[track_id] = 0
            return team

        if team == current:
            self.opposite_streak[track_id] = 0
            self.history[track_id].append((team, confidence))
            return current

        # Only sustained, high-confidence evidence can change a team.
        if confidence >= self.switch_confidence:
            self.opposite_streak[track_id] += 1
        else:
            self.opposite_streak[track_id] = 0

        if self.opposite_streak[track_id] >= self.switch_observations:
            self.current_team[track_id] = team
            self.opposite_streak[track_id] = 0
            self.history[track_id].append((team, confidence))

        return self.current_team[track_id]

    def get_team(self, track_id):
        if track_id is None or track_id < 0:
            return "Unknown"
        return self.current_team.get(track_id, "Unknown")

    def get_history(self, track_id):
        return list(self.history.get(track_id, []))

    def clear_track(self, track_id):
        self.history.pop(track_id, None)
        self.current_team.pop(track_id, None)
        self.opposite_streak.pop(track_id, None)

    def clear_all(self):
        self.history.clear()
        self.current_team.clear()
        self.opposite_streak.clear()