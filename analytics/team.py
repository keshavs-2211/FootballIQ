from collections import Counter

from config import TEAM_A_NAME, TEAM_B_NAME


class TeamCalibrator:
    """
    Determines the two most common reliable outfield jersey colors.

    Once calibrated, the team-color mapping remains fixed unless a new
    TeamCalibrator instance is created.
    """

    def __init__(self, minimum_observations=20):
        self.minimum_observations = minimum_observations
        self.color_counts = Counter()
        self.team_a_color = None
        self.team_b_color = None
        self.calibrated = False

    def observe(self, jersey_color, confidence):
        """Record a reliable jersey-color observation."""
        if confidence < 0.55 or jersey_color == "Unknown":
            return

        # Do not let later observations change an established mapping.
        if self.calibrated:
            return

        self.color_counts[jersey_color] += 1

    def calibrate(self):
        """Calibrate once sufficient evidence from at least two colors exists."""
        if self.calibrated:
            return self.get_mapping()

        if sum(self.color_counts.values()) < self.minimum_observations:
            return None

        if len(self.color_counts) < 2:
            return None

        most_common = self.color_counts.most_common(2)
        self.team_a_color = most_common[0][0]
        self.team_b_color = most_common[1][0]
        self.calibrated = True

        return self.get_mapping()

    def get_team(self, jersey_color, confidence):
        """Map a reliable jersey color to a team."""
        if not self.calibrated or confidence < 0.55:
            return "Unknown"

        if jersey_color == self.team_a_color:
            return TEAM_A_NAME

        if jersey_color == self.team_b_color:
            return TEAM_B_NAME

        return "Unknown"

    def get_mapping(self):
        """Return the established mapping, or None before calibration."""
        if not self.calibrated:
            return None

        return {
            TEAM_A_NAME: self.team_a_color,
            TEAM_B_NAME: self.team_b_color,
        }