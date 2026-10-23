"""Occlusion-aware team possession estimation from offline tracking CSVs.

No YOLO inference. Each frame is assigned Team A / Team B / Contested /
Free ball / Unknown. Possession is evidence-based, never an absolute fact.
Predicted ball positions cannot start or switch possession. Report unknown-time
coverage so sparse detection cannot create misleading percentages.
"""
from collections import Counter
from dataclasses import dataclass
import math


TEAM_A = "Team A"
TEAM_B = "Team B"
CONTESTED = "Contested"
FREE = "Free ball"
UNKNOWN = "Unknown"
STATES = (TEAM_A, TEAM_B, CONTESTED, FREE, UNKNOWN)


@dataclass
class Config:
    fps: float = 30.0
    control_radius: float = 1.8
    contested_margin: float = 0.5
    free_radius: float = 3.0
    confirm_frames: int = 3
    max_occlusion_s: float = 0.6
    max_ball_control_speed: float = 11.0
    estimated_radius_factor: float = 0.8
    ball_observation_sources: tuple = ("measured_yolo", "measured_flow", "measured_template", "observed_filtered")


def number(value, default=None):
    try:
        val = float(value)
        return val if math.isfinite(val) else default
    except (TypeError, ValueError):
        return default


def player_valid(player):
    x = number(player.get("pitch_x_m"))
    y = number(player.get("pitch_y_m"))
    return (x is not None and y is not None and 0 <= x <= 105 and 0 <= y <= 68)


class PossessionEstimator:
    def __init__(self, config):
        self.cfg = config
        self.fps = max(1.0, config.fps)
        self.current = UNKNOWN
        self.owner = None
        self.pending = None
        self.pending_count = 0
        self.last_good_ball = None  # (frame,x,y)
        self.previous_good_ball = None
        self.last_trusted_control = None  # (frame,team,id)
        self.initialized = False

    def _players_near(self, players, x, y):
        ranked = []
        for p in players:
            team = str(p.get("team", UNKNOWN)).strip()
            if team not in (TEAM_A, TEAM_B) or not player_valid(p):
                continue
            observed = str(p.get("is_observed", "1")).strip().lower() in ("1", "true", "yes")
            # An interpolated player can support, but not drive, possession.
            radius = self.cfg.control_radius if observed else self.cfg.control_radius * self.cfg.estimated_radius_factor
            px, py = float(p["pitch_x_m"]), float(p["pitch_y_m"])
            distance = math.hypot(px - x, py - y)
            ranked.append({"team": team, "id": str(p.get("track_id", "")),
                           "d": distance, "radius": radius, "observed": observed})
        ranked.sort(key=lambda p: p["d"])
        return ranked

    def _candidate_at(self, x, y, players, velocity=None):
        ranked = self._players_near(players, x, y)
        if not ranked:
            return UNKNOWN, None, None, "no_team_players"
        first = ranked[0]
        # A clear free ball is far from all labelled players.
        if first["d"] > self.cfg.free_radius:
            return FREE, None, first["d"], "ball_away_from_players"
        if first["d"] > first["radius"]:
            return UNKNOWN, None, first["d"], "near_but_no_clear_control"
        if velocity is not None and velocity > self.cfg.max_ball_control_speed:
            return FREE, None, first["d"], "ball_moving_too_fast_for_control"
        nearest_opponent = next((p for p in ranked if p["team"] != first["team"]), None)
        if nearest_opponent is not None and nearest_opponent["d"] <= nearest_opponent["radius"]:
            if abs(first["d"] - nearest_opponent["d"]) <= self.cfg.contested_margin:
                return CONTESTED, None, first["d"], "opposing_players_equally_close"
        if not first["observed"]:
            return UNKNOWN, None, first["d"], "nearest_player_position_estimated"
        return first["team"], first["id"], first["d"], "clear_nearby_player"

    def _ball_observed(self, ball):
        if ball is None:
            return False
        source = str(ball.get("source", "")).lower().strip()
        x, y = number(ball.get("pitch_x_m")), number(ball.get("pitch_y_m"))
        return (source in self.cfg.ball_observation_sources and
                x is not None and y is not None and 0 <= x <= 105 and 0 <= y <= 68)

    def _transition(self, candidate, owner, measured):
        # A confirmed team can be retained through a short occlusion. No new
        # team can be acquired via a predicted/missing ball frame alone.
        if candidate == self.current:
            if candidate in (TEAM_A, TEAM_B) and owner is not None:
                self.owner = owner
            elif candidate not in (TEAM_A, TEAM_B):
                self.owner = None
            # Do not erase a pending measured-team confirmation on an
            # intervening frame with no new reliable ball observation.
            if not (candidate == UNKNOWN and not measured and self.pending in (TEAM_A, TEAM_B)):
                self.pending = None
                self.pending_count = 0
            return

        if candidate in (UNKNOWN, FREE):
            self.current = candidate
            self.owner = None
            # An isolated missing frame shouldn't wipe a legitimate pending
            # team change, but a long gap or a free ball should.
            if candidate == FREE:
                self.pending = None
                self.pending_count = 0
            return

        if candidate in (TEAM_A, TEAM_B) and not measured:
            # Pending team switches are only advanced by measurements.
            return

        if self.current in (TEAM_A, TEAM_B) and self.current != candidate:
            # Do not continue crediting the previous team while new ball
            # measurements near an opponent are being confirmed.
            self.current, self.owner = UNKNOWN, None
        if candidate == self.pending:
            self.pending_count += 1
        else:
            self.pending, self.pending_count = candidate, 1
        if self.pending_count >= self.cfg.confirm_frames:
            self.current = candidate
            self.owner = owner
            self.pending = None
            self.pending_count = 0

    def update(self, frame, players, ball):
        observed = self._ball_observed(ball)
        source = str(ball.get("source", "missing")) if ball else "missing"
        ball_x = number(ball.get("pitch_x_m")) if ball else None
        ball_y = number(ball.get("pitch_y_m")) if ball else None
        candidate, owner, dist = UNKNOWN, None, None
        reason = "no_reliable_ball"
        age_frames = None
        velocity = None

        if observed:
            x, y = ball_x, ball_y
            if self.last_good_ball is not None:
                lf, lx, ly = self.last_good_ball
                elapsed = (frame - lf) / self.fps
                if elapsed > 0:
                    velocity = math.hypot(x - lx, y - ly) / elapsed
            self.previous_good_ball = self.last_good_ball
            self.last_good_ball = (frame, x, y)
            age_frames = 0
            candidate, owner, dist, reason = self._candidate_at(x, y, players, velocity)
        elif self.last_good_ball is not None:
            previous_frame, x, y = self.last_good_ball
            age_frames = max(0, frame - previous_frame)
            duration = age_frames / self.fps
            if duration <= self.cfg.max_occlusion_s:
                # Occlusion is plausible ONLY if a known owner is still close
                # to the last measured ball position. This does not assign
                # possession merely because an arbitrary player walks nearby.
                last = self.last_trusted_control
                if last is not None and last[0] >= previous_frame - round(0.2 * self.fps):
                    _, old_team, old_id = last
                    near = self._players_near(players, x, y)
                    original = next((p for p in near if p["id"] == old_id and p["team"] == old_team), None)
                    opposition = next((p for p in near if p["team"] != old_team and p["d"] <= p["radius"]), None)
                    if original is not None and original["d"] <= max(original["radius"], self.cfg.control_radius + 0.8):
                        if opposition and abs(original["d"] - opposition["d"]) <= self.cfg.contested_margin:
                            candidate, owner, reason = CONTESTED, None, "occluded_contested"
                        else:
                            candidate, owner, reason = old_team, old_id, "short_occlusion_owner_nearby"
                        dist = original["d"]
                    else:
                        reason = "occluded_owner_moved_away"
                else:
                    reason = "occluded_without_confirmed_owner"
            else:
                reason = "ball_missing_too_long"

        # No predicted-only acquisition, including during short occlusions.
        if not observed and candidate in (TEAM_A, TEAM_B) and candidate != self.current:
            candidate, owner, reason = UNKNOWN, None, "occlusion_cannot_initiate_team_control"
        self._transition(candidate, owner, measured=observed)
        if observed and candidate == self.current and candidate in (TEAM_A, TEAM_B):
            self.last_trusted_control = (frame, candidate, self.owner)

        age_s = None if age_frames is None else round(age_frames / self.fps, 3)
        return {
            "frame": frame,
            "timestamp_seconds": round(frame / self.fps, 4),
            "state": self.current,
            "possessing_player_id": self.owner or "",
            "evidence_state": candidate,
            "reason": reason,
            "ball_source": source,
            "ball_observed": int(observed),
            "last_ball_observation_age_s": age_s if age_s is not None else "",
            "ball_x_m": round(ball_x, 4) if ball_x is not None else "",
            "ball_y_m": round(ball_y, 4) if ball_y is not None else "",
            "nearest_player_distance_m": round(dist, 3) if dist is not None else "",
        }
