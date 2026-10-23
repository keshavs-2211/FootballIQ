"""CPU-friendly presets for FootballIQ. Edit ACTIVE_PRESET first."""

ACTIVE_PRESET = "balanced"  # "fast", "balanced", or "accuracy"

PRESETS = {
    "fast": {
        "image_size": 640,
        "confidence": 0.12,
        "iou": 0.55,
        "max_detections": 100,
        "enable_recovery": False,
        "recovery_every": 20,
        "jersey_every": 6,
        "jersey_min_similarity": 0.5,
        "jersey_min_margin": 0.035,
        "track_activation": 0.19,
        "lost_buffer": 30,
        "tracker_iou": 0.30,
    },
    "balanced": {
        "image_size": 640,
        "confidence": 0.09,
        "iou": 0.58,
        "max_detections": 120,
        "enable_recovery": True,
        "recovery_every": 15,
        "jersey_every": 4,
        "jersey_min_similarity": 0.48,
        "jersey_min_margin": 0.030,
        "track_activation": 0.16,
        "lost_buffer": 40,
        "tracker_iou": 0.30,
    },
    "accuracy": {
        "image_size": 768,
        "confidence": 0.08,
        "iou": 0.60,
        "max_detections": 140,
        "enable_recovery": True,
        "recovery_every": 8,
        "jersey_every": 3,
        "jersey_min_similarity": 0.46,
        "jersey_min_margin": 0.025,
        "track_activation": 0.15,
        "lost_buffer": 45,
        "tracker_iou": 0.25,
    },
}

SETTINGS = PRESETS[ACTIVE_PRESET].copy()

# Detection thresholds are applied AFTER YOLO's initial confidence cutoff.
CLASS_MIN_CONFIDENCE = {
    "player": 0.09,
    "goalkeeper": 0.09,
    "referee": 0.18,
    "ball": 0.1,
}

# Optional threshold override for repeatable A/B tests.
# Git Bash example: FOOTBALLIQ_YOLO_CONF=0.12 python run.py web
import os as _fi_os
_fi_override = _fi_os.environ.get("FOOTBALLIQ_YOLO_CONF")
if _fi_override is not None:
    _fi_confidence = float(_fi_override)
    if not 0.05 <= _fi_confidence <= 0.40:
        raise ValueError("FOOTBALLIQ_YOLO_CONF must be between 0.05 and 0.40")
    SETTINGS["confidence"] = _fi_confidence
    CLASS_MIN_CONFIDENCE["player"] = _fi_confidence
    CLASS_MIN_CONFIDENCE["goalkeeper"] = _fi_confidence


# Recovery uses two overlapping horizontal crops, periodically, not each frame.
RECOVERY_CROP_FRACTION = 0.62
RECOVERY_MIN_CONFIDENCE = 0.11
RECOVERY_DUPLICATE_IOU = 0.45

# Limit work in the team classifier and keep Unknown tracks visible.
REFERENCE_COUNT_PER_ROLE = 3
TEAM_HISTORY_LENGTH = 12
TEAM_MIN_VOTES = 2
TEAM_DECISION_RATIO = 1.30
EXPORT_UNKNOWN_PLAYERS = True  # Recommended for the bird's-eye CSV.

# The inaccurate hard-coded legacy field filter stays off.
SHOW_PITCH_DEBUG = False
USE_LEGACY_PITCH_FILTER = False
