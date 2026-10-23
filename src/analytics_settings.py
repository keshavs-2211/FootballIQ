"""Settings for FootballIQ offline analytics. Uses your existing offline_settings.CLIP."""
from pathlib import Path
import offline_settings as S

CLIP = S.CLIP
PLAYER_CSV = S.OUT_CSV  # outputs/csv/<clip>_professional_coordinates.csv
VIDEO = S.VIDEO
OUTPUT_ROOT = Path('outputs/analytics') / CLIP
CSV_ROOT = Path('outputs/csv')

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Only genuinely observed (and smoothed) positions are used for distance/speed.
MOVEMENT_OBSERVED_ONLY = True
MOVEMENT_INTERVAL_S = 0.40  # wider baseline reduces pixel/projection noise
MOVEMENT_MAX_GAP_S = 0.80   # do not bridge unseen play
MOVEMENT_MAX_SPEED_MPS = 10.5
MOVEMENT_JITTER_DEADZONE_M = 0.12

# Heatmaps may include short interpolation if explicitly enabled (default off).
HEATMAP_OBSERVED_ONLY = True
HEATMAP_BIN_SIZE_M = 2.0
HEATMAP_SIGMA_BINS = 1.65
MIN_PLAYER_HEATMAP_FRAMES = 8

# Team shape relies on simultaneous, genuinely observed outfield positions.
SHAPE_OBSERVED_ONLY = True
SHAPE_INCLUDE_GOALKEEPERS = False
SHAPE_MIN_PLAYERS = 4
MIN_POSITION_SAMPLES_FOR_LABEL = 6

# No assumed direction of attack; x is the calibration pitch axis.
TEAMS = ('Team A', 'Team B')
