"""FootballIQ offline possession: edit CLIP in offline_settings.py to switch matches."""
import offline_settings as S

CLIP = S.CLIP
PLAYER_CSV = S.OUT_CSV
BALL_CSV = S.BALL_CSV
VIDEO = S.VIDEO
SIDE_VIDEO = S.ANNOTATED_VIDEO

TIMELINE_CSV = f"outputs/csv/{CLIP}_possession_timeline.csv"
SUMMARY_CSV = f"outputs/csv/{CLIP}_possession_summary.csv"
SUMMARY_JSON = f"outputs/csv/{CLIP}_possession_summary.json"
OUTPUT_VIDEO = f"outputs/videos/{CLIP}_possession_birdeye.mp4"
OUTPUT_SIDE = f"outputs/videos/{CLIP}_possession_side_by_side.mp4"

# Distances are in real-world metres (requires valid pitch calibration).
CONTROL_RADIUS_M = 2.4
CONTESTED_DISTANCE_MARGIN_M = 0.65
FREE_BALL_RADIUS_M = 3.3

# Require repeated evidence before changing a possession label.
CONFIRM_FRAMES = 5
# The ball is observed only every few frames. We hold possession cautiously.
MAX_OCCLUSION_SECONDS = 0.70
# Predicted-only ball positions NEVER initiate new possession.
# Only actual YOLO positions or image-aided flow/template are evidence.
ACCEPT_BALL_SOURCES = ("measured_yolo", "measured_flow", "measured_template",
                       "observed_filtered")
# Ball moving this quickly is more likely in flight/pass than at a player's feet.
MAX_CONTROL_BALL_SPEED_MPS = 11.0
# If player feet are estimated or interpolated, require stronger proximity.
ESTIMATED_PLAYER_RADIUS_FACTOR = 0.8

# Avoid claiming coverage for a match with sparse reliable ball observations.
MIN_REQUIRED_MEASURED_BALL_FRAMES = 10

# Experimental display only; possession stats are not official event data.
DRAW_POSSESSION_HUD = True
