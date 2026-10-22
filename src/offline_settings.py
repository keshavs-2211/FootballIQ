"""FootballIQ offline second-pass settings. Change CLIP to test another match."""
from pathlib import Path

CLIP = "bluevsred"  # e.g. "bluevsred", "whitevsblue"
VIDEO = f"sample_data/input_videos/{CLIP}.mp4"
ANNOTATED_VIDEO = f"outputs/videos/{CLIP}_improved.mp4"
FIRST_PASS_CSV = f"outputs/csv/{CLIP}_improved_coordinates.csv"
CALIBRATION_JSON = f"references/{CLIP}_pitch_calibration.json"
REFERENCE_DIR = f"references/{CLIP}"

OUT_CSV = f"outputs/csv/{CLIP}_professional_coordinates.csv"
PROPOSALS_CSV = f"outputs/csv/{CLIP}_identity_review.csv"
APPROVALS_CSV = f"outputs/csv/{CLIP}_identity_approvals.csv"
BALL_CSV = f"outputs/csv/{CLIP}_professional_ball.csv"
BALL_AUDIT_CSV = f"outputs/csv/{CLIP}_ball_candidate_audit.csv"
OUT_VIDEO = f"outputs/videos/{CLIP}_professional_birdeye.mp4"
OUT_SIDE = f"outputs/videos/{CLIP}_professional_side_by_side.mp4"
REVIEW_DIR = f"outputs/diagnostics/{CLIP}_id_review"

# Player data controls. Estimates are marked, not treated as measurements.
MAX_BRIDGE_MISSING_FRAMES = 12
MAX_PLAUSIBLE_PLAYER_MPS = 12.0
POSITION_MARGIN_METRES = 4.0
RTS_PROCESS_ACCEL_NOISE = 5.0  # Larger follows turns more closely
MEASUREMENT_NOISE_METRES = 0.50
MAX_SMOOTHING_SEGMENT_GAP = 15
MIN_TRACK_LENGTH_FOR_REID = 4

# Tracklet matching. Automated merges are intentionally OFF by default.
MAX_REID_GAP_FRAMES = 20
MAX_REID_DISTANCE_METRES = 5.0
MAX_REID_SPEED_MPS = 10.0
MIN_REID_APPEARANCE = 0.45
MIN_REID_SCORE = 0.60
MIN_REID_MARGIN = 0.07
AUTO_APPROVE_ID_MERGES = False  # Leave False until visually checked

# Ball detection runs as a SEPARATE pass and only periodically.
BALL_YOLO_EVERY = 2
BALL_YOLO_CONF = 0.13
BALL_YOLO_IMAGE_SIZE = 768  # Optional, depends on detector support
BALL_MAX_PREDICTION_GAP = 11  # frames
BALL_CONFIRM_HITS = 2
BALL_CONFIRM_WINDOW = 7
BALL_MAX_SPEED_MPS = 42.0
BALL_PENALTY_EXCLUSION_METRES = 2.0  # Omits real penalties as well
BALL_LK_MAX_PIXEL_STEP = 55.0
BALL_TEMPLATE_MATCH = True
BALL_TEMPLATE_THRESHOLD = 0.78

DRAW_ESTIMATE_OUTLINES = True
SHOW_CONFIDENCE_STATUS = True
