
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best.pt"

SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
INPUT_VIDEO_DIR = SAMPLE_DATA_DIR / "input_videos"

OUTPUT_DIR = BASE_DIR / "outputs"
VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
JSON_OUTPUT_DIR = OUTPUT_DIR / "json"
HEATMAP_OUTPUT_DIR = OUTPUT_DIR / "heatmaps"
TACTICAL_OUTPUT_DIR = OUTPUT_DIR / "tactical_maps"
UPLOAD_DIR = BASE_DIR / "uploads"

# ==========================================
# YOLO SETTINGS
# ==========================================

# Starting values; evaluate on actual footage.
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.50
DEFAULT_IMAGE_SIZE = 640
DEFAULT_MAX_DETECTIONS = 100

# CPU processing settings
OPENCV_THREADS = 2

# ==========================================
# CLASSES
# ==========================================

PLAYER_CLASSES = {
    "player",
    "goalkeeper",
}

REFEREE_CLASS = "referee"
BALL_CLASS = "ball"

# ==========================================
# TEAMS AND JERSEYS
# ==========================================

TEAM_A_NAME = "Team A"
TEAM_B_NAME = "Team B"
REFEREE_NAME = "Referee"

TEAM_COLOR_MIN_CONFIDENCE = 0.55
TEAM_HISTORY_LENGTH = 10

# ==========================================
# TRACKING AND MEMORY
# ==========================================

TRACK_ACTIVATION_THRESHOLD = 0.25
TRACK_LOST_BUFFER = 30
TRACK_MATCHING_THRESHOLD = 0.30

JERSEY_CLASSIFY_INTERVAL = 5
PARQUET_BATCH_SIZE = 1000

PITCH_DETECT_INTERVAL = 10
PITCH_DETECT_WIDTH = 640

GOALKEEPER_MIN_OBSERVATIONS = 3
GOALKEEPER_TIMEOUT = 45

# ==========================================
# VIDEO
# ==========================================

DEFAULT_CODEC = "mp4v"

# ==========================================
# DIRECTORIES
# ==========================================

DIRECTORIES = [
    INPUT_VIDEO_DIR,
    VIDEO_OUTPUT_DIR,
    CSV_OUTPUT_DIR,
    JSON_OUTPUT_DIR,
    HEATMAP_OUTPUT_DIR,
    TACTICAL_OUTPUT_DIR,
    UPLOAD_DIR,
]

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )
