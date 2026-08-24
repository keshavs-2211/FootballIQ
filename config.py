from pathlib import Path

# ============================================================
# Project Root
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# Model Configuration
# ============================================================

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best.pt"


# ============================================================
# Input / Output Directories
# ============================================================

SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
INPUT_VIDEO_DIR = SAMPLE_DATA_DIR / "input_videos"

OUTPUT_DIR = BASE_DIR / "outputs"

VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
CSV_OUTPUT_DIR = OUTPUT_DIR / "csv"
JSON_OUTPUT_DIR = OUTPUT_DIR / "json"
HEATMAP_OUTPUT_DIR = OUTPUT_DIR / "heatmaps"
TACTICAL_OUTPUT_DIR = OUTPUT_DIR / "tactical_maps"

UPLOAD_DIR = BASE_DIR / "uploads"


# ============================================================
# YOLO Configuration
# ============================================================

DEFAULT_CONFIDENCE = 0.30
DEFAULT_IOU = 0.50


# ============================================================
# Video Configuration
# ============================================================

DEFAULT_CODEC = "mp4v"


# ============================================================
# Create Required Directories
# ============================================================

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
    directory.mkdir(parents=True, exist_ok=True)