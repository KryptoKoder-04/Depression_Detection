import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Media Storage
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Ensure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Model configuration
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "weights", "best_model.pth")

# Video processing configuration
SEQUENCE_LENGTH = 360 # Frame length expected by our model
FPS = 5               # Frame extraction rate (matches downsampling by 5 at 25 fps)

# API Configurations
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
