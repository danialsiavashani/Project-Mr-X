from pathlib import Path

# Project root — two levels up from this file (src/utils/paths.py)
ROOT = Path(__file__).resolve().parents[2]

# Data
DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR   = DATA_DIR / "samples"

# Dataset — birds20
BIRDS20_DIR = RAW_DIR / "birds20"
TRAIN_DIR   = BIRDS20_DIR / "train"
VAL_DIR     = BIRDS20_DIR / "valid"
TEST_DIR    = BIRDS20_DIR / "test"
IMAGES_TO_PREDICT_DIR = BIRDS20_DIR / "images to predict"

# Outputs
OUTPUTS_DIR     = ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR        = OUTPUTS_DIR / "logs"
PLOTS_DIR       = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"