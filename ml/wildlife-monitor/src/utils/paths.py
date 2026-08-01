from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR   = PROCESSED_DIR / "valid"

OUTPUTS_DIR     = ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR        = OUTPUTS_DIR / "logs"
PLOTS_DIR       = OUTPUTS_DIR / "plots"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"