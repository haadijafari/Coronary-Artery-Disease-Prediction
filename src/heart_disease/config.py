"""Project-wide paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
"""Seed shared by every split, model and resampling step."""

TEST_SIZE = 0.2
"""Fraction of patients held out, untouched, for the final evaluation."""

TARGET_RECALL = 0.90
"""Minimum share of disease cases the decision threshold must catch (sensitivity)."""
