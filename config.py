"""
config.py — Single Source of Truth
===================================
All notebooks and src/ modules must import from here.
Do not redefine these variables anywhere else in the project.

Tested Python Version: 3.10
Primary Execution Target: Google Colab (CPU)
Secondary Target: Local machine
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
PARAMS_DIR = BASE_DIR / "params"
COMPANION_ASSET_DIR = BASE_DIR / "companion_asset"
INSTRUCTOR_NOTEBOOK_DIR = BASE_DIR / "instructor_notebooks"
STUDENT_NOTEBOOK_DIR = BASE_DIR / "student_notebooks"
SRC_DIR = BASE_DIR / "src"

# ---------------------------------------------------------------------------
# Data File Paths
# ---------------------------------------------------------------------------
WORKSHOP_SUBSET_PATH = DATA_DIR / "m5_workshop_subset.parquet"

# ---------------------------------------------------------------------------
# Time Series Properties
# ---------------------------------------------------------------------------
HORIZON = 28          # Forecast horizon in days
SEASON_LENGTH = 7     # Weekly seasonality

# ---------------------------------------------------------------------------
# Cross-Validation Strategy
# ---------------------------------------------------------------------------
N_WINDOWS = 3         # Number of backtest windows
STEP_SIZE = 28        # Step between cutoffs (matches horizon)
REFIT = False         # Do not refit between windows — keeps runtime predictable

# ---------------------------------------------------------------------------
# Deterministic Subset Controls
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MICRO_SUBSET_N = 50         # Used for all live-execution cells
WORKSHOP_SUBSET_N = 1000    # Full workshop panel (used for precomputed artifacts)
FM_DEMO_SERIES_N = 3        # Number of series for live Chronos syntax demo

# ---------------------------------------------------------------------------
# Intermittency & History Filters (applied in build_data_subset.py)
# ---------------------------------------------------------------------------
MIN_HISTORY_DAYS = 365      # Minimum training history before first CV cutoff
MAX_ZERO_FRACTION = 0.50    # Exclude series with >50% zeros over training history

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
INTERVAL_COVERAGE = 0.80    # 80% prediction interval — used consistently across all models

# ---------------------------------------------------------------------------
# Execution Toggles
# ---------------------------------------------------------------------------
TUNE_MODELS = False         # If True, run hyperparameter search (take-home only)
USE_TUNED_PARAMS = True     # If True, load from params/ instead of using defaults
USE_INTERVALS = True        # If True, compute conformal prediction intervals (requires refit=True)

# ---------------------------------------------------------------------------
# M5 Source (Offline Use Only — never referenced in live notebooks)
# ---------------------------------------------------------------------------
M5_S3_BASE = "https://m5-forecasting.s3.amazonaws.com"
M5_S3_FILES = {
    "sales_train": f"{M5_S3_BASE}/sales_train_evaluation.csv",
    "calendar":    f"{M5_S3_BASE}/calendar.csv",
    "sell_prices": f"{M5_S3_BASE}/sell_prices.csv",
}

# ---------------------------------------------------------------------------
# Artifact Registry
# Used by checkpointing.py to resolve artifact paths by name.
# Keys are the explicit artifact names passed to load_checkpoint().
# ---------------------------------------------------------------------------
ARTIFACT_REGISTRY = {
    # Module 00
    "00_env_status":               ARTIFACT_DIR / "00_env_status.json",

    # Module 02
    "02_global_config":            ARTIFACT_DIR / "02_global_config.pkl",

    # Module 03
    "03_validated_panel":          ARTIFACT_DIR / "03_validated_panel.parquet",

    # Module 04
    "04_baseline_forecasts":       ARTIFACT_DIR / "04_baseline_forecasts.parquet",
    "04_baseline_cv_scores":       ARTIFACT_DIR / "04_baseline_cv_scores.parquet",

    # Module 05
    "05_ml_forecasts":             ARTIFACT_DIR / "05_ml_forecasts.parquet",
    "05_ml_rich_forecasts":        ARTIFACT_DIR / "05_ml_rich_forecasts.parquet",

    # Module 06
    "06_dl_forecasts":             ARTIFACT_DIR / "06_dl_forecasts.parquet",

    # Module 07
    "07_uncertainty_leaderboard":  ARTIFACT_DIR / "07_uncertainty_leaderboard.parquet",

    # Module 08
    "08_final_master_leaderboard": ARTIFACT_DIR / "08_final_master_leaderboard.csv",
}

# ---------------------------------------------------------------------------
# Artifact Type Routing
# Maps artifact names to their schema validator key in schemas.py.
# ---------------------------------------------------------------------------
ARTIFACT_SCHEMA_MAP = {
    "03_validated_panel":          "panel",
    "04_baseline_forecasts":       "forecast",
    "04_baseline_cv_scores":       "score",
    "05_ml_forecasts":             "forecast",
    "05_ml_rich_forecasts":        "forecast",
    "06_dl_forecasts":             "forecast",
    "07_uncertainty_leaderboard":  "score",
    "08_final_master_leaderboard": "score",
}
