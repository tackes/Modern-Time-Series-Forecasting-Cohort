"""
src/build_data_subset.py — Offline M5 Subset Construction
===========================================================
PURPOSE: Run this script ONCE to generate data/m5_workshop_subset.parquet.
The resulting parquet is committed to the repo and distributed with it.
Students never need to run this script.

What it does:
1. Reads raw M5 files from a local directory.
2. Melts wide sales format into a long panel.
3. Merges calendar dates.
4. Applies the locked subset policy (density, history, intermittency).
5. Saves a deterministic, schema-validated parquet to data/m5_workshop_subset.parquet.

Run from the project root:
    python src/build_data_subset.py --raw-dir "C:/path/to/m5-forecasting-accuracy"

Raw files required in that directory:
    sales_train_evaluation.csv
    calendar.csv
    sell_prices.csv

Download from: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Expected output:
    data/m5_workshop_subset.parquet  (~25 MB, ~1.9M rows, 1,000 series)
"""

import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR,
    RANDOM_SEED,
    WORKSHOP_SUBSET_N,
    MIN_HISTORY_DAYS,
    MAX_ZERO_FRACTION,
    HORIZON,
    N_WINDOWS,
    STEP_SIZE,
    WORKSHOP_SUBSET_PATH,
)
from src.schemas import validate_panel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Load raw M5 files from local directory
# ---------------------------------------------------------------------------

def load_raw_m5(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three raw M5 CSV files from a local directory."""

    required = {
        "sales": "sales_train_evaluation.csv",
        "calendar": "calendar.csv",
        "prices": "sell_prices.csv",
    }

    resolved = {}
    for key, filename in required.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {filename} in {raw_dir}\n"
                f"Download the M5 raw files from:\n"
                f"  https://www.kaggle.com/competitions/m5-forecasting-accuracy/data\n"
                f"Files needed: sales_train_evaluation.csv, calendar.csv, sell_prices.csv"
            )
        resolved[key] = path

    log.info(f"Loading sales_train_evaluation.csv ({resolved['sales'].stat().st_size / 1e6:.0f} MB)...")
    sales_df = pd.read_csv(resolved["sales"])

    log.info("Loading calendar.csv...")
    calendar_df = pd.read_csv(resolved["calendar"])

    log.info("Loading sell_prices.csv...")
    prices_df = pd.read_csv(resolved["prices"])

    log.info(f"  sales shape    : {sales_df.shape}")
    log.info(f"  calendar shape : {calendar_df.shape}")
    log.info(f"  prices shape   : {prices_df.shape}")

    return sales_df, calendar_df, prices_df

# ---------------------------------------------------------------------------
# Step 2: Melt wide sales to long panel
# ---------------------------------------------------------------------------

def melt_sales_to_panel(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Melt M5 wide format (one column per day) to a long panel.

    Input columns: id, item_id, dept_id, cat_id, store_id, state_id, d_1 ... d_N
    Output columns: unique_id, d, y

    unique_id = f"{item_id}_{store_id}"  — the level specified in the workshop spec.
    """
    log.info("Melting wide sales to long panel ...")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]

    long = sales_df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="y",
    )

    long["unique_id"] = long["item_id"] + "_" + long["store_id"]
    long["y"] = long["y"].astype("float64")

    return long[["unique_id", "d", "y"]]


# ---------------------------------------------------------------------------
# Step 3: Merge calendar dates
# ---------------------------------------------------------------------------

def merge_calendar(panel: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map M5 day identifiers (d_1, d_2 ...) to actual calendar dates.
    Output adds column 'ds' (datetime) and drops 'd'.
    """
    log.info("Merging calendar dates ...")

    cal = calendar_df[["d", "date"]].copy()
    cal["ds"] = pd.to_datetime(cal["date"])

    panel = panel.merge(cal[["d", "ds"]], on="d", how="left")
    panel = panel.drop(columns=["d"])

    return panel[["unique_id", "ds", "y"]]


# ---------------------------------------------------------------------------
# Step 4: Apply the locked subset policy
# ---------------------------------------------------------------------------

def _first_cv_cutoff(end_date: pd.Timestamp) -> pd.Timestamp:
    """
    Compute the first cross-validation cutoff.
    Cutoffs step backwards from the end of the training data.

    With N_WINDOWS=3, STEP_SIZE=28:
        cutoff_1 = end - (N_WINDOWS * STEP_SIZE) = end - 84
        This is the earliest cutoff; the series must have MIN_HISTORY_DAYS before it.
    """
    earliest_cutoff = end_date - pd.Timedelta(days=(N_WINDOWS * STEP_SIZE))
    return earliest_cutoff


def apply_subset_policy(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the three-part subset policy:
    1. Strictly positive, dense series only (total sales > 0).
    2. Minimum 365 days of history before the first CV cutoff.
    3. Exclude series with >50% zeros over training history.

    Then select the top WORKSHOP_SUBSET_N series by total sales volume,
    using RANDOM_SEED=42 as the deterministic tie-breaker.
    """
    log.info("Applying subset policy ...")

    end_date = panel["ds"].max()
    first_cutoff = _first_cv_cutoff(end_date)
    min_start_date = first_cutoff - pd.Timedelta(days=MIN_HISTORY_DAYS)

    # Work only on training history (before first cutoff)
    train = panel[panel["ds"] < first_cutoff].copy()

    # --- Filter 1: Minimum history ---
    series_start = train.groupby("unique_id")["ds"].min()
    has_history = series_start[series_start <= min_start_date].index
    log.info(f"  Series with ≥{MIN_HISTORY_DAYS} days history: {len(has_history):,}")

    # --- Filter 2: Intermittency lock (>50% zeros) ---
    zero_frac = (
        train[train["unique_id"].isin(has_history)]
        .groupby("unique_id")["y"]
        .apply(lambda s: (s == 0).mean())
    )
    not_intermittent = zero_frac[zero_frac <= MAX_ZERO_FRACTION].index
    log.info(f"  Series passing intermittency filter (≤{MAX_ZERO_FRACTION*100:.0f}% zeros): "
             f"{len(not_intermittent):,}")

    # --- Filter 3: Strictly positive total sales ---
    total_sales = (
        train[train["unique_id"].isin(not_intermittent)]
        .groupby("unique_id")["y"]
        .sum()
    )
    positive_series = total_sales[total_sales > 0]
    log.info(f"  Series with strictly positive total sales: {len(positive_series):,}")

    # --- Rank by total sales, select top N deterministically ---
    rng = np.random.default_rng(RANDOM_SEED)
    ranked = positive_series.sort_values(ascending=False)

    if len(ranked) < WORKSHOP_SUBSET_N:
        log.warning(
            f"Only {len(ranked)} series passed filters. "
            f"Requested WORKSHOP_SUBSET_N={WORKSHOP_SUBSET_N}. Using all available."
        )
        selected = ranked.index
    else:
        selected = ranked.iloc[:WORKSHOP_SUBSET_N].index

    log.info(f"  Final workshop subset: {len(selected):,} series")

    subset = panel[panel["unique_id"].isin(selected)].copy()
    subset = subset.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    return subset


# ---------------------------------------------------------------------------
# Step 5: Save to parquet
# ---------------------------------------------------------------------------

def save_subset(panel: pd.DataFrame) -> None:
    """Validate schema and write the subset parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Validating panel schema before save ...")
    panel = validate_panel(panel, artifact_name="m5_workshop_subset")

    log.info(f"Saving to {WORKSHOP_SUBSET_PATH} ...")
    panel.to_parquet(WORKSHOP_SUBSET_PATH, index=False)

    size_mb = WORKSHOP_SUBSET_PATH.stat().st_size / (1024 ** 2)
    log.info(
        f"\n  ✓ Subset saved successfully.\n"
        f"    Path    : {WORKSHOP_SUBSET_PATH}\n"
        f"    Rows    : {len(panel):,}\n"
        f"    Series  : {panel['unique_id'].nunique():,}\n"
        f"    Size    : {size_mb:.1f} MB\n"
        f"    Date range : {panel['ds'].min().date()} → {panel['ds'].max().date()}\n"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the M5 workshop subset parquet from local raw files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Path to directory containing sales_train_evaluation.csv, calendar.csv, sell_prices.csv",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    log.info("=" * 60)
    log.info("BUILD DATA SUBSET — Local M5 Ingestion Pipeline")
    log.info(f"  Raw data dir: {raw_dir}")
    log.info("=" * 60)

    sales_df, calendar_df, prices_df = load_raw_m5(raw_dir)
    panel = melt_sales_to_panel(sales_df)
    panel = merge_calendar(panel, calendar_df)
    panel = apply_subset_policy(panel)
    save_subset(panel)

    log.info("Done. Commit data/m5_workshop_subset.parquet to the repo.")


if __name__ == "__main__":
    main()
