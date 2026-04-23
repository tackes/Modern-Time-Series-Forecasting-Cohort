"""
src/evaluation.py — Pooled Evaluation Metrics
===============================================
Computes pooled point and interval metrics across ALL forecasted observations
from ALL backtest windows. Do not average per-series first.

Metric Definitions
------------------
wMAPE (Weighted Mean Absolute Percentage Error):
    Pooled across all observations. Numerator = sum of absolute errors.
    Denominator = sum of actual values. Scale-aware and inventory-relevant.

Interval Score (Winkler Score):
    Penalizes intervals that are wide but still miss the actual.
    Lower is better. Computed at 80% coverage level.
    Formula per observation:
        IS = (hi - lo) + (2/alpha) * max(lo - y, 0) + (2/alpha) * max(y - hi, 0)
    where alpha = 1 - coverage = 0.20 for 80% intervals.

Coverage (Diagnostic Only):
    Fraction of actuals falling within [lo_80, hi_80].
    Not used for model ranking. Used to flag degenerate intervals.

Usage
-----
    from src.evaluation import score_forecasts, build_leaderboard

    scores = score_forecasts(df_forecasts, subset_name="workshop_1000")
    leaderboard = build_leaderboard([scores_module4, scores_module5, scores_module6])
"""

import numpy as np
import pandas as pd
from typing import Optional

from config import INTERVAL_COVERAGE


# ---------------------------------------------------------------------------
# Core Metric Functions
# ---------------------------------------------------------------------------

def pooled_wmape(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Pooled Weighted MAE / sum(actuals).
    Handles zero actuals by excluding them from the denominator.
    If all actuals are zero, returns NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")

    return float(np.sum(np.abs(y_true - y_hat)) / denom)


def pooled_bias(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Pooled forecast bias: sum(y_hat - y) / sum(y).
    Positive = systematic over-forecast. Negative = systematic under-forecast.
    Returns NaN if all actuals are zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_hat  = np.asarray(y_hat,  dtype=float)

    denom = np.sum(y_true)
    if denom == 0:
        return float("nan")

    return float(np.sum(y_hat - y_true) / denom)


def pooled_interval_score(
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    coverage: float = INTERVAL_COVERAGE,
) -> float:
    """
    Mean Winkler (Interval) Score pooled across all observations.
    Lower is better.

    Parameters
    ----------
    y_true   : Actual values.
    lo       : Lower bound of prediction interval.
    hi       : Upper bound of prediction interval.
    coverage : Nominal coverage level (default: 0.80).
    """
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    alpha = 1.0 - coverage
    width = hi - lo
    penalty_lo = (2.0 / alpha) * np.maximum(lo - y_true, 0.0)
    penalty_hi = (2.0 / alpha) * np.maximum(y_true - hi, 0.0)
    scores = width + penalty_lo + penalty_hi

    return float(np.mean(scores))


def coverage_rate(
    y_true: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> float:
    """
    Fraction of actuals within [lo, hi]. Diagnostic only — not used for ranking.
    """
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    return float(np.mean((y_true >= lo) & (y_true <= hi)))


# ---------------------------------------------------------------------------
# Score a Forecast DataFrame
# ---------------------------------------------------------------------------

def score_forecasts(
    df: pd.DataFrame,
    subset_name: str,
    stage: Optional[str] = None,
) -> pd.DataFrame:
    """
    Score all models in a forecast DataFrame.

    Expects df to conform to the forecast schema (see src/schemas.py).
    Scores are computed pooled across all windows and all series.

    Parameters
    ----------
    df          : Forecast DataFrame with columns [unique_id, ds, y, model,
                  y_hat, cutoff, stage]. Optional: [lo_80, hi_80].
    subset_name : Label for the subset (e.g., 'workshop_1000', 'micro_50').
    stage       : If provided, filters df to this stage before scoring.
                  If None, uses whatever 'stage' values are present (all pooled).

    Returns
    -------
    pd.DataFrame with columns matching SCORE_SCHEMA.
    """
    if stage is not None:
        df = df[df["stage"] == stage].copy()

    required = {"unique_id", "ds", "y", "model", "y_hat", "cutoff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"score_forecasts: missing columns {missing}")

    has_intervals = ("lo_80" in df.columns) and ("hi_80" in df.columns)
    records = []

    for model_name, group in df.groupby("model"):
        y_true = group["y"].values
        y_hat = group["y_hat"].values
        inferred_stage = group["stage"].iloc[0] if "stage" in group.columns else "unknown"

        # Point metrics
        records.append({
            "model":             model_name,
            "metric":            "wMAPE",
            "score":             pooled_wmape(y_true, y_hat),
            "stage":             inferred_stage,
            "aggregation_scope": "pooled_all_windows",
            "subset_name":       subset_name,
        })
        records.append({
            "model":             model_name,
            "metric":            "Bias",
            "score":             pooled_bias(y_true, y_hat),
            "stage":             inferred_stage,
            "aggregation_scope": "pooled_all_windows",
            "subset_name":       subset_name,
        })

        # Interval metrics (only if interval columns present)
        if has_intervals:
            lo = group["lo_80"].values
            hi = group["hi_80"].values

            records.append({
                "model":             model_name,
                "metric":            "IntervalScore_80",
                "score":             pooled_interval_score(y_true, lo, hi),
                "stage":             inferred_stage,
                "aggregation_scope": "pooled_all_windows",
                "subset_name":       subset_name,
            })

            records.append({
                "model":             model_name,
                "metric":            "Coverage_80",
                "score":             coverage_rate(y_true, lo, hi),
                "stage":             inferred_stage,
                "aggregation_scope": "pooled_all_windows",
                "subset_name":       subset_name,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Build a Ranked Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(score_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate score DataFrames from multiple modules into a single
    ranked leaderboard, sorted by wMAPE ascending.

    Parameters
    ----------
    score_dfs : List of DataFrames returned by score_forecasts().

    Returns
    -------
    pd.DataFrame — full leaderboard, sorted by wMAPE ascending.
    """
    combined = pd.concat(score_dfs, ignore_index=True)

    # Pivot so each metric is a column, sorted by wMAPE
    pivot = (
        combined
        .pivot_table(index=["model", "stage", "subset_name"],
                     columns="metric",
                     values="score")
        .reset_index()
    )

    pivot.columns.name = None

    if "wMAPE" in pivot.columns:
        pivot = pivot.sort_values("wMAPE", ascending=True).reset_index(drop=True)

    return pivot
