# src/forecast_schema.py

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


FORECAST_SCHEMA_COLS = [
    "unique_id",
    "ds",
    "y",
    "model",
    "y_hat",
    "lo_80",
    "hi_80",
    "cutoff",
    "stage",
]


def _ensure_unique_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unique_id available as a column."""
    out = df.copy()
    if "unique_id" not in out.columns:
        out = out.reset_index()
    return out


def _interval_columns(model_col: str, level: int = 80) -> tuple[str, str]:
    return f"{model_col}-lo-{level}", f"{model_col}-hi-{level}"


def _finalize_forecast_schema(
    df: pd.DataFrame,
    stage: str,
    clip_lower: float | None = 0,
) -> pd.DataFrame:
    out = df.copy()
    out["stage"] = stage

    if clip_lower is not None:
        for col in ["y_hat", "lo_80", "hi_80"]:
            if col in out.columns:
                out[col] = out[col].clip(lower=clip_lower)

    for col in ["lo_80", "hi_80"]:
        if col not in out.columns:
            out[col] = np.nan

    return out[FORECAST_SCHEMA_COLS]


def reshape_statsforecast_cv(
    cv_df: pd.DataFrame,
    stage: str = "baseline",
    level: int = 80,
    model_cols: Iterable[str] | None = None,
    clip_lower: float | None = 0,
) -> pd.DataFrame:
    """
    Convert StatsForecast wide cross-validation output into the workshop forecast schema.

    Expected input:
    unique_id, ds, cutoff, y, and one or more model columns.
    Optional interval columns may be named like AutoETS-lo-80 / AutoETS-hi-80.
    """
    df = _ensure_unique_id_column(cv_df)

    meta_cols = ["unique_id", "ds", "cutoff", "y"]

    if model_cols is None:
        model_cols = [
            c
            for c in df.columns
            if c not in meta_cols and not re.search(r"-(lo|hi)-\d+$", c)
        ]

    records = []

    for model_col in model_cols:
        lo_col, hi_col = _interval_columns(model_col, level)

        chunk = df[meta_cols + [model_col]].copy()
        chunk = chunk.rename(columns={model_col: "y_hat"})
        chunk["model"] = model_col

        chunk["lo_80"] = df[lo_col].values if lo_col in df.columns else np.nan
        chunk["hi_80"] = df[hi_col].values if hi_col in df.columns else np.nan

        records.append(chunk)

    out = pd.concat(records, ignore_index=True)
    return _finalize_forecast_schema(out, stage=stage, clip_lower=clip_lower)


def reshape_mlforecast_cv(
    cv_df: pd.DataFrame,
    model_col: str = "LGBMRegressor",
    stage: str = "ml",
    model_name: str = "LightGBM",
    level: int = 80,
    clip_lower: float | None = 0,
) -> pd.DataFrame:
    """
    Convert MLForecast wide cross-validation output into the workshop forecast schema.

    Interval columns, when present, are expected to be named:
    LGBMRegressor-lo-80 / LGBMRegressor-hi-80.
    """
    df = _ensure_unique_id_column(cv_df)

    lo_col, hi_col = _interval_columns(model_col, level)

    out = df[["unique_id", "ds", "cutoff", "y", model_col]].copy()
    out = out.rename(columns={model_col: "y_hat"})
    out["model"] = model_name

    out["lo_80"] = df[lo_col].values if lo_col in df.columns else np.nan
    out["hi_80"] = df[hi_col].values if hi_col in df.columns else np.nan

    return _finalize_forecast_schema(out, stage=stage, clip_lower=clip_lower)


def reshape_neuralforecast_cv(
    cv_df: pd.DataFrame,
    model_col: str = "NHITS",
    stage: str = "dl",
    model_name: str = "NHITS",
    level: int = 80,
    clip_lower: float | None = 0,
) -> pd.DataFrame:
    """
    Convert NeuralForecast wide cross-validation output into the workshop forecast schema.

    Interval columns, when present, are expected to be named:
    NHITS-lo-80 / NHITS-hi-80.

    Do not fabricate intervals from in-sample residuals. If interval columns are missing,
    lo_80 and hi_80 will be NaN.
    """
    df = _ensure_unique_id_column(cv_df)

    lo_col, hi_col = _interval_columns(model_col, level)

    out = df[["unique_id", "ds", "cutoff", "y", model_col]].copy()
    out = out.rename(columns={model_col: "y_hat"})
    out["model"] = model_name

    out["lo_80"] = df[lo_col].values if lo_col in df.columns else np.nan
    out["hi_80"] = df[hi_col].values if hi_col in df.columns else np.nan

    return _finalize_forecast_schema(out, stage=stage, clip_lower=clip_lower)
