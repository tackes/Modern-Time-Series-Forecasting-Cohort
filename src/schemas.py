"""
src/schemas.py — Artifact Schema Definitions & Validators
===========================================================
Every parquet artifact written to artifacts/ must pass validation here
before the notebook proceeds. Column presence alone is not sufficient —
dtypes are enforced on load.

Called by: src/checkpointing.py (on every load_checkpoint() call)
Called by: notebooks (optionally, after saving a new artifact)
"""

import pandas as pd
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

# Required columns and their enforced pandas dtypes.
# "object" = string in pandas. Datetime columns are handled separately below.

FORECAST_SCHEMA = {
    "unique_id": "object",
    "ds":        "datetime64[ns]",
    "y":         "float64",
    "model":     "object",
    "y_hat":     "float64",
    "cutoff":    "datetime64[ns]",
    "stage":     "object",
}

FORECAST_OPTIONAL_COLUMNS = {
    "lo_80":  "float64",
    "hi_80":  "float64",
}

SCORE_SCHEMA = {
    "model":              "object",
    "metric":             "object",
    "score":              "float64",
    "stage":              "object",
    "aggregation_scope":  "object",   # Must contain only 'pooled_all_windows'
    "subset_name":        "object",
}

PANEL_SCHEMA = {
    "unique_id": "object",
    "ds":        "datetime64[ns]",
    "y":         "float64",
}

SCHEMA_MAP = {
    "forecast": FORECAST_SCHEMA,
    "score":    SCORE_SCHEMA,
    "panel":    PANEL_SCHEMA,
}

SchemaKey = Literal["forecast", "score", "panel"]


# ---------------------------------------------------------------------------
# Core Validator
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, schema_key: SchemaKey, artifact_name: str = "") -> pd.DataFrame:
    """
    Validate and coerce a DataFrame against the named schema.

    Raises ValueError with a clear diagnostic message on any failure.
    Returns the coerced DataFrame on success.

    Parameters
    ----------
    df           : The DataFrame to validate.
    schema_key   : One of 'forecast', 'score', 'panel'.
    artifact_name: Optional name included in error messages for traceability.

    Usage
    -----
    df = validate(df, "forecast", artifact_name="04_baseline_forecasts")
    """
    label = f"[{artifact_name}]" if artifact_name else "[unknown artifact]"
    schema = SCHEMA_MAP.get(schema_key)

    if schema is None:
        raise ValueError(
            f"{label} Unknown schema key: '{schema_key}'. "
            f"Valid keys: {list(SCHEMA_MAP.keys())}"
        )

    # --- 1. Required column presence check ---
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} Schema violation — missing required columns: {missing}\n"
            f"  Found columns: {list(df.columns)}"
        )

    # --- 2. Dtype enforcement ---
    for col, expected_dtype in schema.items():
        if expected_dtype == "datetime64[ns]":
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                raise ValueError(
                    f"{label} Cannot coerce column '{col}' to datetime. Error: {e}"
                )
        elif expected_dtype == "float64":
            try:
                df[col] = df[col].astype("float64")
            except Exception as e:
                raise ValueError(
                    f"{label} Cannot coerce column '{col}' to float64. Error: {e}"
                )
        elif expected_dtype == "object":
            df[col] = df[col].astype("object")

    # --- 3. Forecast-specific optional column enforcement ---
    if schema_key == "forecast":
        optional = {k: v for k, v in FORECAST_OPTIONAL_COLUMNS.items() if k in df.columns}
        for col, expected_dtype in optional.items():
            try:
                df[col] = df[col].astype(expected_dtype)
            except Exception as e:
                raise ValueError(
                    f"{label} Cannot coerce optional column '{col}' to {expected_dtype}. Error: {e}"
                )

    # --- 4. Score-specific business rule: aggregation_scope must be 'pooled_all_windows' ---
    if schema_key == "score":
        invalid_scope = df[df["aggregation_scope"] != "pooled_all_windows"]
        if not invalid_scope.empty:
            bad_values = invalid_scope["aggregation_scope"].unique().tolist()
            raise ValueError(
                f"{label} Score schema violation — 'aggregation_scope' must be "
                f"'pooled_all_windows' for all rows. Found: {bad_values}"
            )

    # --- 5. Null checks on required columns ---
    for col in schema:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(
                f"{label} Null values detected in required column '{col}': "
                f"{null_count} nulls out of {len(df)} rows."
            )

    return df


# ---------------------------------------------------------------------------
# Convenience Wrappers
# ---------------------------------------------------------------------------

def validate_forecast(df: pd.DataFrame, artifact_name: str = "") -> pd.DataFrame:
    return validate(df, "forecast", artifact_name)

def validate_score(df: pd.DataFrame, artifact_name: str = "") -> pd.DataFrame:
    return validate(df, "score", artifact_name)

def validate_panel(df: pd.DataFrame, artifact_name: str = "") -> pd.DataFrame:
    return validate(df, "panel", artifact_name)
