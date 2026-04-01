"""
src/build_offline_artifacts.py — Offline Artifact Construction
================================================================
PURPOSE: Run this script ONCE, offline, before the workshop.
Generates all precomputed artifacts needed for the Red Path recovery.

Prerequisites:
    1. python src/build_data_subset.py must have run successfully.
    2. data/m5_workshop_subset.parquet must exist.

Run from the project root:
    python src/build_offline_artifacts.py [--stages all|chronos|baselines|ml|dl]

Stages
------
chronos   : Fully implemented. Runs amazon/chronos-t5-mini on WORKSHOP_SUBSET_N.
baselines : Scaffolded (TODO). Naive, SeasonalNaive, AutoETS via StatsForecast.
ml        : Scaffolded (TODO). LightGBM via MLForecast.
dl        : Scaffolded (TODO). NHITS via NeuralForecast.
all       : Runs all stages in order (default).

Offline dependencies (not in requirements.txt — install on offline build machine):
    pip install statsforecast mlforecast neuralforecast
    pip install autogluon.timeseries  # for Chronos
    # OR: pip install chronos-forecasting torch  # lighter alternative

Output artifacts:
    artifacts/04_baseline_forecasts.parquet
    artifacts/04_baseline_cv_scores.parquet
    artifacts/05_ml_forecasts.parquet
    artifacts/06_dl_forecasts.parquet
    artifacts/07_uncertainty_leaderboard.parquet
    artifacts/08_final_master_leaderboard.csv
"""

import sys
import argparse
import logging
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ARTIFACT_DIR,
    PARAMS_DIR,
    WORKSHOP_SUBSET_PATH,
    WORKSHOP_SUBSET_N,
    MICRO_SUBSET_N,
    HORIZON,
    SEASON_LENGTH,
    N_WINDOWS,
    STEP_SIZE,
    REFIT,
    RANDOM_SEED,
    INTERVAL_COVERAGE,
)
from src.schemas import validate_forecast, validate_score, validate_panel
from src.evaluation import score_forecasts, build_leaderboard

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def _load_panel() -> pd.DataFrame:
    """Load and validate the workshop subset parquet."""
    if not WORKSHOP_SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"Workshop subset not found: {WORKSHOP_SUBSET_PATH}\n"
            f"Run python src/build_data_subset.py first."
        )
    panel = pd.read_parquet(WORKSHOP_SUBSET_PATH)
    panel = validate_panel(panel, artifact_name="m5_workshop_subset")
    log.info(f"Panel loaded: {panel['unique_id'].nunique():,} series, {len(panel):,} rows.")
    return panel


def _save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    df.to_parquet(path, index=False)
    log.info(f"  ✓ Saved {label}: {path.name} ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Stage 1: Chronos — amazon/chronos-t5-mini (FULLY IMPLEMENTED)
# ---------------------------------------------------------------------------

def run_chronos(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Run amazon/chronos-t5-mini on the full workshop subset (WORKSHOP_SUBSET_N series).
    Uses the StatsForecast cross-validation cutoff logic to generate aligned forecasts.

    This is the most compute-intensive stage. On a CPU machine expect ~2-4 hours
    for 1,000 series × 3 CV windows × 28-step horizon.
    Use a GPU machine or spot instance for this stage if available.

    Returns a DataFrame conforming to the forecast schema with columns:
    [unique_id, ds, y, model, y_hat, lo_80, hi_80, cutoff, stage]
    """
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "Chronos dependencies not found.\n"
            "Install with: pip install chronos-forecasting torch\n"
            "Or via AutoGluon: pip install autogluon.timeseries"
        )

    log.info("Running Chronos (amazon/chronos-t5-mini) on full workshop subset ...")
    log.info(f"  Subset size: {panel['unique_id'].nunique():,} series")
    log.info(f"  CV windows : {N_WINDOWS}, HORIZON={HORIZON}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"  Device     : {device}")

    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-mini",
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    records = []
    panel = panel.sort_values(["unique_id", "ds"])
    end_date = panel["ds"].max()

    # Generate CV cutoffs (matches StatsForecast cross_validation logic)
    cutoffs = [
        end_date - pd.Timedelta(days=(N_WINDOWS - i) * STEP_SIZE)
        for i in range(N_WINDOWS)
    ]

    series_ids = panel["unique_id"].unique()
    log.info(f"  Running inference across {len(cutoffs)} cutoffs ...")

    # IMPLEMENTATION STRATEGY: Series-by-series inference (one pipeline.predict() call per series).
    #
    # This is intentionally the simplest correct implementation. It is easy to
    # verify, debug, and reason about. Each series is fully independent.
    #
    # EXPECTED RUNTIME (CPU): ~2-4 hours for 1,000 series × 3 cutoffs.
    # If offline testing shows this is unacceptably slow, the first and only
    # optimization to consider is batched context tensors:
    #
    #   BATCHING UPGRADE PATH (do not implement until CPU timing is confirmed slow):
    #   -------------------------------------------------------------------------
    #   Instead of calling pipeline.predict() once per series, stack context
    #   tensors for a batch of series and call pipeline.predict() once per batch.
    #   Chronos accepts a list of tensors with variable lengths as context input.
    #
    #   Rough pattern:
    #       BATCH_SIZE = 50  # tune based on available RAM
    #       for batch in chunked(series_ids, BATCH_SIZE):
    #           contexts = [torch.tensor(get_series(uid, cutoff)) for uid in batch]
    #           forecasts = pipeline.predict(contexts, prediction_length=HORIZON, ...)
    #           # forecasts shape: [BATCH_SIZE, num_samples, HORIZON]
    #           # unpack and record per-series as below
    #
    #   Note: Variable-length context tensors in a batch are padded by Chronos
    #   internally. Results are numerically equivalent to series-by-series.
    #   -------------------------------------------------------------------------

    for cutoff_idx, cutoff in enumerate(cutoffs):
        log.info(f"  Cutoff {cutoff_idx + 1}/{len(cutoffs)}: {cutoff.date()}")
        forecast_start = cutoff + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(start=forecast_start, periods=HORIZON, freq="D")

        for uid in series_ids:
            series = panel[(panel["unique_id"] == uid) & (panel["ds"] <= cutoff)]["y"]

            if len(series) < SEASON_LENGTH * 2:
                continue  # Skip series with insufficient history at this cutoff

            # Single-series context tensor: shape [1, series_length]
            # unsqueeze(0) adds the batch dimension Chronos expects
            context = torch.tensor(series.values, dtype=torch.float32).unsqueeze(0)

            forecast_tensor = pipeline.predict(
                inputs=context,
                prediction_length=HORIZON,
                num_samples=20,
            )

            # forecast_tensor shape: [1, num_samples, HORIZON]
            samples = forecast_tensor[0].numpy()  # [num_samples, HORIZON]
            y_hat = np.median(samples, axis=0)
            lo_80 = np.quantile(samples, 0.10, axis=0)
            hi_80 = np.quantile(samples, 0.90, axis=0)

            # Retrieve actuals for this window
            actuals = panel[
                (panel["unique_id"] == uid) &
                (panel["ds"].isin(forecast_dates))
            ][["ds", "y"]].set_index("ds")["y"].reindex(forecast_dates)

            for step_idx, (fdate, y_true) in enumerate(actuals.items()):
                if pd.isna(y_true):
                    continue
                records.append({
                    "unique_id": uid,
                    "ds":        fdate,
                    "y":         float(y_true),
                    "model":     "Chronos-t5-mini",
                    "y_hat":     float(max(y_hat[step_idx], 0.0)),  # Clip negatives
                    "lo_80":     float(max(lo_80[step_idx], 0.0)),
                    "hi_80":     float(max(hi_80[step_idx], 0.0)),
                    "cutoff":    cutoff,
                    "stage":     "baseline",
                })

    df = pd.DataFrame(records)
    df = validate_forecast(df, artifact_name="chronos_full_subset")

    log.info(f"  Chronos complete: {len(df):,} forecast rows across {df['unique_id'].nunique():,} series.")
    return df


# ---------------------------------------------------------------------------
# Stage 2: Classical Baselines — Naive, SeasonalNaive, AutoETS (SCAFFOLDED)
# ---------------------------------------------------------------------------

def run_baselines(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Naive, SeasonalNaive, and AutoETS via StatsForecast cross-validation.

    Returns:
        forecasts_df : Forecast DataFrame (forecast schema), including Chronos if available.
        scores_df    : Score DataFrame (score schema).
    """
    from statsforecast import StatsForecast
    from statsforecast.models import Naive, SeasonalNaive, AutoETS

    log.info("Running StatsForecast baselines (Naive, SeasonalNaive, AutoETS) ...")
    log.info(f"  Panel: {panel['unique_id'].nunique():,} series, {len(panel):,} rows")
    log.info(f"  CV: {N_WINDOWS} windows, h={HORIZON}, step={STEP_SIZE}, refit={REFIT}")

    sf = StatsForecast(
        models=[
            Naive(),
            SeasonalNaive(season_length=SEASON_LENGTH),
            AutoETS(season_length=SEASON_LENGTH),
        ],
        freq="D",
        n_jobs=-1,
    )

    cv = sf.cross_validation(
        df=panel,
        h=HORIZON,
        n_windows=N_WINDOWS,
        step_size=STEP_SIZE,
        refit=REFIT,
        level=[80],
    )
    log.info(f"  CV complete. Shape: {cv.shape}")

    # Reshape wide → long to match forecast schema
    cv = cv.reset_index()
    # reset_index() can add a spurious integer 'index' column if the df had an
    # unnamed integer index in addition to the named unique_id index — exclude it.
    meta_cols  = ["unique_id", "ds", "cutoff", "y", "index"]
    model_cols = [c for c in cv.columns
                  if c not in meta_cols and not any(x in c for x in ["-lo-", "-hi-"])]

    records = []
    for model_name in model_cols:
        lo_col = f"{model_name}-lo-80"
        hi_col = f"{model_name}-hi-80"

        chunk = cv[["unique_id", "ds", "cutoff", "y", model_name]].copy()
        chunk = chunk.rename(columns={model_name: "y_hat"})
        chunk["model"] = model_name
        chunk["stage"] = "baseline"

        if lo_col in cv.columns and hi_col in cv.columns:
            chunk["lo_80"] = cv[lo_col].values
            chunk["hi_80"] = cv[hi_col].values

        records.append(chunk)

    stat_df = pd.concat(records, ignore_index=True)
    # fillna(0) before clip: AutoETS occasionally returns NaN for some series;
    # treat NaN forecasts as zero (demand floor) so scores remain finite.
    stat_df["y_hat"] = stat_df["y_hat"].fillna(0).clip(lower=0)
    stat_df["lo_80"] = stat_df["lo_80"].fillna(0).clip(lower=0)
    stat_df["hi_80"] = stat_df["hi_80"].fillna(0).clip(lower=0)

    log.info(f"  Stat models reshaped: {len(stat_df):,} rows, "
             f"models={stat_df['model'].unique().tolist()}")

    # Merge with precomputed Chronos artifact if available
    chronos_path = ARTIFACT_DIR / "04_chronos_forecasts.parquet"
    if chronos_path.exists():
        chronos_df = pd.read_parquet(chronos_path)
        forecasts_df = pd.concat([stat_df, chronos_df], ignore_index=True)
        log.info(f"  Merged Chronos ({len(chronos_df):,} rows) into baseline artifact.")
    else:
        forecasts_df = stat_df
        log.warning("  04_chronos_forecasts.parquet not found — Chronos excluded from artifact. "
                    "Run --stages chronos first to include it.")

    forecasts_df = validate_forecast(forecasts_df, artifact_name="04_baseline_forecasts")
    _save_parquet(forecasts_df, ARTIFACT_DIR / "04_baseline_forecasts.parquet",
                  "04_baseline_forecasts")

    # Score and save CV scores
    scores_df = score_forecasts(forecasts_df, subset_name=f"workshop_{WORKSHOP_SUBSET_N}")
    scores_df = validate_score(scores_df, artifact_name="04_baseline_cv_scores")
    _save_parquet(scores_df, ARTIFACT_DIR / "04_baseline_cv_scores.parquet",
                  "04_baseline_cv_scores")

    return forecasts_df, scores_df


# ---------------------------------------------------------------------------
# Stage 3: ML — LightGBM via MLForecast (SCAFFOLDED)
# ---------------------------------------------------------------------------

def run_ml(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Run LightGBM via MLForecast with autoregressive lag features.

    # TODO: Implement after confirming lag window config with workshop authors.
    #
    # Implementation notes:
    # - MLForecast(models=[lgb.LGBMRegressor(...)], freq='D',
    #              lags=[7, 14, 21, 28], lag_transforms={...},
    #              date_features=['dayofweek', 'month'])
    # - Load params from params/mlforecast_lgb_tuned.json if USE_TUNED_PARAMS.
    # - Use cross_validation(df=panel, h=HORIZON, n_windows=N_WINDOWS,
    #                         step_size=STEP_SIZE, refit=REFIT)
    # - Conformal prediction intervals:
    #     Compute residuals per window; derive empirical 10th/90th percentile bands.
    #     Add lo_80, hi_80 columns before saving.
    # - Save artifacts/05_ml_forecasts.parquet
    # - Add 'stage' = 'ml'.
    """
    log.info("ML stage: SCAFFOLDED — see TODO in run_ml()")
    _save_placeholder_forecast(
        path=ARTIFACT_DIR / "05_ml_forecasts.parquet",
        model_names=["LightGBM"],
        panel=panel,
        stage="ml",
        label="05_ml_forecasts (placeholder)",
    )
    return pd.read_parquet(ARTIFACT_DIR / "05_ml_forecasts.parquet")


# ---------------------------------------------------------------------------
# Stage 4: Deep Learning — NHITS via NeuralForecast (SCAFFOLDED)
# ---------------------------------------------------------------------------

def run_dl(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Run NHITS via NeuralForecast.

    # TODO: Implement after confirming max_steps, val_check_steps, and
    #       early_stop_patience that reliably clear 90s on a Colab CPU.
    #
    # Implementation notes (known-safe starting point for offline testing):
    #   NHITS(h=HORIZON,
    #         input_size=2*HORIZON,
    #         max_steps=500,
    #         val_check_steps=50,
    #         early_stop_patience_steps=5,
    #         n_freq_downsample=[2, 1, 1])
    # - Load params from params/nhits_tuned.json if USE_TUNED_PARAMS.
    # - Use NeuralForecast.cross_validation() — same window/step logic.
    # - NHITS does not natively produce intervals.
    #   Options: (a) use conformal wrappers, (b) load precomputed interval artifact
    #   generated separately on GPU. Decision deferred.
    # - Add 'stage' = 'dl'.
    # - Save artifacts/06_dl_forecasts.parquet
    """
    log.info("DL stage: SCAFFOLDED — see TODO in run_dl()")
    _save_placeholder_forecast(
        path=ARTIFACT_DIR / "06_dl_forecasts.parquet",
        model_names=["NHITS"],
        panel=panel,
        stage="dl",
        label="06_dl_forecasts (placeholder)",
    )
    return pd.read_parquet(ARTIFACT_DIR / "06_dl_forecasts.parquet")


# ---------------------------------------------------------------------------
# Stage 5: Uncertainty Leaderboard & Final Leaderboard
# ---------------------------------------------------------------------------

def build_final_artifacts(
    baselines_df: pd.DataFrame,
    baseline_scores_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    dl_df: pd.DataFrame,
) -> None:
    """
    Combine all forecast DataFrames into the final leaderboard artifacts.
    This stage runs fully from whatever data the preceding stages produced —
    no TODOs. Placeholder data will produce a coherent but meaningless leaderboard.
    """
    log.info("Building uncertainty leaderboard and final leaderboard ...")

    # Merge all forecasts into one panel for scoring
    all_forecasts = pd.concat([baselines_df, ml_df, dl_df], ignore_index=True)

    # Uncertainty leaderboard (interval scores, models with lo_80/hi_80 only)
    has_intervals = all_forecasts[
        all_forecasts["lo_80"].notna() & all_forecasts["hi_80"].notna()
    ] if "lo_80" in all_forecasts.columns else pd.DataFrame()

    if not has_intervals.empty:
        uncertainty_scores = score_forecasts(has_intervals, subset_name=f"workshop_{WORKSHOP_SUBSET_N}")
        uncertainty_scores = validate_score(uncertainty_scores, artifact_name="07_uncertainty_leaderboard")
        _save_parquet(uncertainty_scores, ARTIFACT_DIR / "07_uncertainty_leaderboard.parquet",
                      "07_uncertainty_leaderboard")
    else:
        log.warning("No interval columns found — skipping uncertainty leaderboard. "
                    "Run full model stages to generate real intervals.")

    # Final point score leaderboard
    point_scores = score_forecasts(all_forecasts, subset_name=f"workshop_{WORKSHOP_SUBSET_N}")
    leaderboard = build_leaderboard([point_scores])

    leaderboard_path = ARTIFACT_DIR / "08_final_master_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    log.info(f"  ✓ Saved 08_final_master_leaderboard: {leaderboard_path.name} ({len(leaderboard)} models)")


# ---------------------------------------------------------------------------
# Placeholder Generators (used by scaffolded stages)
# ---------------------------------------------------------------------------

def _save_placeholder_forecast(
    path: Path,
    model_names: list[str],
    panel: pd.DataFrame,
    stage: str,
    label: str,
) -> None:
    """
    Write a schema-valid placeholder forecast artifact with zero y_hat values.
    Ensures downstream stages (uncertainty leaderboard, final leaderboard)
    can run without errors even before real model runs are complete.
    """
    end_date = panel["ds"].max()
    cutoffs = [
        end_date - pd.Timedelta(days=(N_WINDOWS - i) * STEP_SIZE)
        for i in range(N_WINDOWS)
    ]

    # Use a small subset for placeholder generation speed
    sample_ids = panel["unique_id"].unique()[:min(10, len(panel["unique_id"].unique()))]

    records = []
    for model in model_names:
        for cutoff in cutoffs:
            forecast_start = cutoff + pd.Timedelta(days=1)
            forecast_dates = pd.date_range(start=forecast_start, periods=HORIZON, freq="D")
            for uid in sample_ids:
                actuals = panel[
                    (panel["unique_id"] == uid) &
                    (panel["ds"].isin(forecast_dates))
                ][["ds", "y"]]
                for _, row in actuals.iterrows():
                    records.append({
                        "unique_id": uid,
                        "ds":        row["ds"],
                        "y":         float(row["y"]),
                        "model":     model,
                        "y_hat":     0.0,  # Placeholder
                        "lo_80":     0.0,
                        "hi_80":     0.0,
                        "cutoff":    cutoff,
                        "stage":     stage,
                    })

    df = pd.DataFrame(records)
    df = validate_forecast(df, artifact_name=label)
    _save_parquet(df, path, label)


def _save_placeholder_scores(
    path: Path,
    model_names: list[str],
    stage: str,
    label: str,
) -> None:
    """Write a schema-valid placeholder score artifact."""
    records = []
    for model in model_names:
        for metric in ["wMAPE", "IntervalScore_80", "Coverage_80"]:
            records.append({
                "model":             model,
                "metric":            metric,
                "score":             0.0,  # Placeholder
                "stage":             stage,
                "aggregation_scope": "pooled_all_windows",
                "subset_name":       f"workshop_{WORKSHOP_SUBSET_N}",
            })
    df = pd.DataFrame(records)
    df = validate_score(df, artifact_name=label)
    _save_parquet(df, path, label)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

STAGE_MAP = {
    "chronos":   run_chronos,
    "baselines": run_baselines,
    "ml":        run_ml,
    "dl":        run_dl,
}


def main(stages: list[str]) -> None:
    log.info("=" * 60)
    log.info("BUILD OFFLINE ARTIFACTS")
    log.info(f"  Stages: {stages}")
    log.info("=" * 60)

    panel = _load_panel()

    chronos_df    = None
    baselines_df  = None
    baseline_scores_df = None
    ml_df         = None
    dl_df         = None

    if "chronos" in stages:
        chronos_df = run_chronos(panel)
        # Save standalone Chronos artifact for inspection
        validate_forecast(chronos_df, "chronos_offline")
        _save_parquet(chronos_df, ARTIFACT_DIR / "04_chronos_forecasts.parquet",
                      "04_chronos_forecasts (standalone)")

    if "baselines" in stages:
        baselines_df, baseline_scores_df = run_baselines(panel)
        # If Chronos also ran, merge it into the baselines artifact
        if chronos_df is not None:
            baselines_df = pd.concat([baselines_df, chronos_df], ignore_index=True)
            baselines_df.to_parquet(ARTIFACT_DIR / "04_baseline_forecasts.parquet", index=False)
            log.info("  Merged Chronos into 04_baseline_forecasts.parquet")

    if "ml" in stages:
        ml_df = run_ml(panel)

    if "dl" in stages:
        dl_df = run_dl(panel)

    # Build final leaderboard artifacts if we have enough to work with
    if all(x is not None for x in [baselines_df, baseline_scores_df, ml_df, dl_df]):
        build_final_artifacts(baselines_df, baseline_scores_df, ml_df, dl_df)

    log.info("\nOffline artifact build complete.")
    log.info("Run python -c \"from src.checkpointing import list_checkpoints; list_checkpoints()\" to verify.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline workshop artifacts.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "chronos", "baselines", "ml", "dl"],
        help="Which stages to run. Default: all.",
    )
    args = parser.parse_args()

    stages = list(STAGE_MAP.keys()) if "all" in args.stages else args.stages
    main(stages)
