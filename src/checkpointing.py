"""
src/checkpointing.py — Red Path Recovery Interface
====================================================
Every checkpoint recovery cell in every notebook must use this module.
No ad hoc artifact loading logic belongs inside a notebook.

Usage (in any notebook)
------------------------
    from src.checkpointing import load_checkpoint

    df = load_checkpoint("04_baseline_forecasts")
    df = load_checkpoint("04_baseline_cv_scores")
    df = load_checkpoint("03_validated_panel")

The function resolves the path from the ARTIFACT_REGISTRY in config.py,
validates the schema via src/schemas.py, and returns a clean DataFrame.
"""

import json
import pickle
import pandas as pd
from pathlib import Path

# config.py is at the project root. When called from a notebook,
# sys.path must include the project root (handled by 00_env_check.ipynb).
from config import ARTIFACT_REGISTRY, ARTIFACT_SCHEMA_MAP
from src.schemas import validate


# ---------------------------------------------------------------------------
# Primary Interface
# ---------------------------------------------------------------------------

def load_checkpoint(artifact_name: str) -> pd.DataFrame:
    """
    Load a precomputed artifact by its exact registry name.

    Resolves the file path from ARTIFACT_REGISTRY, reads the file,
    validates the schema and enforces dtypes, and returns a DataFrame.

    Parameters
    ----------
    artifact_name : str
        Exact key from config.ARTIFACT_REGISTRY.
        Examples: "04_baseline_forecasts", "07_uncertainty_leaderboard"

    Returns
    -------
    pd.DataFrame — validated and dtype-coerced.

    Raises
    ------
    KeyError   — artifact_name not in ARTIFACT_REGISTRY.
    FileNotFoundError — artifact file does not exist on disk.
    ValueError — schema validation failure (see src/schemas.py).
    """
    # --- 1. Resolve path ---
    if artifact_name not in ARTIFACT_REGISTRY:
        available = sorted(ARTIFACT_REGISTRY.keys())
        raise KeyError(
            f"Unknown artifact: '{artifact_name}'.\n"
            f"Available artifacts:\n  " + "\n  ".join(available)
        )

    artifact_path = ARTIFACT_REGISTRY[artifact_name]

    # --- 2. Existence check ---
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_path}\n"
            f"Run src/build_offline_artifacts.py to generate precomputed artifacts "
            f"before the workshop session."
        )

    # --- 3. Load by file type ---
    suffix = artifact_path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(artifact_path)
    elif suffix == ".csv":
        df = pd.read_csv(artifact_path)
    elif suffix == ".pkl":
        return _load_pickle(artifact_path, artifact_name)
    elif suffix == ".json":
        return _load_json(artifact_path, artifact_name)
    else:
        raise ValueError(
            f"Unsupported artifact file type: '{suffix}' for artifact '{artifact_name}'."
        )

    # --- 4. Schema validation ---
    schema_key = ARTIFACT_SCHEMA_MAP.get(artifact_name)
    if schema_key:
        df = validate(df, schema_key, artifact_name=artifact_name)
    else:
        # Artifacts without a schema entry (e.g., 02_global_config) are returned as-is.
        pass

    _print_recovery_confirmation(artifact_name, artifact_path, len(df))
    return df


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: Path, artifact_name: str) -> object:
    """Load a .pkl artifact. Returns raw object — no schema validation."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"  ✓ [{artifact_name}] Loaded from {path.name}")
    return obj


def _load_json(path: Path, artifact_name: str) -> dict:
    """Load a .json artifact. Returns raw dict — no schema validation."""
    with open(path, "r") as f:
        obj = json.load(f)
    print(f"  ✓ [{artifact_name}] Loaded from {path.name}")
    return obj


def _print_recovery_confirmation(artifact_name: str, path: Path, n_rows: int) -> None:
    """
    Print a consistent, visible confirmation message after every checkpoint load.
    This is intentional — students must see that recovery succeeded.
    """
    print(
        f"\n  ✓ RED PATH RECOVERY COMPLETE\n"
        f"    Artifact : {artifact_name}\n"
        f"    File     : {path.name}\n"
        f"    Rows     : {n_rows:,}\n"
    )


# ---------------------------------------------------------------------------
# Utility: List all available checkpoints
# ---------------------------------------------------------------------------

def list_checkpoints() -> None:
    """
    Print all registered artifacts and whether they exist on disk.
    Useful for a quick pre-workshop readiness check.

    Usage
    -----
    from src.checkpointing import list_checkpoints
    list_checkpoints()
    """
    print("\n  ARTIFACT REGISTRY STATUS\n  " + "-" * 40)
    for name, path in sorted(ARTIFACT_REGISTRY.items()):
        status = "✓ EXISTS" if path.exists() else "✗ MISSING"
        print(f"  {status}  {name:<35}  {path.name}")
    print()
