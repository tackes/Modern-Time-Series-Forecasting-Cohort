# TROUBLESHOOTING

Common errors, causes, and fixes. Organized by failure category.

If your error is not here, open a GitHub issue with the full error message and the cell that produced it.

---

## 1. Environment & Installation

### `ModuleNotFoundError: No module named 'X'`

A required package is missing from the environment.

```bash
pip install -r requirements.txt
```

On Colab, if you installed packages and the error persists, the runtime needs a restart:
**Runtime → Restart session** — then re-run `!pip install -r requirements.txt` before opening any notebook.

---

### `ImportError` on `neuralforecast` after a successful install

NeuralForecast requires PyTorch and PyTorch Lightning. On Colab, a pre-existing PyTorch installation may conflict with the pinned version in `requirements.txt`.

Fix:

```bash
pip install --upgrade neuralforecast==3.1.5 pytorch-lightning==2.4.0
```

If the conflict persists, use a fresh Colab runtime (**Runtime → Factory reset runtime**) and install from scratch.

---

### `ImportError: cannot import name 'ChronosPipeline' from 'chronos'`

Chronos is an offline-only dependency. It is not in `requirements.txt` because it is not used during the live session — only during the precompute phase on the build machine.

If you are running the offline build:

```bash
pip install chronos-forecasting torch
```

If you see this error during the live workshop, the instructor demo cell has a dependency that should not be in the student notebook. Use the Red Path:

```python
# 🔴 RED PATH
from src.checkpointing import load_checkpoint
forecasts_df = load_checkpoint("04_baseline_forecasts")
```

---

### Version mismatch warnings in `00_env_check.ipynb`

The env check warns (but does not fail hard) if an installed package version differs from the pinned version. Warnings are safe to proceed with for minor patch differences (e.g., `2.0.2` vs `2.0.3`).

For major version differences — especially in the Nixtla stack — reinstall from `requirements.txt` on a clean environment.

---

## 2. Data Errors

### `FileNotFoundError: data/m5_workshop_subset.parquet not found`

The workshop data file is not included in the repository. It must be placed in `data/` before the session.

**Fix A (recommended):** Download `m5_workshop_subset.parquet` from the Packt workshop materials link and place it in the `data/` directory.

**Fix B:** Build it from raw M5 files on an offline machine:

```bash
python src/build_data_subset.py
```

This script pulls from public S3. Do not run it during a live session.

---

### `ValueError: Schema violation — missing required columns`

Your `m5_workshop_subset.parquet` was built with a different column naming convention. The workshop expects `unique_id`, `ds`, `y`.

**Fix:** Regenerate using the current `src/build_data_subset.py`. The script enforces the correct column names.

---

### `ValueError: Only N series found, expected ~1000`

The subset policy filters (history depth, intermittency threshold) removed more series than expected. This can happen if the raw M5 file used was the training set instead of the evaluation set.

**Fix:** Confirm that `build_data_subset.py` is pulling `sales_train_evaluation.csv` from S3, not `sales_train_validation.csv`. The evaluation file has the full 1,941-day history required.

---

### `ValueError: Date range too short`

The parquet file does not have enough history to support the cross-validation configuration (3 windows × 28-step horizon, plus 365 days of pre-cutoff history).

**Fix:** Same as above — confirm the evaluation file was used, not the validation file.

---

## 3. Runtime and Performance

### A cell runs for more than 90 seconds

The 90-second ceiling is the hard limit for any live cell. If a cell exceeds it:

1. Interrupt the kernel (square stop button in Colab toolbar, or `I, I` keyboard shortcut)
2. Use the Red Path recovery cell immediately below the slow cell

```python
# 🔴 RED PATH — use this if the cell above timed out
from src.checkpointing import load_checkpoint
forecasts_df = load_checkpoint("05_ml_forecasts")  # adjust name to current module
```

This is expected behavior, not a failure. The Red Path exists precisely for this.

---

### NHITS training exceeds 90 seconds

NHITS runtime on Colab CPU varies based on the session's allocated resources. The notebook uses `MICRO_SUBSET_N = 50` series and a capped `max_steps` specifically to target sub-90-second execution — but Colab CPU allocation is not guaranteed.

If NHITS times out:

```python
# 🔴 RED PATH
from src.checkpointing import load_checkpoint
dl_forecasts = load_checkpoint("06_dl_forecasts")
```

---

### LightGBM runs slowly on the micro subset

LightGBM on 50 series should complete in under 30 seconds. If it is slow:

- Confirm `MICRO_SUBSET_N` is imported from `config.py`, not redefined in the notebook
- Check **Runtime → Manage sessions** in Colab — you may be on a shared CPU allocation
- If you added lag windows or extra features beyond what the notebook specifies, revert to the notebook defaults for the live run

---

### Colab disconnects mid-session

1. Reconnect: **Runtime → Reconnect**
2. Re-run the setup cells at the top of your current notebook (imports and config load)
3. Use `load_checkpoint()` to restore the last completed artifact — you do not need to re-run earlier modules

```python
from src.checkpointing import load_checkpoint, list_checkpoints

# See what is available
list_checkpoints()

# Restore the last artifact you completed
df = load_checkpoint("05_ml_forecasts")
```

---

## 4. Checkpoint and Artifact Errors

### `FileNotFoundError: Artifact not found — run src/build_offline_artifacts.py`

The precomputed artifact is missing from `artifacts/`. This is an instructor setup issue.

**Instructor fix:** Run `src/build_offline_artifacts.py` on the build machine and copy the resulting files into `artifacts/` before the session. Check readiness with:

```python
from src.checkpointing import list_checkpoints
list_checkpoints()
```

---

### `KeyError: Unknown artifact: 'XYZ'`

The name passed to `load_checkpoint()` is not registered in `config.ARTIFACT_REGISTRY`. Use the exact names listed below.

Valid artifact names:

```
03_validated_panel
04_baseline_forecasts
04_baseline_cv_scores
05_ml_forecasts
06_dl_forecasts
07_uncertainty_leaderboard
08_final_master_leaderboard
02_global_config
00_env_status
```

---

### `ValueError: Schema violation — wrong dtype or missing column`

An artifact in `artifacts/` was generated by an older version of the pipeline with a different schema. The current `src/schemas.py` is rejecting it.

**Fix:** Regenerate the artifact by running the upstream module or re-running `src/build_offline_artifacts.py`.

---

### `ValueError: aggregation_scope must be 'pooled_all_windows'`

A score artifact contains rows where `aggregation_scope` is not `'pooled_all_windows'`. This means the artifact was saved with an incorrect value before the schema check ran.

**Fix:** Regenerate the score artifact. The upstream scoring call in `src/evaluation.py` always sets this correctly — the issue is a manually constructed or patched artifact file.

---

## 5. Model-Specific Errors

### AutoETS: `optimization did not converge` warnings

StatsForecast's AutoETS handles convergence failures internally by falling back to a simpler model. Convergence warnings are expected on some M5 series and do not stop execution. The forecast will still be generated.

---

### Chronos: `CUDA out of memory`

The live instructor demo of Chronos runs on 3 series only. If CUDA OOM occurs, force CPU:

```python
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-mini",
    device_map="cpu",
    torch_dtype=torch.float32,
)
```

CPU inference on 3 series is fast enough for the demo. The full-subset Chronos results always come from the precomputed artifact regardless.

---

### `ValueError: sum of actuals is zero — pooled wMAPE undefined`

The evaluation window for one or more series contains all-zero actuals. The intermittency filter in `build_data_subset.py` should have excluded these series, but a corner case slipped through.

**Fix:** This will not crash the full pipeline — `pooled_wmape()` in `src/evaluation.py` returns `NaN` in this case rather than dividing by zero. The NaN will propagate to the leaderboard and be visible. No action required unless the affected model count is large.

---

### `AttributeError` or `TypeError` after a Nixtla library upgrade

The Nixtla stack (statsforecast, mlforecast, neuralforecast) has a history of breaking API changes between minor versions. If you installed versions outside of `requirements.txt`:

```bash
pip install -r requirements.txt --force-reinstall
```

Then restart the kernel.

---

## 6. Notebook-Specific

### Student notebook: `__FILL_IN__` block causes a `SyntaxError`

The `__FILL_IN__` placeholder is intentional — it marks code you are expected to write during the workshop. Replace it with the correct expression before running the cell.

If you are stuck, look at the matching instructor notebook for the solution.

---

### `config.py` values are not reflecting after a change

`config.py` is imported once at kernel startup. If you edit it, you must restart the kernel for changes to take effect. Do not restart mid-session unless necessary — import the updated value directly instead:

```python
import importlib, config
importlib.reload(config)
from config import HORIZON  # re-import the specific value
```

---

### `ModuleNotFoundError: No module named 'config'` or `'src'`

The project root is not on `sys.path`. This happens when the notebook is opened from a different working directory (common on Colab after a runtime reset).

Fix by adding the path explicitly at the top of the notebook:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve()))  # adds current directory
```

Or, if running from a subdirectory:

```python
sys.path.insert(0, str(Path().resolve().parent))
```
