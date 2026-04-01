# Modern Time Series Forecasting with Python
## Packt Live Workshop

**Authors:** Jeff Tackes & Manu Joseph  
**Companion book:** *Modern Time Series Forecasting with Python* (Packt Publishing)  
**Format:** 4-hour live coding cohort (230 minutes + 10-minute buffer)  
**Execution target:** Google Colab (CPU) — local execution supported as secondary

---

## What This Workshop Builds

You will construct a full forecasting pipeline — from raw panel data to a final scored leaderboard — using the same tool stack and workflow patterns used in production environments at scale.

By the end of the session you will have:

- A performance floor from classical baselines (Naive, SeasonalNaive, AutoETS) and a modern zero-shot benchmark (Chronos)
- An ML pipeline (LightGBM via MLForecast) with autoregressive lag features and date covariates
- A global deep learning model (NHITS via NeuralForecast)
- Scored 80% prediction intervals across every model
- A final leaderboard with a justified, defensible model selection recommendation

This is not a survey of forecasting libraries. It is a disciplined, end-to-end workflow you can take directly into a production review.

---

## Before You Start: Run the Environment Check

**Run `00_env_check.ipynb` before the session begins. Do not skip it.**

It validates your Python version, all package imports, the workshop data subset, and the Red Path artifact readiness — without making any network calls. Every check either passes or gives you an explicit fix instruction.

All five checks must pass before the workshop starts.

---

## The Green Path and the Red Path

Every module has two routes:

**Green Path** — You run the cells live. This is the goal.

**Red Path** — If a cell fails or exceeds the runtime ceiling, you load a precomputed artifact and continue immediately. Every notebook has clearly labeled recovery cells.

```python
# 🔴 RED PATH — load a precomputed artifact and continue
from src.checkpointing import load_checkpoint
forecasts_df = load_checkpoint("04_baseline_forecasts")
```

Red Path cells are marked `# 🔴 RED PATH` in every notebook. Recovery takes under 2 seconds. You will never be stranded.

---

## Setup

**Recommended for the live workshop session: Google Colab.** Zero install friction, identical environment for every student, no platform issues.

Local setup is for instructors and post-workshop use. Windows requires conda — a pure pip install will not work due to OpenMP conflicts in the compiled binary stack.

---

### Option A: Google Colab (Recommended for students)

```python
# Run in a Colab cell
!git clone https://github.com/YOUR_REPO_URL packt-modern-time-series
%cd packt-modern-time-series
!pip install -q torch --index-url https://download.pytorch.org/whl/cpu
!pip install -q -r requirements.txt
```

Then open and run `00_env_check.ipynb`.

---

### Option B: Local — Windows (requires Anaconda or Miniconda)

Windows requires conda to manage the binary packages (PyTorch, LightGBM, scikit-learn, scipy). Mixing conda and pip for these packages causes OpenMP runtime conflicts that crash Jupyter kernels.

**Step 1 — Create the conda environment with all binary dependencies:**

```bat
conda create -n packt_timeseries_cohort python=3.12 -y
conda activate packt_timeseries_cohort
conda install -c conda-forge lightgbm scikit-learn scipy numpy pandas pyarrow -y
conda install -c conda-forge pytorch pytorch-lightning -y
```

**Step 2 — Install the pure-Python forecasting stack via pip:**

```bat
pip install -r requirements.txt
```

**Step 3 — Set the OpenMP environment variable (one time only):**

```bat
setx KMP_DUPLICATE_LIB_OK TRUE
```

Restart your terminal and VS Code after running this.

**Step 4 — Select the correct kernel in VS Code:**

When opening a notebook, select the kernel from `anaconda3\envs\packt_timeseries_cohort` — not any venv version. Check the kernel path shown in the top-right of the notebook matches `anaconda3`.

Then open and run `00_env_check.ipynb`.

> **Why conda?** The Nixtla stack depends on compiled Rust/C extensions (`grouped-array`, `coreforecast`) that have no Windows pip wheels. Conda-forge provides pre-built Windows binaries for the entire binary stack, which eliminates the OpenMP conflict that causes kernel crashes.

---

### Option C: Local — macOS / Linux

```bash
git clone https://github.com/YOUR_REPO_URL packt-modern-time-series
cd packt-modern-time-series

python -m venv packt_timeseries_cohort
source packt_timeseries_cohort/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then open and run `00_env_check.ipynb`.

---

### Why the named environment?

All options use `packt_timeseries_cohort` as the environment name. Your terminal shows `(packt_timeseries_cohort)` when active. Add it to `.gitignore` — environment folders should never be committed.

---

### Tested Environment

| Component | Version | Notes |
|---|---|---|
| Python | 3.12 (recommended) | 3.13 works on Colab and macOS/Linux |
| Platform | Google Colab CPU | Primary target |
| Platform | Windows 11 (conda) | Requires conda setup above |
| Platform | macOS / Linux | pip setup works cleanly |
| pandas | latest | Unpinned — conda-forge manages on Windows |
| statsforecast | latest | |
| mlforecast | latest | |
| neuralforecast | latest | |
| lightgbm | latest | conda-forge on Windows, pip elsewhere |
| torch | 2.10+ (CPU) | conda-forge on Windows, pip elsewhere |

---

## Data Setup

The workshop uses a curated, locked subset of the M5 Forecasting dataset: 1,000 item-store series selected by total sales volume, filtered for history depth and intermittency.

`data/m5_workshop_subset.parquet` must be present before the session begins. The `data/` directory exists in the repo but the parquet file is not committed — it must be obtained separately using one of the two options below.

### Option A: Download the prebuilt subset (recommended for students)

The instructor distributes `m5_workshop_subset.parquet` via a shared link before the session. Download it and place it in the `data/` directory:

```
packt-modern-time-series/
└── data/
    └── m5_workshop_subset.parquet   ← place it here
```

Then run `00_env_check.ipynb` to confirm it is valid.

### Option B: Build from raw M5 data (for instructors and advanced users)

Run the following command **on an offline machine before the workshop**. This script pulls raw files from public S3. It should never be run during a live session — an S3 outage would break it, and it takes several minutes to complete.

```bash
python src/build_data_subset.py
```

Expected output: `data/m5_workshop_subset.parquet` (~25 MB, ~1.9M rows, 1,000 series)

---

## Instructor Pre-Workshop Checklist

Complete all steps at least 24 hours before the session.

**Step 1 — Build the data subset (once per workshop)**
```bash
python src/build_data_subset.py
```
Verify: `data/m5_workshop_subset.parquet` exists and `00_env_check.ipynb` passes all checks.

**Step 2 — Precompute Red Path artifacts (once per workshop)**
```bash
python src/build_offline_artifacts.py --stages all
```
This runs the full model pipeline offline and populates `artifacts/` with all seven precomputed files. Runtime: 2–4 hours on CPU (Chronos is the bottleneck — use a GPU machine or spot instance for this step).

Verify readiness:
```python
from src.checkpointing import list_checkpoints
list_checkpoints()  # All artifacts should show ✓ EXISTS
```

**Step 3 — Distribute to students**

Share the following via Google Drive, Packt materials page, or a presigned S3 URL:
- `data/m5_workshop_subset.parquet`
- All files from the `artifacts/` directory

Students place them in the correct directories and run `00_env_check.ipynb` before the session starts.

**Step 4 — Verify your own environment**

Run `00_env_check.ipynb` end-to-end. All five checks must pass. Pay particular attention to the Red Path artifact readiness check — if any artifact is missing, the corresponding recovery cell will fail live.

---

## Workshop Schedule

| Module | Topic | Time | Type |
|---|---|---|---|
| 0 | Async Setup (pre-workshop) | — | Async |
| 1 | Welcome & Orientation | 5 min | Watch Only |
| 2 | Problem Framing & Config | 20 min | Code With Me |
| 3 | Fast EDA & Data Health | 15 min | Code With Me |
| 4 | Baselines & Foundation Model Benchmark | 45 min | Code With Me / Demo |
| 5 | ML Forecasting (LightGBM) | 45 min | Code With Me |
| 6 | Deep Learning (NHITS) | 40 min | Code With Me |
| 7 | Prediction Intervals | 25 min | Watch Only |
| 8 | Final Evaluation & Ship-It Decision | 15 min | Watch Only |
| 9 | Q&A & Next Steps | 20 min | Discussion |

**Total:** 230 minutes. 10-minute buffer is built in.

---

## Runtime Ceilings

Every live cell is designed to execute within these limits on Colab CPU:

| Cell type | Target | Hard ceiling |
|---|---|---|
| Any live cell | < 30 seconds | 90 seconds |
| NHITS training | < 60 seconds | 90 seconds |
| Chronos demo (3 series) | < 60 seconds | 90 seconds |

If a cell exceeds the hard ceiling, interrupt it and use the Red Path recovery cell below it.

---

## Repository Structure

```text
packt-modern-time-series/
│
├── README.md                          ← This file
├── TROUBLESHOOTING.md                 ← Errors and fixes
├── requirements.txt                   ← Strictly pinned dependencies
├── config.py                          ← Single source of truth for all settings
├── 00_env_check.ipynb                 ← Run before the workshop
├── take_home_bonus.ipynb              ← Advanced: HPO and scaling
│
├── companion_asset/
│   └── enterprise_forecasting_tech_stack_rubric.md
│
├── data/
│   └── m5_workshop_subset.parquet     ← Workshop data (not in repo — see Data Setup)
│
├── params/                            ← Pre-tuned model parameter files
│   ├── mlforecast_lgb_default.json
│   ├── mlforecast_lgb_tuned.json
│   └── nhits_tuned.json
│
├── src/                               ← Shared helpers
│   ├── __init__.py
│   ├── checkpointing.py               ← load_checkpoint() — all Red Path recovery
│   ├── evaluation.py                  ← pooled wMAPE and interval score
│   ├── schemas.py                     ← artifact schema validation
│   ├── plotting.py                    ← shared visualization helpers
│   ├── build_data_subset.py           ← offline: S3 → local parquet
│   └── build_offline_artifacts.py     ← offline: precompute Red Path artifacts
│
├── artifacts/                         ← Precomputed outputs (populated before the session)
│   ├── 00_env_status.json
│   ├── 04_baseline_forecasts.parquet
│   ├── 04_baseline_cv_scores.parquet
│   ├── 05_ml_forecasts.parquet
│   ├── 06_dl_forecasts.parquet
│   ├── 07_uncertainty_leaderboard.parquet
│   └── 08_final_master_leaderboard.csv
│
├── instructor_notebooks/
│   ├── 01_welcome.ipynb
│   ├── 02_framing_and_config.ipynb
│   ├── 03_eda_and_health.ipynb
│   ├── 04_baselines_and_fm.ipynb
│   ├── 05_ml_forecasting.ipynb
│   ├── 06_deep_learning.ipynb
│   ├── 07_prediction_intervals.ipynb
│   ├── 08_final_evaluation.ipynb
│   └── 09_qa_and_next_steps.ipynb
│
└── student_notebooks/
    └── ... (matching structure, __FILL_IN__ blocks, cleared outputs)
```

---

## Key Design Rules

**Single source of truth.** `config.py` governs every constant — horizon, season length, subset sizes, random seed, file paths. Notebooks import from it. They do not redefine values.

**No magic.** Core modeling logic stays visible in the notebooks. `src/` handles checkpointing, plotting, and evaluation only. Feature engineering, model configuration, and training are always in the notebook cells.

**Pooled scoring.** All metrics (wMAPE, Interval Score) are computed pooled across all observations from all backtest windows. Per-series averages are not used for ranking.

**CPU-safe ceilings.** Any cell meant to run live is tested against the 90-second hard ceiling on Colab CPU. Cells that cannot reliably clear this ceiling use precomputed artifacts instead.

---

## Companion Asset

The **Enterprise Forecasting Tech Stack Rubric** is a practitioner evaluation tool for scoring and selecting forecasting infrastructure across six dimensions: model selection, feature pipeline architecture, serving strategy, evaluation rigor, monitoring, and organizational fit.

Available in `companion_asset/enterprise_forecasting_tech_stack_rubric.md`.

---

## Issues and Questions

See `TROUBLESHOOTING.md` for common environment and runtime errors.

For issues not covered there, open a GitHub issue on the workshop repository.
