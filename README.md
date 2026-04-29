# Modern Time Series Forecasting with Python
## Packt Live Workshop

**Authors:** Jeff Tackes & Manu Joseph  
**Companion book:** *Modern Time Series Forecasting with Python* (Packt Publishing)  
**Format:** 4-hour live coding cohort

---

## Workshop Notebooks

Open each notebook directly in Colab using the links below. Run them in order.

| # | Topic | Open in Colab |
|---|---|---|
| 01 | Welcome & Orientation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/01_welcome.ipynb) |
| 02 | Problem Framing & Config | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/02_framing_and_config.ipynb) |
| 03 | EDA & Data Health | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/03_eda_and_health.ipynb) |
| 04 | Baselines & Foundation Model Benchmark | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/04_baselines_and_fm.ipynb) |
| 05 | ML Forecasting (LightGBM) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/05_ml_forecasting.ipynb) |
| 06 | Deep Learning (NHITS) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/06_deep_learning.ipynb) |
| 07 | Prediction Intervals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/07_prediction_intervals.ipynb) |
| 08 | Final Evaluation & Ship-It Decision | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/08_final_evaluation.ipynb) |
| 09 | Q&A & Next Steps | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/09_qa_and_next_steps.ipynb) |

> **Each notebook has a setup cell at the top.** Run it first — it clones the repo (data and artifacts included) and installs only the packages that notebook needs. You do not need to upload any files.

---

## Environment Setup

**Recommended: VS Code local (Windows/Mac/Linux) or Google Colab.**

See [SETUP.md](SETUP.md) for full step-by-step instructions for all platforms. If you are following along with Colab, the setup cell in each notebook handles everything.

---

## What This Workshop Builds

A complete forecasting pipeline — from raw panel data to a final scored leaderboard — using the same tool stack and workflow patterns used in production.

By the end you will have:

- A performance floor from classical baselines (Naive, SeasonalNaive, AutoETS) and a zero-shot benchmark (Chronos)
- An ML pipeline (LightGBM via MLForecast) with lag features and date covariates
- A global deep learning model (NHITS via NeuralForecast)
- Scored 80% prediction intervals across every model
- A final leaderboard with a justified, defensible model selection recommendation

---

## The Green Path and Red Path

Every module has two routes.

**Green Path** — You run the cells live. This is the goal.

**Red Path** — If a cell fails or exceeds the runtime ceiling, you load a precomputed artifact and continue immediately. Recovery takes under 2 seconds. You will not fall behind.

```python
# 🔴 RED PATH — load a precomputed artifact and continue
from src.checkpointing import load_checkpoint
forecasts_df = load_checkpoint("04_baseline_forecasts")
```

---

## Workshop Schedule

| Module | Topic | Time | Type |
|---|---|---|---|
| 1 | Welcome & Orientation | 5 min | Watch Only |
| 2 | Problem Framing & Config | 20 min | Code With Me |
| 3 | Fast EDA & Data Health | 15 min | Code With Me |
| 4 | Baselines & Foundation Model Benchmark | 45 min | Code With Me / Demo |
<<  5 minute break >>
| 5 | ML Forecasting (LightGBM) | 45 min | Code With Me |
<<  5 minute break >>
| 6 | Deep Learning (NHITS) | 40 min | Code With Me |
| 7 | Prediction Intervals | 25 min | Watch Only |
| 8 | Final Evaluation & Ship-It Decision | 15 min | Watch Only |
| 9 | Q&A & Next Steps | 20 min | Discussion |

---

## Runtime Ceilings

Every live cell is designed to execute within these limits on Colab CPU:

| Cell type | Target | Hard ceiling |
|---|---|---|
| Any live cell | < 30 seconds | 90 seconds |
| NHITS training | < 60 seconds | 90 seconds |
| Chronos demo | < 60 seconds | 90 seconds |

If a cell exceeds the hard ceiling, interrupt it and use the Red Path recovery cell below it.

---

## Repository Structure

```text
packt-modern-time-series/
│
├── README.md                          ← This file
├── SETUP.md                           ← Full environment setup guide
├── TROUBLESHOOTING.md                 ← Errors and fixes
├── requirements.txt                   ← Dependencies
├── config.py                          ← Single source of truth for all settings
├── 00_env_check.ipynb                 ← Run before the workshop (local setup)
├── take_home_bonus.ipynb              ← Advanced: HPO and scaling
│
├── data/
│   └── m5_workshop_subset.parquet     ← Workshop dataset (included in repo)
│
├── artifacts/                         ← Precomputed Red Path artifacts (included in repo)
│
├── params/                            ← Pre-tuned model parameter files
│
├── src/                               ← Shared helpers
│   ├── checkpointing.py               ← load_checkpoint() — all Red Path recovery
│   ├── evaluation.py                  ← pooled wMAPE and interval score
│   ├── schemas.py                     ← artifact schema validation
│   ├── build_offline_artifacts.py     ← offline: precompute Red Path artifacts
│   └── build_data_subset.py           ← offline: rebuild data subset from raw M5
│
├── student_notebooks/                 ← Notebooks for workshop participants
│   ├── 01_welcome.ipynb
│   ├── 02_framing_and_config.ipynb
│   └── ... (through 09)
│
└── instructor_notebooks/              ← Instructor versions with full outputs
    └── ... (matching structure)
```

---

## Key Design Rules

**Single source of truth.** `config.py` governs every constant. Notebooks import from it and never redefine values.

**No magic.** Core modeling logic stays visible in the notebooks. `src/` handles checkpointing, plotting, and evaluation only.

**Pooled scoring.** All metrics (wMAPE, Interval Score) are computed pooled across all observations from all backtest windows — not averaged per series first.

**CPU-safe ceilings.** Any cell meant to run live is tested against the 90-second hard ceiling on Colab CPU.

---

## Companion Asset

The **Enterprise Forecasting Tech Stack Rubric** is a six-dimension maturity assessment for scoring and selecting forecasting infrastructure. Available in `companion_asset/enterprise_forecasting_tech_stack_rubric.md`.

---

## Issues and Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common environment and runtime errors.
