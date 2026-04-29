# Modern Time Series Forecasting Cohort — Student Notebooks

Welcome to the hands-on cohort workshop for **Modern Time Series Forecasting with Python**.

This workshop is designed to be practical, workflow-driven, and Nixtla-first. We will work through a realistic forecasting pipeline using the M5 dataset, moving from setup and problem framing through EDA, baselines, machine learning, deep learning, prediction intervals, and a final ship-it decision.

The goal is not to build a toy model zoo.

The goal is to learn how a forecasting practitioner thinks through a full modeling workflow:

- What problem are we actually solving?
- What data do we trust?
- What is the performance floor?
- When is machine learning worth the added complexity?
- How do we evaluate uncertainty?
- What would we actually consider putting into production?

---

## How to use these notebooks

The easiest way to follow the workshop is to open each notebook directly in Google Colab using the links below.

Each notebook opens in a Jupyter-style Colab environment.

> **Important:** Opening a notebook from GitHub does not automatically clone the full repository into your Colab runtime.  
> The first setup cell in the notebooks will handle the required repo setup, paths, and dependencies.

Run the notebooks in order unless the instructor tells you otherwise.

---

## Workshop Notebooks

| Module | Notebook | Purpose | Open in Colab |
|---|---|---|---|
| 01 | Welcome | Workshop orientation, workflow overview, Green Path / Red Path logic | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/01_welcome.ipynb) |
| 02 | Framing & Config | Define the forecasting problem, lock core settings, understand the workflow contract | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/02_framing_and_config.ipynb) |
| 03 | EDA & Health | Inspect the panel, validate data health, identify modeling risks before training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/03_eda_and_health.ipynb) |
| 04 | Baselines & Foundation Model Benchmark | Build the baseline performance floor using StatsForecast and a Chronos benchmark | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/04_baselines_and_fm.ipynb) |
| 05 | ML Forecasting | Use MLForecast and LightGBM to build a feature-based global forecasting model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/05_ml_forecasting.ipynb) |
| 06 | Deep Learning | Use NeuralForecast and NHITS to introduce deep learning for time series forecasting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/06_deep_learning.ipynb) |
| 07 | Prediction Intervals | Evaluate forecast uncertainty using coverage and interval score / Winkler-style scoring | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/07_prediction_intervals.ipynb) |
| 08 | Final Evaluation | Compare models across accuracy, bias, uncertainty, runtime, and production readiness | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/08_final_evaluation.ipynb) |
| 09 | Q&A & Next Steps | Wrap-up, key takeaways, extension ideas, and next steps after the workshop | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tackes/Modern-Time-Series-Forecasting-Cohort/blob/main/student_notebooks/09_qa_and_next_steps.ipynb) |

---

## Recommended workflow

1. Start with **Notebook 01**.
2. Run each notebook from top to bottom.
3. If you fall behind, use the provided checkpoint / artifact loading cells.
4. Do not worry if every model does not fully train live. Some steps are intentionally supported by precomputed artifacts so the workshop can focus on the forecasting workflow rather than waiting on long-running jobs.
5. Keep the final evaluation mindset in view: the best model is not always the most complex model.

---

## Green Path and Red Path

This workshop uses a **Green Path / Red Path** design.

### Green Path

The Green Path is the intended live execution path.

Use it when:

- The notebook runs successfully.
- Runtime is reasonable.
- You are keeping pace with the workshop.

### Red Path

The Red Path is the recovery path.

Use it when:

- A model takes too long.
- A dependency fails.
- You fall behind.
- The instructor tells the group to load precomputed artifacts.

The Red Path is not a failure path. It is a practical workflow pattern. In real forecasting systems, expensive model runs are often separated from downstream analysis through saved artifacts.

---

## Workshop philosophy

This workshop is built around a practical forecasting workflow:

```text
Problem framing
    ↓
Data health and EDA
    ↓
Strong baselines
    ↓
Machine learning
    ↓
Deep learning
    ↓
Foundation model benchmark
    ↓
Prediction intervals
    ↓
Final ship-it decision