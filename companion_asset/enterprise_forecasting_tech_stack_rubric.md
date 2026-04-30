# Enterprise Forecasting System Maturity Rubric
## A Practitioner Tool for Diagnosing Forecasting Infrastructure, Governance, and Operational Readiness

**Authors:** Jeff Tackes & Manu Joseph  
**Companion to:** *Modern Time Series Forecasting with Python* (Packt Publishing)

---

This rubric is a decision-support tool for evaluating forecasting infrastructure across six dimensions. Use it to score your current stack, identify gaps, and prioritize investments. It is not a benchmarking framework — it is an operational maturity assessment.

**How to use it:**  
Score each dimension from 1 to 4 using the descriptors below. Total your score. Use the gap analysis at the end to identify where to invest next.

The six dimensions follow the natural investment sequence:

| # | Dimension | The question it answers |
|---|---|---|
| 1 | Metric Architecture & Evaluation Rigor | Are we measuring the right thing? |
| 2 | Model Selection & Champion/Challenger Discipline | Are we choosing the right model? |
| 3 | Data Contract & Feature Pipeline Architecture | Can we reproduce the inputs? |
| 4 | Serving and Inference Architecture | Can we deliver forecasts reliably? |
| 5 | Monitoring and Drift Detection | Can we detect failure? |
| 6 | Organizational Fit and Ownership | Can the organization sustain it? |

---

## Dimension 1: Metric Architecture & Evaluation Rigor

*Are you measuring what matters — and measuring it correctly?*

| Score | Descriptor |
|---|---|
| **1** | RMSE or MAE only. No interval evaluation. No connection to business outcomes. Per-series averaging in use. |
| **2** | wMAPE computed. Baselines included. But still averaging per-series, intervals are not evaluated, and there is no business metric translation. |
| **3** | Pooled wMAPE and bias tracked. Interval Score and Coverage used correctly: coverage as a diagnostic, Interval Score for ranking. Business metric translation documented (e.g., "1 wMAPE point = X days of safety stock"). |
| **4** | Full metric suite: pooled point and interval metrics, bias decomposed by segment, segment-level evaluation (by category, volatility tier, or business unit), and a documented threshold for what improvement is worth deploying. |

**Your score:** ___

**The gap that costs the most:** Averaging per-series wMAPE. It gives equal weight to a SKU selling 1 unit/day and one selling 10,000. Your evaluation does not reflect where the money is. Pooling is not a technical detail — it is a business-alignment decision.

---

## Dimension 2: Model Selection & Champion/Challenger Discipline

*Are you choosing models based on evidence, with a process you can repeat and defend?*

| Score | Descriptor |
|---|---|
| **1** | No formal selection process. Models are chosen because "we've always used this" or because a vendor demo looked good. |
| **2** | Some backtesting exists but uses a single holdout window, does not include baselines, and results are not documented or reproducible. |
| **3** | Baselines always included. Multi-window cross-validation. Results documented and reproducible. A deployment threshold exists — the challenger must beat the champion by a defined margin before replacing it. |
| **4** | Full champion/challenger process: every retraining cycle runs the challenger against the champion on held-out data. Model changes are version-controlled. Deployment decisions are documented with evidence. Regression detection is automated. |

**Your score:** ___

**The gap that costs the most:** No deployment threshold. Without a clear standard for what improvement justifies a model change, every retrain is a judgment call. Judgment calls accumulate bias over time — usually toward complexity.

---

## Dimension 3: Data Contract & Feature Pipeline Architecture

*Can your target definition, calendar, features, and data snapshots be trusted in production?*

| Score | Descriptor |
|---|---|
| **1** | Target definition is informal. Features computed ad hoc in notebooks. No versioning. No handling of outliers, stockouts, or late-arriving data. |
| **2** | Target and grain are documented. Feature computation is scripted but runs inline with training. Backfill is manual. No audit trail. |
| **3** | Target definition is explicit and enforced: grain, date alignment, actuals finalization logic, and outlier handling are all specified. Governed feature pipeline with versioned snapshots. Late-arriving data triggers targeted backfill jobs. |
| **4** | Full data contract: target, horizon, and actuals finalization are formally specified and validated before every training run. Feature lineage tracked. Data quality SLAs enforced. Training and inference consume features from the same versioned snapshot. |

**Your score:** ___

**The gap that costs the most:** An undefined target. Many forecasting failures happen before features are built — the wrong grain, misaligned dates, actuals that include returns or substitutions, or a horizon that does not match the business decision. A model trained on the wrong question answers the wrong question precisely.

---

## Dimension 4: Serving and Inference Architecture

*Can you deliver forecasts reliably at the cadence your business requires?*

| Score | Descriptor |
|---|---|
| **1** | Forecasts are generated by running a Jupyter notebook manually. Delivery is ad hoc. |
| **2** | Scheduled script or cron job. Single point of failure. No monitoring on job completion or forecast freshness. |
| **3** | Orchestrated pipeline (Airflow, Prefect, or equivalent). Retries on failure. Forecast freshness monitored. Alerts on job failure. |
| **4** | Fully observable pipeline with SLAs. Forecast freshness SLA enforced. Degraded-mode fallback (e.g., serve last-good forecast if retraining fails). Canary deployment for model updates. |

**Your score:** ___

**The gap that costs the most:** No degraded-mode fallback. When your retraining job fails at 2am — and it will — your procurement team needs a forecast by 7am. Without a fallback, they get nothing, or they use a stale forecast without knowing it is stale.

---

## Dimension 5: Monitoring and Drift Detection

*Do you know when your model stops working before your business does?*

| Score | Descriptor |
|---|---|
| **1** | No monitoring. Model degradation is discovered when a stakeholder complains. |
| **2** | Manual review of forecast accuracy on a monthly or quarterly basis. No automated alerts. |
| **3** | Automated wMAPE tracking per model per segment. Alerts when rolling accuracy drops below threshold. Retraining triggered automatically on drift detection. |
| **4** | Multi-layer monitoring: feature drift detection, prediction distribution shift, business metric correlation, and a documented incident response playbook for forecast failures. Coverage calibration tracked over time — interval drift detected independently from point accuracy drift. |

**Your score:** ___

**The gap that costs the most:** Monitoring point accuracy without monitoring interval calibration. A model can maintain its wMAPE while its intervals widen progressively — a signal that input volatility has increased. By the time point accuracy degrades, your safety stock buffers have already been exhausted.

---

## Dimension 6: Organizational Fit and Ownership

*Does the forecasting system have a sustainable owner?*

| Score | Descriptor |
|---|---|
| **1** | One person built it and only they understand it. No documentation. Bus factor = 1. |
| **2** | Some documentation exists. More than one person can run the pipeline, but not debug it. No formal handoff process. |
| **3** | Documented architecture. Runbook exists. On-call rotation covers forecast pipeline failures. New team members can be onboarded in under a week. |
| **4** | Forecasting treated as a product, not a project. Dedicated owner with a roadmap. SLAs agreed with stakeholders. Regular accuracy reviews with business partners. Clear escalation path when a forecast drives a bad business outcome. |

**Your score:** ___

**The gap that costs the most:** Bus factor = 1. When the person who built the system leaves or is unavailable during a critical period, the business loses access to its forecasting capability entirely. This is a risk that no amount of model accuracy improvement can mitigate.

---

## Scoring Summary

| Dimension | Your Score (1–4) |
|---|---|
| 1. Metric Architecture & Evaluation Rigor | ___ |
| 2. Model Selection & Champion/Challenger Discipline | ___ |
| 3. Data Contract & Feature Pipeline Architecture | ___ |
| 4. Serving and Inference Architecture | ___ |
| 5. Monitoring and Drift Detection | ___ |
| 6. Organizational Fit and Ownership | ___ |
| **Total** | ___ / 24 |

---

## Maturity Bands

| Total Score | Band | What it means |
|---|---|---|
| 6–10 | **Experimental** | Forecasting is a research activity, not a production system. The gap between your notebook and a reliable business tool is measured in months of engineering. |
| 11–15 | **Functional** | You have a working pipeline but meaningful operational gaps. One or two failures away from a business incident. Prioritize monitoring and fallback before adding model complexity. |
| 16–20 | **Operational** | Solid foundation. Most gaps are in the refinement tier — better interval calibration, more granular evaluation, or tighter SLAs. You can safely invest in model improvement now. |
| 21–24 | **Production-grade** | Your forecasting infrastructure is a genuine business asset. Focus on continuous improvement, stakeholder trust, and expanding coverage to new use cases. |

---

## Gap Analysis: Where to Invest Next

**If your lowest score is Dimension 1 (Metric Architecture):**  
Fix your measurement before anything else. A leaderboard built on flawed metrics will steer every downstream decision — model selection, retraining triggers, deployment thresholds — in the wrong direction.

**If your lowest score is Dimension 2 (Model Selection):**  
Stop adding model complexity. You do not have a reliable process for knowing whether a new model is better than the one it replaces. Fix the selection discipline first.

**If your lowest score is Dimension 3 (Data Contract & Features):**  
Your models may be answering the wrong question or running on inputs that cannot be reproduced. Invest in a governed feature pipeline before scaling the model portfolio.

**If your lowest score is Dimension 4 (Serving):**  
You have a forecast system, not a forecast service. The difference matters when the business depends on it at 6am on a Monday.

**If your lowest score is Dimension 5 (Monitoring):**  
You will discover model degradation after it has already caused a business problem. Add monitoring before adding models.

**If your lowest score is Dimension 6 (Organizational Fit):**  
Technical maturity without organizational ownership is a liability, not an asset. Fix the bus factor before the bus arrives.

---

## A Note on Sequencing

The dimensions are numbered in the order they compound. A score of 4 on Model Selection with a score of 1 on Metric Architecture means you have an excellent process for choosing among models that are evaluated on the wrong metrics. A score of 4 on Feature Pipeline with a score of 1 on Serving means you built excellent inputs for a system that cannot deliver outputs reliably.

You can temporarily work around a weak dimension, but you cannot scale safely while ignoring it.

---

*This rubric is a starting point, not a final answer. Your industry, data characteristics, and organizational context will require modifications. Use it as a conversation starter with your team, not as a compliance checklist.*

*For a deeper treatment of each dimension, see* Modern Time Series Forecasting with Python *(Packt Publishing).*
