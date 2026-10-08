# Early-Warning Targeting of At-Risk Students

Ranking 100,000 students by their risk of failing an exam, so a school with limited support capacity can help the students who need it most. Compares Logistic Regression, Random Forest and XGBoost against a simple rule-based baseline, with a leakage audit, feature ablations, a cost-based flagging policy and a subgroup fairness audit.

> **Note on the data:** this is a practice dataset (`StudentExamPerformance_practice.csv`) with some synthetic-looking traits, such as a clean 50-point pass cutoff. The project is about methodology, and I make no claims about real-world impact.

## The problem

A school can only give intensive support (tutoring, counselling) to a fraction of its students. So the useful question is not "who will fail?" but **"if we can support the top 10-20% of students, how many of the students who would fail do we reach?"**

Everything is evaluated with that in mind: ranking metrics (PR-AUC, recall@k, precision@k) rather than accuracy, and a rule a teacher might use ("lowest previous exam score") as the baseline to beat.

## Headline results (held-out test set, 20,000 students)

| Model | PR-AUC | ROC-AUC | recall@20% | precision@10% |
|---|---|---|---|---|
| Random ranking | 0.228 | 0.502 | 0.210 | 0.232 |
| Rule: lowest previous exam score | 0.469 | 0.745 | 0.424 | 0.551 |
| Logistic Regression | 0.720 | 0.890 | 0.600 | 0.819 |
| XGBoost (default) | 0.740 | 0.899 | 0.613 | 0.838 |
| XGBoost (tuned) | 0.742 | 0.900 | 0.618 | 0.841 |

Supporting the top 20% of students ranked by the tuned XGBoost reaches **61.8% of all failing students**, against 42.4% for the rule and 21.0% for random selection.

![Recall at different support capacities](reports/figures/capacity_curve.png)

## Key findings

1. **Models beat the rule by a wide margin, but not each other by much.** XGBoost led Logistic Regression by about 0.02 PR-AUC in every cross-validation fold; Random Forest did not beat Logistic Regression; hyperparameter tuning added about 0.002. The signal appears mostly additive and roughly linear.
2. **Behaviour matters, demographics do not.** Dropping each feature group in turn from the full model (leave-one-group-out):

   | Group removed | PR-AUC drop |
   |---|---|
   | Study behaviour | 0.183 |
   | Prior performance | 0.160 |
   | Exam difficulty | 0.076 |
   | Wellbeing (sleep, stress, screen time...) | 0.032 |
   | Demographics and access | 0.000 |

3. **Leakage was found and removed.** `pass_status` is exactly `exam_score >= 50`, grades are bands of the score, and `questions_correct` correlates 0.975 with the score. Those columns (and IDs) were dropped before modelling.
4. **The model is well calibrated.** Brier score 0.102 vs 0.175 for a base-rate-only model. Cost-optimal thresholds chosen on training data matched the theoretical value 1 / (1 + cost ratio), so scores can be read as probabilities.
5. **Fairness: one clear gap.** At a fixed 20% capacity, recall was similar across school type and urban/rural, and gender gaps were within sampling noise. Across **family income** there was a 13-point recall gap (High-income failing students caught less often: 0.528 vs 0.659 for Low-income). Per-group mean predictions matched observed failure rates, so the gap is consistent with differing base rates under one shared cutoff, not systematic under-prediction. Group-specific thresholds chosen on training data narrowed it to about 8 points with almost no change in overall recall or capacity, at the cost of shifting support from low-income to higher-income failing students.

## Method

- **Target:** `Fail` (22.6% of students). Features: 37 pre-exam variables after removing leakage and IDs.
- **Split:** 80/20 stratified. The test set was used once, after all choices were fixed.
- **Preprocessing (inside pipelines, fit per CV fold):** median imputation with missing-value indicators, a "Missing" category for categoricals, one-hot encoding, scaling for Logistic Regression only.
- **Evaluation:** 5-fold out-of-fold predictions for comparison; PR-AUC, ROC-AUC, recall@k and precision@k.
- **Ablations:** cumulative feature-group ablation and leave-one-group-out, to check that conclusions do not depend on the order features are added.
- **Decision layer:** thresholds chosen by expected cost for several miss-to-false-alarm cost ratios (1:1 to 10:1), selected on training data and applied unchanged to test.
- **Interpretation:** SHAP for XGBoost compared with Logistic Regression coefficients. They agree on the main drivers; they differ on `previous_gpa`, which overlaps with previous exam score.

## Limitations

- Practice dataset; results will not transfer to a real school without validation.
- **`exam_difficulty` is the single most influential feature** (about 0.08 PR-AUC) and may not be known before an exam in practice. Without it, Logistic Regression drops from 0.726 to 0.648 PR-AUC in cross-validation, still well above the 0.460 rule baseline.
- Cost ratios are illustrative assumptions, not estimates from real costs, and the cost model assumes support always helps a flagged student.
- Fairness analysis uses one operating point, reports no confidence intervals, compares many groups (some gaps could arise by chance) and does not look at intersections of groups.
- SHAP and coefficients describe what the model relies on, not what causes failure.
- Group-specific thresholds use a sensitive attribute in the decision rule, which can be legally or ethically restricted. It is shown as an option to discuss, not a recommendation.

## Repository structure

```
at-risk-students/
├── README.md
├── requirements.txt
├── data/                      # place the CSV here (not committed)
├── notebooks/
│   └── at_risk_students.ipynb # full analysis, run top to bottom
└── reports/
    └── figures/
        └── capacity_curve.png
```

## How to reproduce

```bash
git clone https://github.com/<your-username>/at-risk-students.git
cd at-risk-students
pip install -r requirements.txt
# put StudentExamPerformance_practice.csv in data/
jupyter notebook notebooks/at_risk_students.ipynb
```

The notebook also runs in Google Colab: upload the CSV when the first cell asks for it. A fixed random seed (42) is used throughout.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn.

## What I would do next

- Add bootstrap confidence intervals to the subgroup metrics.
- Test robustness under distribution shift (train on one school type, test on another).
- Per-group calibration curves rather than group averages.
- Validate on real data with real intervention costs.
