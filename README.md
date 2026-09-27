# Smart Grid Electricity Theft & Consumption Anomaly Detection — Review 1 Dashboard

A Streamlit dashboard for the Review 1 academic project. It covers Unit 1 (probability analysis and polynomial curve fitting) and Unit 2 (Bayesian Linear Regression and SVM).

## Important status
- The dashboard code has been aligned to the notebook's Review-1 feature definitions.
- It uses `log1p(target_mean_30)` for Bayesian prediction, matching the model training.
- The consumer-analysis view uses the same 30-day target setup as the project.
- The SVM input supports both the 20-feature baseline model and the 23-feature final polynomial-feature model.
- The dashboard does **not** invent final SVM metrics; `SVM_FINAL_METRICS` stays blank until you paste the actual final evaluation.

## Run
```bash
python -m venv venv
venv\\Scripts\\activate      # Windows
pip install -r requirements.txt
streamlit run app.py
```

Without `data/cleaned_data.csv`, the app runs in clearly-labeled DEMO MODE with synthetic data only. The fixed Model Results numbers are the reported project metrics; they are not recomputed from demo data.

## Real-data mode
Place your cleaned SGCC-derived CSV at:
`data/cleaned_data.csv`

Required columns:
- `CONS_NO`
- optional `FLAG`
- daily date columns (any parseable date format accepted by pandas)

## Model files
Place model artifacts in `models/`:
- `bayesian_model.joblib` — your trained Bayesian pipeline
- `svm_model.joblib` — preferably the complete trained SVM pipeline; the app also supports an estimator + optional `scaler.joblib`
- `scaler.joblib` — only needed if the SVM file is not already a pipeline that scales/imputes features

Expected SVM features:
- 20 features for the baseline SVM
- 23 features for the final polynomial-feature SVM

## Presentation-safe wording
Use **Normal** / **Potentially Suspicious — Requires Investigation**. Do not write **Theft Confirmed**.
