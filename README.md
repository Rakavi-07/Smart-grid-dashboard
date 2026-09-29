# Smart Grid Electricity Theft & Consumption Anomaly Detection — Dashboard

A simple Streamlit dashboard for the academic project *"Smart Grid
Electricity Theft & Consumption Anomaly Detection Using Bayesian Machine
Learning."* Built as a clean, minimal demo suitable for a faculty project
review — not a commercial-style product.

The system flags consumers as **Normal** or **Potentially Suspicious**
based on deviation from Bayesian-predicted expected consumption. It never
claims to prove theft.

## Project structure

```
smart-grid-dashboard/
├── app.py                   # main Streamlit app (all 4 pages)
├── requirements.txt
├── README.md                 # this file
├── utils/
│   ├── theme.py               # colors, CSS, small UI helpers
│   ├── data_utils.py           # dataset loading + demo data generator
│   └── model_utils.py          # feature engineering, model loading, scoring
├── data/
│   └── README.md               # where to put cleaned_data.csv
└── models/
    └── README.md               # where to put the .joblib model files
```

## Setup

1. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional, for REAL mode) Add your data and models:
   - Place your cleaned dataset at `data/cleaned_data.csv` — see
     `data/README.md` for the expected format.
   - Place your trained model files in `models/` — see
     `models/README.md` for exact filenames.

   If these are not present, the dashboard runs in **DEMO MODE**
   automatically, using a small synthetic dataset and a transparent
   fallback scoring rule, so it's fully explorable without the real
   (large, non-public) SGCC dataset.

4. Run the dashboard:

   ```bash
   streamlit run app.py
   ```

   Then open the URL Streamlit prints (typically `http://localhost:8501`).

## Pages

- **Home / Overview** — project purpose, pipeline, dataset summary.
- **Consumer Analysis** — look up a `CONS_NO`, see its consumption
  history, rolling means, expected consumption, uncertainty, anomaly
  score, and classification.
- **Model Results** — Bayesian regression metrics, anomaly-score
  comparison by ground-truth flag, probability analysis, and SVM
  metrics (baseline + a placeholder for the final polynomial-feature
  SVM, which is still being evaluated).
- **Methodology** — preprocessing, feature list, polynomial fitting,
  Bayesian model, anomaly formula, and SVM classification, plus the
  framing note that flags are *"Potentially Suspicious"*, never
  *"Theft Confirmed."*

## Notes

- The app **never retrains models on startup** — it only loads
  pre-trained `.joblib` files via `joblib.load(...)`.
- All Bayesian/SVM performance numbers on the Model Results page are
  fixed constants taken directly from the project write-up
  (`utils/model_utils.py`) — they are not recomputed or fabricated by
  the dashboard.
- The **final SVM (with polynomial features)** metrics are left as an
  explicit placeholder (`mu.SVM_FINAL_METRICS = None`) until that
  evaluation is complete — fill it in in `utils/model_utils.py` once
  you have real numbers.
