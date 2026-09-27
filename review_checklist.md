# Dashboard review checklist

## Passed by static inspection
- Python files compile successfully.
- No final SVM metrics are fabricated.
- `FLAG` is not used as a model input.
- Presentation wording uses "Potentially Suspicious" rather than "Theft Confirmed".
- Bayesian target inversion uses `expm1()` for a `log1p()` target.
- Dashboard anomaly scoring uses the 30-day target mean, not a single last-day reading.
- Review-1 feature windows preserve missingness rather than collapsing the series with `dropna()`.
- Same-period-last-year feature is date-based rather than position-based.
- Polynomial features are computed explicitly for the SVM/dashboard path.
- The model-results anomaly visualization uses exact reported summary statistics instead of an invented density curve.

## Not verified in this environment
- Live Streamlit rendering: the execution environment does not have Streamlit installed and external package installation failed due unavailable network/DNS.
- Real model artifact loading: the uploaded ZIP does not contain `.joblib` files.
- Final polynomial-feature SVM metrics: not present in the ZIP and therefore intentionally left blank.

## Before presentation
1. Put your actual cleaned/sample dataset in `data/cleaned_data.csv`.
2. Put your actual Bayesian model and SVM artifacts in `models/`.
3. Paste the final SVM metrics into `utils/model_utils.py` after you finish that evaluation.
4. Run `streamlit run app.py` locally and verify one real consumer ID end-to-end.
