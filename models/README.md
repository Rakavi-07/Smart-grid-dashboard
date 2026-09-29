# Models folder

Place your pre-trained model files here (all optional — the app falls
back to a clearly-labeled demo estimator for any file that's missing):

    models/bayesian_model.joblib   # fitted BayesianRidge (or similar) regressor
                                    # trained on log(consumption) with the
                                    # documented behavioral features
    models/svm_model.joblib        # fitted final SVM classifier
                                    # (baseline + polynomial features)
    models/scaler.joblib           # optional: fitted feature scaler used
                                    # before the SVM

The dashboard NEVER retrains models on startup — it only loads these
files with `joblib.load(...)`. Train and save them separately, e.g.:

```python
import joblib
joblib.dump(bayesian_model, "models/bayesian_model.joblib")
joblib.dump(svm_model, "models/svm_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
```
