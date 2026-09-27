# Models

Put your real trained artifacts here.

`bayesian_model.joblib`
- Trained using the 17 Review-1 behavioral features.
- Target: `log1p(target_mean_30)`.
- A complete sklearn Pipeline (imputer + scaler + BayesianRidge) is preferred.

`svm_model.joblib`
- Baseline SVM: 20 features.
- Final Review-1 SVM: 23 features (17 behavioral + 3 Bayesian anomaly + 3 polynomial).
- A complete sklearn Pipeline is preferred.

`scaler.joblib` is optional and is only used when the saved SVM is a bare estimator rather than a Pipeline.
