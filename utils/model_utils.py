"""
model_utils.py
--------------
- Loads pre-trained models from models/*.joblib if present (real mode).
- Falls back to a lightweight, clearly-labeled DEMO estimator if the
  trained models aren't found, so the dashboard still runs end-to-end.
- Holds the fixed, reported project metrics as constants (these come
  directly from the project write-up and are never recomputed or
  invented here).

Expected files in models/ for REAL mode:
    models/bayesian_model.joblib   -> fitted BayesianRidge (or similar) regressor
    models/svm_model.joblib        -> fitted SVM classifier (final, poly-feature version)
    models/scaler.joblib           -> fitted feature scaler used before the SVM (optional)

If any of these are missing, that specific piece falls back to the DEMO
estimator described below, and the UI clearly flags it as such.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

BAYESIAN_MODEL_PATH = os.path.join(MODELS_DIR, "bayesian_model.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

# ---------------------------------------------------------------------------
# FIXED, REPORTED PROJECT RESULTS (from the project write-up — do not alter
# or recompute; these are the ground-truth numbers for the Model Results page)
# ---------------------------------------------------------------------------
DATASET_STATS = {
    "consumers_original": 42372,
    "columns_original": 1036,
    "daily_columns": 1034,
    "flag0_count": 38757,
    "flag1_count": 3615,
    "consumers_after_cleaning": 40185,
    "consumers_modeling": 40003,
}

BAYESIAN_METRICS = {
    "MAE": 0.6510,
    "RMSE": 0.8329,
    "R2": 0.3802,
    "mean_uncertainty": 0.8231,
}

ANOMALY_STATS = {
    "FLAG0": {"mean": 0.772, "median": 0.676, "p95": 1.750},
    "FLAG1": {"mean": 0.990, "median": 0.863, "p95": 2.014},
}

PROBABILITY_ANALYSIS = {
    "threshold": 1.7501,
    "P_FLAG1": 0.0806,
    "P_FLAG0": 0.9194,
    "P_anomaly_given_FLAG1": 0.1085,
    "P_anomaly_given_FLAG0": 0.0500,
    "P_anomaly": 0.0547,
    "P_FLAG1_given_anomaly": 0.1598,
}

SVM_BASELINE_METRICS = {
    "Accuracy": 0.6972,
    "Precision": 0.1475,
    "Recall": 0.5767,
    "F1": 0.2349,
    "ROC_AUC": 0.6964,
}

# Final SVM (with polynomial features) is still being evaluated — do not
# invent numbers. Keep as None; the UI renders an explicit placeholder.
SVM_FINAL_METRICS = None

FEATURE_NAMES = [
    "mean_7", "mean_30", "std_30", "mean_90", "std_90", "min_30", "max_30",
    "missing_ratio_30", "missing_ratio_90", "recent_change", "ratio_7_30",
    "ratio_30_90", "change_30_90", "relative_change_30_90", "cv_30",
    "max_to_mean_30", "mean_same_period_last_year",
]

ANOMALY_THRESHOLD = PROBABILITY_ANALYSIS["threshold"]


# ---------------------------------------------------------------------------
# Model availability
# ---------------------------------------------------------------------------
def bayesian_model_available() -> bool:
    return os.path.isfile(BAYESIAN_MODEL_PATH)


def svm_model_available() -> bool:
    return os.path.isfile(SVM_MODEL_PATH)


def load_bayesian_model():
    return joblib.load(BAYESIAN_MODEL_PATH) if bayesian_model_available() else None


def load_svm_model():
    return joblib.load(SVM_MODEL_PATH) if svm_model_available() else None


def load_scaler():
    return joblib.load(SCALER_PATH) if os.path.isfile(SCALER_PATH) else None


# ---------------------------------------------------------------------------
# Feature engineering (matches the documented feature list)
# ---------------------------------------------------------------------------
def compute_features(series: pd.Series) -> dict:
    """series: daily consumption indexed by date, most recent last.
    Returns a dict of the documented behavioral features computed on the
    available history (using whatever tail of `series` is available)."""
    s = series.dropna()
    if s.empty:
        return {name: np.nan for name in FEATURE_NAMES}

    last_7 = s.iloc[-7:]
    last_30 = s.iloc[-30:]
    last_90 = s.iloc[-90:]

    mean_7 = last_7.mean()
    mean_30 = last_30.mean()
    std_30 = last_30.std(ddof=0) if len(last_30) > 1 else 0.0
    mean_90 = last_90.mean()
    std_90 = last_90.std(ddof=0) if len(last_90) > 1 else 0.0
    min_30 = last_30.min()
    max_30 = last_30.max()

    missing_ratio_30 = float(series.iloc[-30:].isna().mean()) if len(series) >= 1 else 0.0
    missing_ratio_90 = float(series.iloc[-90:].isna().mean()) if len(series) >= 1 else 0.0

    recent_change = float(s.iloc[-1] - s.iloc[-2]) if len(s) >= 2 else 0.0
    ratio_7_30 = float(mean_7 / mean_30) if mean_30 else np.nan
    ratio_30_90 = float(mean_30 / mean_90) if mean_90 else np.nan
    change_30_90 = float(mean_30 - mean_90)
    relative_change_30_90 = float((mean_30 - mean_90) / mean_90) if mean_90 else np.nan
    cv_30 = float(std_30 / mean_30) if mean_30 else np.nan
    max_to_mean_30 = float(max_30 / mean_30) if mean_30 else np.nan

    # same calendar period a year earlier, if that much history exists
    if len(s) >= 365 + 7:
        mean_same_period_last_year = float(s.iloc[-(365 + 7):-365].mean())
    else:
        mean_same_period_last_year = float(mean_30)  # fallback when history is short

    return {
        "mean_7": mean_7, "mean_30": mean_30, "std_30": std_30,
        "mean_90": mean_90, "std_90": std_90, "min_30": min_30, "max_30": max_30,
        "missing_ratio_30": missing_ratio_30, "missing_ratio_90": missing_ratio_90,
        "recent_change": recent_change, "ratio_7_30": ratio_7_30,
        "ratio_30_90": ratio_30_90, "change_30_90": change_30_90,
        "relative_change_30_90": relative_change_30_90, "cv_30": cv_30,
        "max_to_mean_30": max_to_mean_30,
        "mean_same_period_last_year": mean_same_period_last_year,
    }


# ---------------------------------------------------------------------------
# Prediction + anomaly score + classification
# ---------------------------------------------------------------------------
def predict_expected_consumption(features: dict, bayesian_model=None):
    """Returns (expected_consumption, prediction_std) in ORIGINAL (kWh) units.

    REAL mode: uses the loaded Bayesian model's .predict() (assumed to
    return (mean_log, std_log) or a mean with the model exposing
    `.predict(X, return_std=True)` as scikit-learn's BayesianRidge does).

    DEMO mode (no model file): uses a simple, transparent heuristic —
    predicted_log = log(mean_30), and prediction_std = the reported mean
    uncertainty constant from the actual project (0.8231 on the log
    scale) — i.e. we reuse the *real* reported uncertainty rather than
    inventing a new number.
    """
    mean_30 = features.get("mean_30", np.nan)
    if bayesian_model is not None:
        X = np.array([[features[f] for f in FEATURE_NAMES]])
        try:
            mean_log, std_log = bayesian_model.predict(X, return_std=True)
            mean_log, std_log = float(mean_log[0]), float(std_log[0])
        except TypeError:
            mean_log = float(bayesian_model.predict(X)[0])
            std_log = BAYESIAN_METRICS["mean_uncertainty"]
    else:
        safe_mean_30 = mean_30 if (mean_30 and mean_30 > 0) else 1.0
        mean_log = float(np.log(safe_mean_30))
        std_log = BAYESIAN_METRICS["mean_uncertainty"]

    expected_consumption = float(np.exp(mean_log))
    return expected_consumption, std_log, mean_log


def compute_anomaly_score(actual_value: float, predicted_log: float, prediction_std: float) -> float:
    """z = (actual_log - predicted_log) / prediction_std ; anomaly_score = |z|"""
    safe_actual = actual_value if actual_value and actual_value > 0 else 1e-6
    actual_log = np.log(safe_actual)
    if not prediction_std:
        return 0.0
    z = (actual_log - predicted_log) / prediction_std
    return float(abs(z))


def classify_consumer(features: dict, anomaly_score: float, svm_model=None, scaler=None):
    """Returns (label, method_note).

    REAL mode: uses the loaded, trained SVM on the feature vector
    (+ polynomial features, if the pipeline stored in the joblib file
    includes them).

    DEMO mode (no model file): falls back to the documented anomaly
    threshold (95th percentile of normal-consumer scores = 1.7501) —
    this is the same threshold reported in the project's probability
    analysis, not an invented cutoff.
    """
    if svm_model is not None:
        X = np.array([[features[f] for f in FEATURE_NAMES]])
        if scaler is not None:
            X = scaler.transform(X)
        pred = svm_model.predict(X)[0]
        label = "Potentially Suspicious" if int(pred) == 1 else "Normal"
        return label, "trained SVM model"
    else:
        label = "Potentially Suspicious" if anomaly_score > ANOMALY_THRESHOLD else "Normal"
        return label, f"demo rule: anomaly score vs. reported threshold ({ANOMALY_THRESHOLD})"
