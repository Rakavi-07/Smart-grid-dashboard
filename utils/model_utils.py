"""Project feature engineering, model loading, scoring, and fixed reported metrics."""
from __future__ import annotations

import inspect
import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
BAYESIAN_MODEL_PATH = os.path.join(MODELS_DIR, "bayesian_model.joblib")
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")

DATASET_STATS = {
    "consumers_original": 42372,
    "columns_original": 1036,
    "daily_columns": 1034,
    "flag0_count": 38757,
    "flag1_count": 3615,
    "consumers_after_cleaning": 40185,
    "consumers_modeling": 40003,
}

BAYESIAN_METRICS = {"MAE": 0.6510, "RMSE": 0.8329, "R2": 0.3802, "mean_uncertainty": 0.8231}
ANOMALY_STATS = {"FLAG0": {"mean": 0.772, "median": 0.676, "p95": 1.750},
                 "FLAG1": {"mean": 0.990, "median": 0.863, "p95": 2.014}}
PROBABILITY_ANALYSIS = {"threshold": 1.7501, "P_FLAG1": 0.0806, "P_FLAG0": 0.9194,
                        "P_anomaly_given_FLAG1": 0.1085, "P_anomaly_given_FLAG0": 0.0500,
                        "P_anomaly": 0.0547, "P_FLAG1_given_anomaly": 0.1598}
SVM_BASELINE_METRICS = {"Accuracy": 0.6972, "Precision": 0.1475, "Recall": 0.5767,
                        "F1": 0.2349, "ROC_AUC": 0.6964}
SVM_BASELINE_CONFUSION = np.array([[5206, 2150], [273, 372]])
SVM_FINAL_METRICS = None  # Filled only when the user's final 23-feature SVM is evaluated.

FEATURE_NAMES = [
    "mean_7", "mean_30", "std_30", "mean_90", "std_90", "min_30", "max_30",
    "missing_ratio_30", "missing_ratio_90", "recent_change", "ratio_7_30", "ratio_30_90",
    "change_30_90", "relative_change_30_90", "cv_30", "max_to_mean_30", "mean_same_period_last_year",
]
POLY_FEATURE_NAMES = ["poly_rmse_30", "poly_end_slope_30", "poly_end_residual_30"]
BASELINE_SVM_FEATURE_NAMES = FEATURE_NAMES + ["anomaly_score", "residual", "prediction_std"]
FINAL_SVM_FEATURE_NAMES = BASELINE_SVM_FEATURE_NAMES + POLY_FEATURE_NAMES
ANOMALY_THRESHOLD = PROBABILITY_ANALYSIS["threshold"]
EPS = 1e-6


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


def _sorted_windows(series: pd.Series) -> dict[str, pd.Series]:
    """Exact Review-1 layout: previous 90d, recent 30d, recent 7d, target 30d."""
    s = series.sort_index()
    if len(s) < 150:
        return {"recent_7": s.iloc[-7:], "recent_30": s.iloc[-30:], "previous_90": s.iloc[-120:-30], "target_30": s.iloc[-30:]}
    return {
        "recent_7": s.iloc[-37:-30],
        "recent_30": s.iloc[-60:-30],
        "previous_90": s.iloc[-150:-60],
        "target_30": s.iloc[-30:],
    }


def compute_features(series: pd.Series) -> dict[str, float]:
    """Compute the same 17 features used in the notebook, preserving calendar-window missingness."""
    w = _sorted_windows(series)
    recent_7, recent_30, previous_90 = w["recent_7"], w["recent_30"], w["previous_90"]

    mean_7 = float(recent_7.mean())
    mean_30 = float(recent_30.mean())
    std_30 = float(recent_30.std(ddof=1)) if recent_30.notna().sum() > 1 else 0.0
    mean_90 = float(previous_90.mean())
    std_90 = float(previous_90.std(ddof=1)) if previous_90.notna().sum() > 1 else 0.0
    min_30 = float(recent_30.min())
    max_30 = float(recent_30.max())

    missing_ratio_30 = float(recent_30.isna().mean())
    missing_ratio_90 = float(previous_90.isna().mean())
    recent_change = float(mean_7 - mean_30)
    ratio_7_30 = float(mean_7 / (mean_30 + EPS))
    ratio_30_90 = float(mean_30 / (mean_90 + EPS))
    change_30_90 = float(mean_30 - mean_90)
    relative_change_30_90 = float((mean_30 - mean_90) / (mean_90 + EPS))
    cv_30 = float(std_30 / (mean_30 + EPS))
    max_to_mean_30 = float(max_30 / (mean_30 + EPS))

    target = w["target_30"]
    if not target.empty:
        # Use the same calendar dates one year earlier; DateOffset handles leap years correctly.
        start = target.index.min() - pd.DateOffset(years=1)
        end = target.index.max() - pd.DateOffset(years=1)
        last_year = series.loc[(series.index >= start) & (series.index <= end)]
        mean_same_period_last_year = float(last_year.mean()) if last_year.notna().any() else np.nan
    else:
        mean_same_period_last_year = np.nan

    return {
        "mean_7": mean_7, "mean_30": mean_30, "std_30": std_30,
        "mean_90": mean_90, "std_90": std_90, "min_30": min_30, "max_30": max_30,
        "missing_ratio_30": missing_ratio_30, "missing_ratio_90": missing_ratio_90,
        "recent_change": recent_change, "ratio_7_30": ratio_7_30, "ratio_30_90": ratio_30_90,
        "change_30_90": change_30_90, "relative_change_30_90": relative_change_30_90,
        "cv_30": cv_30, "max_to_mean_30": max_to_mean_30,
        "mean_same_period_last_year": mean_same_period_last_year,
    }


def compute_polynomial_features(series: pd.Series) -> dict[str, float]:
    w = _sorted_windows(series)
    values = w["recent_30"].astype(float).to_numpy()
    x = np.arange(len(values), dtype=float)
    valid = ~np.isnan(values)
    if valid.sum() < 10:
        return {name: np.nan for name in POLY_FEATURE_NAMES}
    coeff = np.polyfit(x[valid], values[valid], deg=2)
    poly = np.poly1d(coeff)
    fitted = poly(x[valid])
    rmse = float(np.sqrt(np.mean((values[valid] - fitted) ** 2)))
    derivative = np.polyder(poly)
    end_slope = float(derivative(x[-1]))
    end_residual = float(values[valid][-1] - poly(x[valid][-1]))
    return {"poly_rmse_30": rmse, "poly_end_slope_30": end_slope, "poly_end_residual_30": end_residual}


def get_target_mean(series: pd.Series) -> float:
    target = _sorted_windows(series)["target_30"]
    return float(target.mean())


def build_model_frame(series: pd.Series) -> tuple[dict[str, float], dict[str, float], float]:
    features = compute_features(series)
    poly = compute_polynomial_features(series)
    target_mean = get_target_mean(series)
    return features, poly, target_mean


def _model_expected_input(model: Any) -> int | None:
    return getattr(model, "n_features_in_", None)


def predict_expected_consumption(features: dict[str, float], bayesian_model=None):
    """Return expected consumption in original units, log1p predictive std, and mean log prediction."""
    X = pd.DataFrame([[features.get(f, np.nan) for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    if bayesian_model is None:
        mean_30 = float(features.get("mean_30", 1.0))
        mean_log = float(np.log1p(max(mean_30, 0.0)))
        std_log = BAYESIAN_METRICS["mean_uncertainty"]
    else:
        try:
            mean_log_arr, std_arr = bayesian_model.predict(X, return_std=True)
            mean_log, std_log = float(mean_log_arr[0]), float(std_arr[0])
        except (TypeError, ValueError):
            mean_log = float(bayesian_model.predict(X)[0])
            std_log = BAYESIAN_METRICS["mean_uncertainty"]
    expected = float(np.expm1(mean_log))
    return expected, std_log, mean_log


def compute_anomaly_score(actual_target_mean: float, predicted_log: float, prediction_std: float) -> tuple[float, float]:
    actual_log = float(np.log1p(max(actual_target_mean, 0.0)))
    if prediction_std <= 0:
        return 0.0, actual_log - predicted_log
    residual = actual_log - predicted_log
    return float(abs(residual / prediction_std)), float(residual)


def _prepare_svm_input(features: dict[str, float], anomaly_score: float, residual: float,
                       prediction_std: float, poly_features: dict[str, float], svm_model: Any) -> pd.DataFrame:
    row = dict(features)
    row.update({"anomaly_score": anomaly_score, "residual": residual, "prediction_std": prediction_std})
    row.update(poly_features)
    n = _model_expected_input(svm_model)
    if n == 20:
        cols = BASELINE_SVM_FEATURE_NAMES
    elif n == 23:
        cols = FINAL_SVM_FEATURE_NAMES
    else:
        # Prefer an explicit feature_names_in_ when the estimator provides it.
        names = getattr(svm_model, "feature_names_in_", None)
        if names is not None:
            cols = list(names)
        else:
            cols = FINAL_SVM_FEATURE_NAMES
    return pd.DataFrame([[row.get(c, np.nan) for c in cols]], columns=cols)


def classify_consumer(features: dict[str, float], poly_features: dict[str, float],
                      anomaly_score: float, residual: float, prediction_std: float,
                      svm_model=None, scaler=None):
    if svm_model is None:
        label = "Potentially Suspicious" if anomaly_score >= ANOMALY_THRESHOLD else "Normal"
        return label, f"demo rule: anomaly score vs. threshold {ANOMALY_THRESHOLD:.4f}"

    X = _prepare_svm_input(features, anomaly_score, residual, prediction_std, poly_features, svm_model)

    # If svm_model is already a Pipeline, it is responsible for imputation/scaling.
    is_pipeline = hasattr(svm_model, "steps") or hasattr(svm_model, "named_steps")
    if not is_pipeline and scaler is not None:
        X = scaler.transform(X)
    pred = int(svm_model.predict(X)[0])
    label = "Potentially Suspicious" if pred == 1 else "Normal"
    return label, "trained SVM model"


def model_status() -> dict[str, bool]:
    return {"bayesian": bayesian_model_available(), "svm": svm_model_available()}
