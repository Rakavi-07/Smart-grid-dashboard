"""Data loading and small demo-data generation helpers."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REAL_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")
N_DEMO_DAYS = 420
DEMO_SEED = 42


def real_data_available() -> bool:
    return os.path.isfile(REAL_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_real_dataset() -> pd.DataFrame:
    return pd.read_csv(REAL_DATA_PATH)


@st.cache_data(show_spinner=False)
def generate_demo_dataset(n_consumers: int = 12) -> pd.DataFrame:
    """Generate illustrative data only; it is never used for project metrics."""
    rng = np.random.default_rng(DEMO_SEED)
    dates = pd.date_range(end=pd.Timestamp("2016-10-31"), periods=N_DEMO_DAYS, freq="D")
    rows = []
    for i in range(n_consumers):
        cons_no = f"DEMO{100001 + i}"
        is_suspicious = i % 4 == 0
        base = rng.uniform(6, 14)
        series = base + rng.normal(0, 1.1, size=N_DEMO_DAYS)
        series += 1.2 * np.sin(np.arange(N_DEMO_DAYS) * (2 * np.pi / 7))
        if is_suspicious:
            drift = np.linspace(0, -2.5, N_DEMO_DAYS)
            series = series + drift
            spike_days = rng.choice(N_DEMO_DAYS, size=3, replace=False)
            series[spike_days] += rng.uniform(5, 9, size=3)
        series = np.clip(series, 0.05, None)
        # Add a few missing observations to demonstrate handling.
        missing_idx = rng.choice(N_DEMO_DAYS, size=8, replace=False)
        series[missing_idx] = np.nan
        row = {"CONS_NO": cons_no, "FLAG": int(is_suspicious)}
        for d, v in zip(dates, series):
            row[d.strftime("%Y-%m-%d")] = None if np.isnan(v) else round(float(v), 3)
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_dataset() -> tuple[pd.DataFrame, bool]:
    if real_data_available():
        return load_real_dataset(), True
    return generate_demo_dataset(), False


def list_consumer_ids(df: pd.DataFrame) -> list[str]:
    return df["CONS_NO"].astype(str).tolist()


def get_date_columns(df: pd.DataFrame) -> list[str]:
    non_date_cols = {"CONS_NO", "FLAG"}
    return [c for c in df.columns if c not in non_date_cols]


def get_consumer_series(df: pd.DataFrame, cons_no: str) -> pd.Series | None:
    match = df[df["CONS_NO"].astype(str) == str(cons_no)]
    if match.empty:
        return None
    row = match.iloc[0]
    date_cols = get_date_columns(df)
    raw = pd.to_numeric(row[date_cols], errors="coerce")
    dt = pd.to_datetime(raw.index, errors="coerce")
    valid = ~dt.isna()
    series = pd.Series(raw.to_numpy(dtype=float)[valid], index=dt[valid])
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def get_consumer_flag(df: pd.DataFrame, cons_no: str):
    match = df[df["CONS_NO"].astype(str) == str(cons_no)]
    if match.empty or "FLAG" not in match.columns:
        return None
    return int(match.iloc[0]["FLAG"])
