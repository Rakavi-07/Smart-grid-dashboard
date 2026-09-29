"""
data_utils.py
-------------
Handles getting consumer time-series data into the app.

Two modes:
1. REAL MODE   -> a cleaned SGCC-style CSV is present at data/cleaned_data.csv
                  (columns: CONS_NO, FLAG, then one column per date).
2. DEMO MODE   -> no dataset found, so we synthesize a small set of
                  plausible consumers so the dashboard is fully explorable
                  without the real (large, non-public) dataset.

Nothing here fabricates the *scientific results* reported in Model Results —
those are hard-coded constants from the project write-up (see model_utils.py).
This module only ever fabricates *illustrative per-consumer daily readings*
for demo mode, clearly labeled as such in the UI.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REAL_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")

N_DEMO_DAYS = 90
DEMO_SEED = 42


def real_data_available() -> bool:
    return os.path.isfile(REAL_DATA_PATH)


@st.cache_data(show_spinner=False)
def load_real_dataset() -> pd.DataFrame:
    """Expected format: CONS_NO, FLAG, <date columns...>.
    Place your cleaned SGCC-derived CSV at data/cleaned_data.csv to use this.
    """
    df = pd.read_csv(REAL_DATA_PATH)
    return df


@st.cache_data(show_spinner=False)
def generate_demo_dataset(n_consumers: int = 12) -> pd.DataFrame:
    """Synthesize a small illustrative dataset with the same shape as the
    real one (CONS_NO, FLAG, date columns) purely so the dashboard has
    something to display in demo mode. Values are NOT drawn from or
    intended to resemble the real SGCC statistics reported elsewhere in
    this app (those figures are fixed constants from the actual project).
    """
    rng = np.random.default_rng(DEMO_SEED)
    dates = pd.date_range(end=pd.Timestamp("2016-10-31"), periods=N_DEMO_DAYS, freq="D")
    rows = []
    for i in range(n_consumers):
        cons_no = f"DEMO{100001 + i}"
        is_suspicious = i % 4 == 0  # roughly 25% flagged, just for demo variety
        base = rng.uniform(6, 14)
        series = base + rng.normal(0, 1.1, size=N_DEMO_DAYS)
        # weekly seasonality
        series += 1.5 * np.sin(np.arange(N_DEMO_DAYS) * (2 * np.pi / 7))
        if is_suspicious:
            # inject a gradual under-reporting drift + occasional spikes,
            # illustrative only
            drift = np.linspace(0, -3.0, N_DEMO_DAYS)
            series = series + drift
            spike_days = rng.choice(N_DEMO_DAYS, size=2, replace=False)
            series[spike_days] += rng.uniform(6, 10, size=2)
        series = np.clip(series, 0.2, None)
        row = {"CONS_NO": cons_no, "FLAG": int(is_suspicious)}
        for d, v in zip(dates, series):
            row[d.strftime("%Y-%m-%d")] = round(float(v), 2)
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def get_dataset() -> tuple[pd.DataFrame, bool]:
    """Returns (dataframe, is_real). Prefers the real dataset if present."""
    if real_data_available():
        return load_real_dataset(), True
    return generate_demo_dataset(), False


def list_consumer_ids(df: pd.DataFrame) -> list[str]:
    return df["CONS_NO"].astype(str).tolist()


def get_date_columns(df: pd.DataFrame) -> list[str]:
    non_date_cols = {"CONS_NO", "FLAG"}
    return [c for c in df.columns if c not in non_date_cols]


def get_consumer_series(df: pd.DataFrame, cons_no: str) -> pd.Series | None:
    """Returns a pandas Series indexed by date (as Timestamp) for one consumer,
    or None if the consumer isn't found."""
    match = df[df["CONS_NO"].astype(str) == str(cons_no)]
    if match.empty:
        return None
    date_cols = get_date_columns(df)
    row = match.iloc[0]
    series = row[date_cols].astype(float)
    series.index = pd.to_datetime(series.index, errors="coerce")
    series = series.sort_index()
    return series


def get_consumer_flag(df: pd.DataFrame, cons_no: str):
    match = df[df["CONS_NO"].astype(str) == str(cons_no)]
    if match.empty:
        return None
    return int(match.iloc[0]["FLAG"]) if "FLAG" in match.columns else None
