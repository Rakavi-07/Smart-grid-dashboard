"""Streamlit dashboard for Review 1 of the Smart Grid project."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import data_utils as du
from utils import model_utils as mu
from utils import theme

st.set_page_config(page_title="Smart Grid Anomaly Detection", page_icon="⚡", layout="wide")
theme.inject_css()

PAGES = ["Home / Overview", "Consumer Analysis", "Model Results", "Methodology"]
with st.sidebar:
    st.markdown(f"<div style='color:{theme.NAVY};font-weight:800;font-size:1.05rem;'>⚡ Project Navigation</div>", unsafe_allow_html=True)
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption("Review 1 • Unit 1 + Unit 2")


df, is_real = du.get_dataset()
status = mu.model_status()

# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
if page == "Home / Overview":
    theme.header("Smart Grid Electricity Theft & Consumption Anomaly Detection",
                 "Using Bayesian Machine Learning — Review 1")
    if not is_real:
        theme.demo_mode_badge()
        st.caption("Demo data are illustrative. Reported project metrics below come from the actual SGCC analysis.")

    st.info("The system identifies consumption patterns that are statistically unusual. A positive classification is **Potentially Suspicious / Requires Investigation**, not proof of theft.", icon="ℹ️")

    st.markdown("#### Review 1 Pipeline")
    pipeline = [
        "SGCC Data", "Cleaning", "Feature Engineering", "Probability",
        "Polynomial Fit", "Bayesian Regression", "Uncertainty", "Anomaly Score", "SVM"
    ]
    cols = st.columns(len(pipeline))
    for c, step in zip(cols, pipeline):
        with c:
            st.markdown(f"<div class='sg-card' style='text-align:center;min-height:62px;display:flex;align-items:center;justify-content:center;font-size:0.78rem;'>{step}</div>", unsafe_allow_html=True)

    st.markdown("#### Dataset Summary")
    s = mu.DATASET_STATS
    a, b, c, d = st.columns(4)
    a.metric("Original consumers", f"{s['consumers_original']:,}")
    b.metric("Original columns", f"{s['columns_original']:,}")
    c.metric("After cleaning", f"{s['consumers_after_cleaning']:,}")
    d.metric("Modeling consumers", f"{s['consumers_modeling']:,}")

    st.markdown("#### Model Readiness")
    st.write({"Bayesian model loaded": status["bayesian"], "SVM model loaded": status["svm"], "Dashboard mode": "REAL DATA" if is_real else "DEMO DATA"})

# ---------------------------------------------------------------------------
# Consumer Analysis
# ---------------------------------------------------------------------------
elif page == "Consumer Analysis":
    theme.header("Consumer Analysis", "Evaluation view using the same Review-1 30-day target setup.")
    if not is_real:
        theme.demo_mode_badge()

    ids = du.list_consumer_ids(df)
    default_id = ids[0] if ids else ""
    cons_no = st.text_input("Consumer ID (CONS_NO)", value=default_id)

    series = du.get_consumer_series(df, cons_no) if cons_no else None
    if series is None:
        st.error("Consumer ID not found in the loaded dataset.")
    else:
        features, poly, target_mean = mu.build_model_frame(series)
        bayes_model = mu.load_bayesian_model()
        svm_model = mu.load_svm_model()
        scaler = mu.load_scaler()
        expected, pred_std, pred_log = mu.predict_expected_consumption(features, bayes_model)
        anomaly_score, residual = mu.compute_anomaly_score(target_mean, pred_log, pred_std)
        label, method_note = mu.classify_consumer(features, poly, anomaly_score, residual, pred_std, svm_model, scaler)

        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### Consumer `{cons_no}`")
            st.caption("Target = final 30 calendar days; historical features are computed from the preceding windows, matching Review 1.")
        with c2:
            theme.classification_banner(label)
        
        cards = st.columns(7)
        cards[0].metric("7-day mean", f"{features['mean_7']:.2f}")
        cards[1].metric("30-day mean", f"{features['mean_30']:.2f}")
        cards[2].metric("90-day mean", f"{features['mean_90']:.2f}")
        cards[3].metric("Expected target", f"{expected:.2f}")
        cards[4].metric("Uncertainty", f"{pred_std:.2f}", "log")
        cards[5].metric("Actual target", f"{target_mean:.2f}")
        cards[6].metric("Anomaly score", f"{anomaly_score:.2f}")

        target = series.sort_index().iloc[-30:]
        recent = series.sort_index().iloc[-60:-30]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent.index, y=recent.values, mode="lines+markers", name="Recent 30-day history"))
        fig.add_trace(go.Scatter(x=target.index, y=target.values, mode="lines+markers", name="Target 30-day actual"))
        fig.add_hline(y=expected, line_dash="dash", annotation_text="Bayesian expected target")
        fig.update_layout(height=390, plot_bgcolor="white", paper_bgcolor="white", xaxis_title="Date", yaxis_title="Consumption")
        st.plotly_chart(fig, use_container_width=True)

        pcols = st.columns(2)
        with pcols[0]:
            st.markdown("#### Polynomial Trend — Recent 30 Days")
            recent30 = series.sort_index().iloc[-60:-30]
            vals = recent30.to_numpy(dtype=float)
            x = np.arange(len(vals), dtype=float)
            mask = ~np.isnan(vals)
            figp = go.Figure()
            figp.add_trace(go.Scatter(x=x, y=vals, mode="markers+lines", name="Actual"))
            if mask.sum() >= 10:
                coeff = np.polyfit(x[mask], vals[mask], 2)
                polyline = np.poly1d(coeff)(x)
                figp.add_trace(go.Scatter(x=x, y=polyline, mode="lines", name="Quadratic fit"))
            figp.update_layout(height=330, plot_bgcolor="white", paper_bgcolor="white", xaxis_title="Day", yaxis_title="Consumption")
            st.plotly_chart(figp, use_container_width=True)
            st.dataframe(pd.DataFrame({"Polynomial feature": list(poly.keys()), "Value": list(poly.values())}), use_container_width=True, hide_index=True)
        with pcols[1]:
            st.markdown("#### Target Window")
            target_table = pd.DataFrame({"Date": target.index.strftime("%Y-%m-%d"), "Actual": target.values.round(3)})
            target_table["Expected"] = round(expected, 3)
            target_table["Residual (approx.)"] = (target_table["Actual"] - expected).round(3)
            st.dataframe(target_table.sort_values("Date", ascending=False), use_container_width=True, hide_index=True, height=330)

        st.caption(f"Classification method: {method_note}. Ground-truth FLAG is not used as an inference feature.")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
elif page == "Model Results":
    theme.header("Model Results", "Reported Review-1 results from the SGCC analysis.")
    st.markdown("### Bayesian Linear Regression (log1p target)")
    b = mu.BAYESIAN_METRICS
    cols = st.columns(4)
    cols[0].metric("MAE", f"{b['MAE']:.4f}")
    cols[1].metric("RMSE", f"{b['RMSE']:.4f}")
    cols[2].metric("R²", f"{b['R2']:.4f}")
    cols[3].metric("Mean uncertainty", f"{b['mean_uncertainty']:.4f}")
    st.caption("Metrics are the reported log-scale test metrics from the notebook.")

    st.markdown("### Anomaly Score")
    a = mu.ANOMALY_STATS
    stat_df = pd.DataFrame({
        "Statistic": ["Mean", "Median", "95th percentile"],
        "FLAG = 0": [a["FLAG0"]["mean"], a["FLAG0"]["median"], a["FLAG0"]["p95"]],
        "FLAG = 1": [a["FLAG1"]["mean"], a["FLAG1"]["median"], a["FLAG1"]["p95"]],
    })
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=stat_df["Statistic"], y=stat_df["FLAG = 0"], name="FLAG = 0"))
    fig.add_trace(go.Bar(x=stat_df["Statistic"], y=stat_df["FLAG = 1"], name="FLAG = 1"))
    fig.update_layout(barmode="group", height=320, plot_bgcolor="white", paper_bgcolor="white", yaxis_title="Anomaly score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Probability Analysis")
    p = mu.PROBABILITY_ANALYSIS
    prob_df = pd.DataFrame({
        "Quantity": ["Threshold", "P(FLAG=1)", "P(FLAG=0)", "P(Anomaly | FLAG=1)", "P(Anomaly | FLAG=0)", "P(Anomaly)", "P(FLAG=1 | Anomaly)"],
        "Value": [p["threshold"], p["P_FLAG1"], p["P_FLAG0"], p["P_anomaly_given_FLAG1"], p["P_anomaly_given_FLAG0"], p["P_anomaly"], p["P_FLAG1_given_anomaly"]],
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.markdown("### SVM")
    sm = mu.SVM_BASELINE_METRICS
    st.dataframe(pd.DataFrame({"Metric": list(sm.keys()), "Value": list(sm.values())}), use_container_width=True, hide_index=True)
    cm = mu.SVM_BASELINE_CONFUSION
    st.write("Baseline confusion matrix — rows = actual, columns = predicted")
    st.dataframe(pd.DataFrame(cm, index=["FLAG 0", "FLAG 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)
    if mu.SVM_FINAL_METRICS is None:
        st.info("Final polynomial-feature SVM metrics are intentionally left blank until the final evaluation is completed.")

# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------
elif page == "Methodology":
    theme.header("Methodology", "Review 1 implementation details.")
    st.markdown("### 1. Preprocessing")
    st.markdown("- Removed 10/03/2014 (99.96% missing) and 03/19/2014 (50.75% missing).\n- Replaced confirmed erroneous 800003.32 with NaN.\n- Retained consumers with at least 300 valid readings.\n- Modeling target = final 30 calendar days (10/02/2016–10/31/2016).")
    st.markdown("### 2. Unit 1 — Probability")
    st.latex(r"P(F=1\mid A)=\frac{P(A\mid F=1)P(F=1)}{P(A)}")
    st.write("The high-anomaly event uses the 95th percentile of FLAG=0 anomaly scores as the reported threshold.")
    st.markdown("### 3. Unit 1 — Polynomial Curve Fitting")
    st.latex(r"y=a_2x^2+a_1x+a_0")
    st.write("Quadratic fitting over the recent 30-day history produces RMSE, end slope, and end residual features.")
    st.markdown("### 4. Unit 2 — Bayesian Linear Regression")
    st.write("The model predicts log1p(target_mean_30) and returns predictive uncertainty. The original consumption estimate is recovered with expm1().")
    st.markdown("### 5. Anomaly Score")
    st.latex(r"z=\frac{y_{actual}-y_{predicted}}{\sigma_{pred}},\qquad AnomalyScore=|z|")
    st.markdown("### 6. Unit 2 — SVM")
    st.write("The SVM combines behavioral features, Bayesian anomaly features, and polynomial features when the final model artifact is available.")
    st.warning("A positive label means Potentially Suspicious / Requires Investigation. It is not proof of electricity theft.")
