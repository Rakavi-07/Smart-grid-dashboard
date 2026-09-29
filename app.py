"""
app.py
------
Smart Grid Electricity Theft & Consumption Anomaly Detection — Dashboard

A simple academic demo dashboard (Streamlit) for a project review. Run with:

    streamlit run app.py

See README.md for setup, and for where to place the real dataset /
trained model files. Without them, the app runs fully in DEMO MODE using
synthetic consumer data (real project metrics/statistics shown in the
Model Results page are fixed constants from the write-up, never invented
or recomputed).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import theme
from utils import data_utils as du
from utils import model_utils as mu

st.set_page_config(
    page_title="Smart Grid Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_css()

PAGES = ["Home / Overview", "Consumer Analysis", "Model Results", "Methodology"]

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<div style='color:{theme.NAVY};font-weight:800;font-size:1.05rem;"
        f"margin-bottom:0.8rem;'>⚡ Project Navigation</div>",
        unsafe_allow_html=True,
    )
    page = st.radio("Go to", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        f"<div style='color:{theme.TEXT_MUTED};font-size:0.85rem;'>"
        "<b>Project Overview</b><br>"
        "This system uses Bayesian Linear Regression and a Support Vector "
        "Machine to flag electricity consumption patterns that deviate from "
        "a consumer's expected behavior.</div>",
        unsafe_allow_html=True,
    )

df, is_real = du.get_dataset()

# ---------------------------------------------------------------------------
# PAGE 1: Home / Overview
# ---------------------------------------------------------------------------
if page == "Home / Overview":
    theme.header(
        "Smart Grid Electricity Theft & Consumption Anomaly Detection",
        "Using Bayesian Machine Learning — Academic Project Review",
    )
    if not is_real:
        theme.demo_mode_badge()

    st.markdown("#### Purpose")
    st.write(
        "The system learns electricity-consumption behavior from historical "
        "smart-meter data, predicts expected consumption using **Bayesian "
        "Linear Regression**, estimates prediction uncertainty, calculates an "
        "**anomaly score**, and uses an **SVM classifier** to flag each "
        "consumer as **Normal** or **Potentially Suspicious**."
    )
    st.info(
        "This system does **not** claim to prove electricity theft. Flagged "
        "consumers are labeled *Potentially Suspicious* / *Requires "
        "Investigation* only.",
        icon="ℹ️",
    )

    st.markdown("#### Pipeline")
    pipeline_steps = [
        "SGCC Smart Meter Dataset", "Data Cleaning", "Feature Engineering",
        "Probability Analysis", "Polynomial Curve Fitting",
        "Bayesian Linear Regression", "Prediction + Uncertainty",
        "Bayesian Anomaly Score", "SVM Classification",
    ]
    cols = st.columns(len(pipeline_steps))
    for c, step in zip(cols, pipeline_steps):
        with c:
            st.markdown(
                f"<div class='sg-card' style='text-align:center;font-size:0.78rem;"
                f"min-height:70px;display:flex;align-items:center;justify-content:center;'>"
                f"{step}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### Dataset Summary")
    stats = mu.DATASET_STATS
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.metric_card("Original Consumers", f"{stats['consumers_original']:,}")
    with c2:
        theme.metric_card("Original Columns", f"{stats['columns_original']:,}")
    with c3:
        theme.metric_card("After Cleaning", f"{stats['consumers_after_cleaning']:,}")
    with c4:
        theme.metric_card("Modeling Dataset", f"{stats['consumers_modeling']:,}")

    c5, c6 = st.columns(2)
    with c5:
        theme.metric_card("FLAG = 0 (Normal, ground truth)", f"{stats['flag0_count']:,}")
    with c6:
        theme.metric_card("FLAG = 1 (Suspicious, ground truth)", f"{stats['flag1_count']:,}")

    st.caption(
        "Final modeling target period: 10/02/2016 to 10/31/2016. "
        "Severely damaged dates (e.g. 10/03/2014 at 99.96% missing, "
        "03/19/2014 at 50.75% missing) were removed during cleaning."
    )

# ---------------------------------------------------------------------------
# PAGE 2: Consumer Analysis
# ---------------------------------------------------------------------------
elif page == "Consumer Analysis":
    theme.header("Consumer Analysis", "Enter a consumer ID to view their consumption pattern and model prediction.")
    if not is_real:
        theme.demo_mode_badge()

    ids = du.list_consumer_ids(df)
    default_id = ids[0] if ids else ""

    left, right = st.columns([3, 1])
    with left:
        cons_no = st.text_input("Consumer ID (CONS_NO)", value=default_id)
    with right:
        st.write("")
        st.write("")
        analyze = st.button("🔍 Analyze", use_container_width=True)

    if not is_real:
        with st.expander("Available demo consumer IDs"):
            st.write(", ".join(ids))

    if cons_no:
        series = du.get_consumer_series(df, cons_no)
        if series is None:
            st.error(f"Consumer ID '{cons_no}' not found in the loaded dataset.")
        else:
            features = mu.compute_features(series)
            bayes_model = mu.load_bayesian_model()
            svm_model = mu.load_svm_model()
            scaler = mu.load_scaler()

            expected, pred_std, pred_log = mu.predict_expected_consumption(features, bayes_model)
            actual_latest = float(series.dropna().iloc[-1])
            anomaly_score = mu.compute_anomaly_score(actual_latest, pred_log, pred_std)
            label, method_note = mu.classify_consumer(features, anomaly_score, svm_model, scaler)

            # ---- Metric cards row ----
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
            with m1:
                theme.metric_card("7-Day Mean", f"{features['mean_7']:.2f}", "kWh")
            with m2:
                theme.metric_card("30-Day Mean", f"{features['mean_30']:.2f}", "kWh")
            with m3:
                theme.metric_card("90-Day Mean", f"{features['mean_90']:.2f}", "kWh")
            with m4:
                theme.metric_card("Expected Consumption", f"{expected:.2f}", "kWh")
            with m5:
                theme.metric_card("Prediction Uncertainty", f"{pred_std:.2f}", "(log)")
            with m6:
                theme.metric_card("Anomaly Score", f"{anomaly_score:.2f}")
            with m7:
                theme.metric_card("SVM Classification", label)

            st.markdown("<br>", unsafe_allow_html=True)
            chart_col, banner_col = st.columns([3, 1])

            with chart_col:
                st.markdown("<div class='sg-section-title'>Consumption History</div>", unsafe_allow_html=True)
                hist = series.dropna().iloc[-90:]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist.values, mode="lines+markers", name="Actual Consumption",
                    line=dict(color=theme.ACCENT, width=2), marker=dict(size=4),
                ))
                fig.add_trace(go.Scatter(
                    x=hist.index, y=[expected] * len(hist), mode="lines", name="Expected Consumption (Bayesian)",
                    line=dict(color=theme.WARN_RED, width=1.5, dash="dash"),
                ))
                upper = [expected + pred_std * expected] * len(hist)  # illustrative band in kWh terms
                lower = [max(expected - pred_std * expected, 0)] * len(hist)
                fig.add_trace(go.Scatter(
                    x=list(hist.index) + list(hist.index[::-1]),
                    y=upper + lower[::-1],
                    fill="toself", fillcolor="rgba(46,125,209,0.12)",
                    line=dict(color="rgba(0,0,0,0)"), name="Uncertainty Range (±1 std)", showlegend=True,
                ))
                fig.update_layout(
                    height=380, margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    yaxis_title="Daily Consumption (kWh)", xaxis_title="Date",
                )
                st.plotly_chart(fig, use_container_width=True)

            with banner_col:
                theme.classification_banner(label)
                st.caption(f"Method: {method_note}")

            st.markdown("<div class='sg-section-title'>Recent Consumption Data</div>", unsafe_allow_html=True)
            recent = hist.iloc[-8:].sort_index(ascending=False)
            table = pd.DataFrame({
                "Date": recent.index.strftime("%Y-%m-%d"),
                "Actual (kWh)": recent.values.round(2),
                "Expected (kWh)": [round(expected, 2)] * len(recent),
                "Residual": (recent.values - expected).round(2),
            })
            st.dataframe(table, use_container_width=True, hide_index=True)

            with st.expander("Feature Summary (model inputs)"):
                feat_df = pd.DataFrame(
                    {"Feature": list(features.keys()), "Value": [round(v, 3) if isinstance(v, (int, float)) and not pd.isna(v) else v for v in features.values()]}
                )
                st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE 3: Model Results
# ---------------------------------------------------------------------------
elif page == "Model Results":
    theme.header("Model Results", "Reported performance of the Bayesian regression, anomaly scoring, and SVM classifier.")

    st.markdown("<div class='sg-section-title'>Bayesian Linear Regression (log scale)</div>", unsafe_allow_html=True)
    b = mu.BAYESIAN_METRICS
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        theme.metric_card("MAE", f"{b['MAE']:.4f}")
    with c2:
        theme.metric_card("RMSE", f"{b['RMSE']:.4f}")
    with c3:
        theme.metric_card("R²", f"{b['R2']:.4f}")
    with c4:
        theme.metric_card("Mean Uncertainty", f"{b['mean_uncertainty']:.4f}")

    st.markdown("<br><div class='sg-section-title'>Anomaly Score Comparison (by ground-truth FLAG)</div>", unsafe_allow_html=True)
    a = mu.ANOMALY_STATS
    stat_df = pd.DataFrame({
        "Statistic": ["Mean", "Median", "95th Percentile"],
        "FLAG = 0 (Normal)": [a["FLAG0"]["mean"], a["FLAG0"]["median"], a["FLAG0"]["p95"]],
        "FLAG = 1 (Suspicious)": [a["FLAG1"]["mean"], a["FLAG1"]["median"], a["FLAG1"]["p95"]],
    })
    st.dataframe(stat_df, use_container_width=True, hide_index=True)

    # Illustrative anomaly-score distribution built from the reported
    # mean/median/p95 (NOT raw per-consumer data, which isn't shipped with
    # this dashboard). Clearly labeled as an approximation.
    x = np.linspace(0, 5, 400)

    def approx_density(mean, p95):
        # rough log-normal-ish shape matched to the reported mean & p95, for
        # illustration only
        sigma = max((np.log(p95 + 1e-6) - np.log(mean + 1e-6)) / 1.645, 0.05)
        mu_ = np.log(mean + 1e-6) - 0.5 * sigma ** 2
        with np.errstate(divide="ignore"):
            y = np.where(x > 0, (1 / (x * sigma * np.sqrt(2 * np.pi) + 1e-9)) *
                         np.exp(-((np.log(x + 1e-9) - mu_) ** 2) / (2 * sigma ** 2)), 0)
        return y

    y0 = approx_density(a["FLAG0"]["mean"], a["FLAG0"]["p95"])
    y1 = approx_density(a["FLAG1"]["mean"], a["FLAG1"]["p95"])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x, y=y0, mode="lines", name="Normal (FLAG=0)", line=dict(color=theme.ACCENT)))
    fig2.add_trace(go.Scatter(x=x, y=y1, mode="lines", name="Suspicious (FLAG=1)", line=dict(color=theme.WARN_RED)))
    fig2.add_vline(x=mu.PROBABILITY_ANALYSIS["threshold"], line_dash="dash", line_color="black",
                    annotation_text=f"Threshold = {mu.PROBABILITY_ANALYSIS['threshold']:.3f}")
    fig2.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis_title="Anomaly Score", yaxis_title="Density (illustrative)")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Curves are illustrative approximations fitted to the reported mean/95th-percentile "
               "statistics — the underlying per-consumer scores aren't bundled with this dashboard.")

    st.markdown("<br><div class='sg-section-title'>Probability Analysis</div>", unsafe_allow_html=True)
    p = mu.PROBABILITY_ANALYSIS
    prob_df = pd.DataFrame({
        "Quantity": [
            "Anomaly Threshold (95th pct. of FLAG=0 scores)",
            "P(FLAG = 1)", "P(FLAG = 0)",
            "P(Anomaly | FLAG = 1)", "P(Anomaly | FLAG = 0)",
            "P(Anomaly)", "P(FLAG = 1 | Anomaly)",
        ],
        "Value": [
            p["threshold"], p["P_FLAG1"], p["P_FLAG0"],
            p["P_anomaly_given_FLAG1"], p["P_anomaly_given_FLAG0"],
            p["P_anomaly"], p["P_FLAG1_given_anomaly"],
        ],
    })
    st.dataframe(prob_df, use_container_width=True, hide_index=True)

    st.markdown("<br><div class='sg-section-title'>SVM Classification</div>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Baseline SVM (already evaluated)**")
        sm = mu.SVM_BASELINE_METRICS
        base_df = pd.DataFrame({"Metric": list(sm.keys()), "Value": [round(v, 4) for v in sm.values()]})
        st.dataframe(base_df, use_container_width=True, hide_index=True)
    with s2:
        st.markdown("**Final SVM (with polynomial features)**")
        if mu.SVM_FINAL_METRICS is None:
            st.markdown(
                "<div class='sg-placeholder'>Final evaluation is still in progress. "
                "Results will be populated here once available — "
                "(placeholder: <code>mu.SVM_FINAL_METRICS</code> in "
                "<code>utils/model_utils.py</code>).</div>",
                unsafe_allow_html=True,
            )
        else:
            final_df = pd.DataFrame({"Metric": list(mu.SVM_FINAL_METRICS.keys()),
                                      "Value": [round(v, 4) for v in mu.SVM_FINAL_METRICS.values()]})
            st.dataframe(final_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE 4: Methodology
# ---------------------------------------------------------------------------
elif page == "Methodology":
    theme.header("Methodology", "Pipeline, preprocessing, modeling, and evaluation details.")

    st.markdown("<div class='sg-section-title'>1. Preprocessing</div>", unsafe_allow_html=True)
    st.markdown(
        "- Removed severely damaged dates: **10/03/2014** (99.96% missing), **03/19/2014** (50.75% missing)\n"
        "- Confirmed erroneous reading **800003.32** was replaced with NaN\n"
        "- Retained consumers with at least **300 valid readings**\n"
        "- Final target period: **10/02/2016 to 10/31/2016**\n"
        f"- Consumers: {mu.DATASET_STATS['consumers_original']:,} → cleaned to "
        f"{mu.DATASET_STATS['consumers_after_cleaning']:,} → modeling set of "
        f"{mu.DATASET_STATS['consumers_modeling']:,}"
    )

    st.markdown("<div class='sg-section-title'>2. Feature Engineering</div>", unsafe_allow_html=True)
    st.write("Behavioral features computed per consumer from rolling windows of daily readings:")
    st.code(", ".join(mu.FEATURE_NAMES), language="text")

    st.markdown("<div class='sg-section-title'>3. Polynomial Curve Fitting</div>", unsafe_allow_html=True)
    st.markdown(
        "A quadratic polynomial is fit to the most recent 30 days of consumption "
        "history, generating:\n"
        "- `poly_rmse_30`\n- `poly_end_slope_30`\n- `poly_end_residual_30`"
    )

    st.markdown("<div class='sg-section-title'>4. Bayesian Linear Regression</div>", unsafe_allow_html=True)
    st.markdown(
        "An Enhanced Bayesian Linear Regression model is trained on the behavioral "
        "features above (log-scale target) to predict expected consumption and its "
        f"uncertainty. Reported performance: MAE = {mu.BAYESIAN_METRICS['MAE']}, "
        f"RMSE = {mu.BAYESIAN_METRICS['RMSE']}, R² = {mu.BAYESIAN_METRICS['R2']}, "
        f"mean uncertainty = {mu.BAYESIAN_METRICS['mean_uncertainty']}."
    )

    st.markdown("<div class='sg-section-title'>5. Anomaly Score</div>", unsafe_allow_html=True)
    st.latex(r"z = \frac{\text{actual\_log} - \text{predicted\_log}}{\text{prediction\_std}} "
             r"\qquad \text{anomaly\_score} = |z|")

    st.markdown("<div class='sg-section-title'>6. SVM Classification</div>", unsafe_allow_html=True)
    st.markdown(
        "An SVM classifier consumes the behavioral + polynomial features to output a "
        "binary label, surfaced in the UI as **Normal** or **Potentially Suspicious** "
        "(never *'Theft Confirmed'*). A baseline SVM has been evaluated; a final "
        "version using polynomial features is still being evaluated."
    )

    st.markdown("<div class='sg-section-title'>7. Important Framing Note</div>", unsafe_allow_html=True)
    st.warning(
        "This system flags **statistical deviation from expected consumption**, "
        "not confirmed electricity theft. All positive flags should read as "
        "'Potentially Suspicious — Requires Investigation'.",
        icon="⚠️",
    )
