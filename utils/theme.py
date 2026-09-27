"""
theme.py
--------
Central place for the dashboard's visual style: colors, CSS injection,
and small style helpers used across all pages. Keeping this separate
from app.py keeps the page logic (Model/Controller-ish) clean.
"""

import streamlit as st

# ---- Color palette (dark blue / white academic theme) --------------------
NAVY = "#0B1F3A"
NAVY_LIGHT = "#13294B"
BLUE = "#1E56A0"
ACCENT = "#2E7DD1"
BG = "#F4F6F9"
CARD_BG = "#FFFFFF"
TEXT_DARK = "#0B1F3A"
TEXT_MUTED = "#5B6B82"
GOOD_GREEN = "#1E7A34"
GOOD_BG = "#E7F5EA"
WARN_RED = "#B3261E"
WARN_BG = "#FBEAEA"


def inject_css():
    """Inject the shared CSS. Call once per page render, right after
    st.set_page_config(). Keeping it minimal and card-based on purpose —
    this is meant to look like a faculty-review academic demo, not a
    commercial product site."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {BG};
        }}

        /* Header banner */
        .sg-header {{
            background-color: {NAVY};
            color: white;
            padding: 1.2rem 1.6rem;
            border-radius: 10px;
            margin-bottom: 1.2rem;
        }}
        .sg-header h1 {{
            margin: 0;
            font-size: 1.55rem;
            color: white;
        }}
        .sg-header p {{
            margin: 0.2rem 0 0 0;
            color: #C9D6E8;
            font-size: 0.95rem;
        }}

        /* Metric / info cards */
        .sg-card {{
            background-color: {CARD_BG};
            border: 1px solid #E1E6EE;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            box-shadow: 0 1px 3px rgba(11,31,58,0.06);
        }}
        .sg-card-label {{
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            margin-bottom: 0.15rem;
        }}
        .sg-card-value {{
            color: {TEXT_DARK};
            font-size: 1.5rem;
            font-weight: 700;
        }}
        .sg-card-unit {{
            color: {TEXT_MUTED};
            font-size: 0.8rem;
            font-weight: 400;
            margin-left: 0.2rem;
        }}

        /* Classification banner */
        .sg-result-normal {{
            background-color: {GOOD_BG};
            border: 1px solid {GOOD_GREEN};
            border-radius: 10px;
            padding: 1rem 1.2rem;
        }}
        .sg-result-normal h2 {{
            color: {GOOD_GREEN};
            margin: 0.1rem 0;
        }}
        .sg-result-suspicious {{
            background-color: {WARN_BG};
            border: 1px solid {WARN_RED};
            border-radius: 10px;
            padding: 1rem 1.2rem;
        }}
        .sg-result-suspicious h2 {{
            color: {WARN_RED};
            margin: 0.1rem 0;
        }}

        .sg-section-title {{
            color: {NAVY};
            font-weight: 700;
            margin-top: 0.6rem;
        }}

        .sg-placeholder {{
            border: 1px dashed #B7C2D3;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            color: {TEXT_MUTED};
            background-color: #FAFBFC;
            font-size: 0.9rem;
        }}

        .sg-demo-badge {{
            display: inline-block;
            background-color: #FFF4D6;
            color: #7A5B00;
            border: 1px solid #E8C766;
            border-radius: 6px;
            padding: 0.15rem 0.6rem;
            font-size: 0.78rem;
            margin-bottom: 0.6rem;
        }}

        /* Tighten default streamlit spacing a bit */
        div.block-container {{padding-top: 1.2rem;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="sg-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, unit: str = ""):
    st.markdown(
        f"""
        <div class="sg-card">
            <div class="sg-card-label">{label}</div>
            <div class="sg-card-value">{value}<span class="sg-card-unit">{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def classification_banner(label: str):
    """label is expected to be 'Normal' or 'Potentially Suspicious'."""
    if label == "Normal":
        st.markdown(
            f"""
            <div class="sg-result-normal">
                <div style="font-size:0.85rem;color:{GOOD_GREEN};font-weight:600;">CLASSIFICATION RESULT</div>
                <h2>✅ Normal</h2>
                <div style="color:{TEXT_MUTED};font-size:0.88rem;">
                    This consumer's consumption pattern is within the expected range.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="sg-result-suspicious">
                <div style="font-size:0.85rem;color:{WARN_RED};font-weight:600;">CLASSIFICATION RESULT</div>
                <h2>⚠️ Potentially Suspicious</h2>
                <div style="color:{TEXT_MUTED};font-size:0.88rem;">
                    This consumption pattern deviates from the expected model and may
                    require further investigation. This is <b>not</b> proof of theft.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def demo_mode_badge():
    st.markdown(
        '<div class="sg-demo-badge">⚠ Running in DEMO MODE — synthetic data / '
        'placeholder model (no dataset or trained models found)</div>',
        unsafe_allow_html=True,
    )
