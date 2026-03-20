"""
Syndromic Surveillance Dashboard
=================================
Streamlit application with two views:
  1. Hospital Hub View   — per-hospital patient volume, cost, anomaly ticker
  2. Command Center      — UMAP+HDBSCAN clustering, SC map, drill-down
"""
from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()  # reads .env in the current working directory

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    SC_HOSPITAL_HUBS,
    HUB_LOOKUP,
    Colors,
    NUM_DAYS,
    SC_CENTER_LAT,
    SC_CENTER_LNG,
    SC_ZOOM,
    INPUT_COST_PER_M,
    OUTPUT_COST_PER_M,
    FEWSHOT_DIR,
    GENERATION_V3_DIR,
    CCDA_DIR,
    CLEAN_TRANSCRIPTS_DIR,
)
from data_loader import load_simulation_data, load_embeddings
from clustering_engine import (
    precompute_all_days, build_categorical_matrix, build_onehot_matrix,
    _build_detail_types, _build_novelty_flags, get_symptom_names, WINDOW_SIZE,
)

# ─── Cluster color palette (tab10 style, matches clustering_pipeline) ────
CLUSTER_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

DATE_LABELS = {
    1: "Dec 26", 2: "Dec 27", 3: "Dec 28", 4: "Dec 29",
    5: "Dec 30", 6: "Dec 31", 7: "Jan 1",
}

CASE_DISPLAY = {
    "novel_virus": "Novel Virus",
    "flu_like": "Flu-Like",
    "differential": "Differential",
    "healthy": "Healthy",
}

CASE_COLORS = {
    "novel_virus":  "#e31a1c",
    "flu_like":     "#1f78b4",
    "differential": "#6a3d9a",
    "healthy":      "#33a02c",
}

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Syndromic Surveillance  |  SC",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Google Fonts
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@500;700&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

_CSS = """
<style>
/* ── page background ────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background-color: #F8FAFC; }
[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }

/* ── kill top padding ───────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > div:first-child { padding-top: 0 !important; }
.main .block-container { padding-top: 0.15rem !important; max-width: 100%; }
[data-testid="stAppViewBlockContainer"] { padding-top: 0.15rem !important; }
header[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }

/* ── typography ─────────────────────────────────────────────────────── */
html, body, [class*="st-"], p, span, div, label, td, th,
[data-testid="stMarkdownContainer"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #0F172A;
}
h1, h2, h3, h4, h5, h6, .card h3, [data-testid="stMetricLabel"] {
    font-family: 'Source Code Pro', monospace !important;
}
/* restore Streamlit icon font overridden by span selector above */
span[data-testid="stIconMaterial"],
.material-symbols-rounded,
[data-testid="stBaseButton-headerNoPadding"] span,
[data-testid="stExpander"] summary span[data-testid="stIconMaterial"],
[data-testid="stExpander"] summary .material-symbols-rounded,
details summary span.e1nzilvr5,
details[data-testid="stExpander"] summary span {
    font-family: 'Material Symbols Rounded' !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
    font-style: normal !important;
    font-variant: normal !important;
    text-rendering: auto !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ── toggle bar ─────────────────────────────────────────────────────── */
.toggle-bar {
    display: flex; gap: 0; margin-bottom: 1.2rem;
    border: 1px solid #E2E8F0; border-radius: 8px;
    overflow: hidden; width: fit-content;
}
.toggle-bar .tb {
    padding: 8px 28px;
    font-family: 'Source Code Pro', monospace;
    font-size: 0.82rem; font-weight: 500;
    cursor: pointer; border: none; transition: all 0.15s ease;
    text-decoration: none; display: inline-block;
    letter-spacing: 0.03em;
}
.toggle-bar .tb.active {
    background: #2563EB; color: #FFFFFF;
}
.toggle-bar .tb.inactive {
    background: #FFFFFF; color: #64748B;
}
.toggle-bar .tb.inactive:hover {
    background: #F1F5F9; color: #0F172A;
}

/* ── cards ──────────────────────────────────────────────────────────── */
.card {
    background: #FFFFFF; border-radius: 10px;
    padding: 1.4rem 1.6rem; border: 1px solid #E2E8F0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04); margin-bottom: 1rem;
}
.card h3 {
    font-size: 0.78rem; font-weight: 600; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0 0 0.5rem 0;
}
.card .big-num {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 2.2rem; font-weight: 700; color: #0F172A; line-height: 1.1;
}
.card .sub { font-size: 0.82rem; color: #64748B; margin-top: 0.2rem; }

/* ── badge ──────────────────────────────────────────────────────────── */
.badge {
    display: inline-block; background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 6px; padding: 3px 10px; font-size: 0.70rem;
    font-weight: 600; color: #2563EB; letter-spacing: 0.03em; margin-top: 0.5rem;
}

/* ── anomaly card ───────────────────────────────────────────────────── */
.anomaly-item {
    background: #EFF6FF; border-left: 3px solid #2563EB;
    border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}
.anomaly-item .term { font-weight: 700; color: #2563EB; font-size: 0.95rem; }
.anomaly-item .quote { font-style: italic; color: #64748B; font-size: 0.82rem; margin-top: 0.25rem; }
.anomaly-item .pid { font-size: 0.72rem; color: #94A3B8; margin-top: 0.15rem; }

/* ── status badges ──────────────────────────────────────────────────── */
.status-badge {
    display: inline-block; border-radius: 6px; padding: 4px 12px;
    font-family: 'Source Code Pro', monospace; font-size: 0.75rem;
    font-weight: 600; letter-spacing: 0.04em;
}
.status-badge.outbreak { background: #FEF2F2; border: 1px solid #FECACA; color: #DC2626; }
.status-badge.monitoring { background: #F0FDF4; border: 1px solid #BBF7D0; color: #16A34A; }

/* ── placeholder panel ──────────────────────────────────────────────── */
.placeholder-panel {
    background: #FFFFFF; border: 2px dashed #CBD5E1; border-radius: 10px;
    padding: 2.5rem 2rem; text-align: center; margin: 1.5rem 0;
}
.placeholder-panel h4 {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 1rem; font-weight: 600; color: #64748B; margin: 0 0 0.4rem 0;
}
.placeholder-panel p { font-size: 0.85rem; color: #94A3B8; margin: 0; }

/* ── section header ─────────────────────────────────────────────────── */
.section-header {
    font-family: 'Source Code Pro', monospace;
    font-size: 0.95rem; font-weight: 600; color: #0F172A;
    border-bottom: 2px solid #E2E8F0; padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── code blocks + JSON viewer (prompt / scenario viewer) ────────────── */
[data-testid="stCode"],
[data-testid="stCode"] pre,
[data-testid="stCode"] code,
.stCodeBlock,
.stCodeBlock pre,
.stCodeBlock code {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}
[data-testid="stCode"] pre code,
.stCodeBlock pre code {
    border: none !important;
}
[data-testid="stJson"],
[data-testid="stJson"] * {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
[data-testid="stExpander"] [data-testid="stCode"],
[data-testid="stExpander"] [data-testid="stCode"] pre,
[data-testid="stExpander"] [data-testid="stCode"] code,
[data-testid="stExpander"] [data-testid="stJson"],
[data-testid="stExpander"] [data-testid="stJson"] * {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}

/* ── streamlit overrides ────────────────────────────────────────────── */
[data-testid="stExpander"] summary span { font-family: 'Space Grotesk', sans-serif !important; font-weight: 500; color: #0F172A !important; }
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
[data-testid="stMetricValue"] { font-family: 'Source Code Pro', monospace !important; font-size: 2rem; font-weight: 700; color: #0F172A !important; }
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #64748B !important; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stSelectbox"] > div > div, [data-testid="stSlider"], .stButton > button { border-radius: 8px !important; }
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600;
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    transition: all 0.15s ease;
    padding: 0.45rem 1.4rem;
    min-width: fit-content;
    white-space: nowrap;
}
.stButton > button:hover {
    background-color: #EFF6FF !important;
    border-color: #2563EB !important;
    color: #2563EB !important;
}
.stButton > button:focus {
    background-color: #EFF6FF !important;
    border-color: #2563EB !important;
    color: #2563EB !important;
    box-shadow: none !important;
}

/* ── link button (GitHub Repository) ──────────────────────────────────────── */
.stLinkButton > a,
.stLinkButton a,
a[data-testid="baseLinkButton-secondary"],
[data-testid="stLinkButton"] > a,
[data-testid="stLinkButton"] a,
div.stLinkButton > a {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    transition: all 0.15s ease;
    padding: 0.45rem 1.4rem !important;
    min-width: fit-content;
    white-space: nowrap;
    text-decoration: none !important;
    display: inline-flex !important;
    align-items: center !important;
}
.stLinkButton > a:hover,
.stLinkButton a:hover,
a[data-testid="baseLinkButton-secondary"]:hover,
[data-testid="stLinkButton"] > a:hover,
[data-testid="stLinkButton"] a:hover,
div.stLinkButton > a:hover {
    background-color: #EFF6FF !important;
    border-color: #2563EB !important;
    color: #2563EB !important;
}
.stLinkButton > a *,
[data-testid="stLinkButton"] a * {
    color: inherit !important;
}

[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── tabs styling ───────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    font-family: 'Source Code Pro', monospace !important;
    font-weight: 600;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
    color: #64748B !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.7rem 1.6rem;
    transition: all 0.15s ease;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 3px solid #2563EB !important;
}
[data-testid="stTabs"] button:hover {
    color: #0F172A !important;
    background: #F1F5F9 !important;
    border-radius: 6px 6px 0 0;
}

/* ── anomaly scroll container ───────────────────────────────────────── */
.anomaly-scroll {
    max-height: 420px;
    overflow-y: auto;
    padding-right: 0.5rem;
}
.anomaly-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
}
@media (max-width: 768px) {
    .anomaly-grid { grid-template-columns: 1fr; }
}

/* ── slider and other text visibility fixes ─────────────────────────── */
[data-testid="stSlider"] label, [data-testid="stSelectbox"] label,
[data-testid="stSlider"] div, [data-testid="stTabs"] div {
    color: #0F172A !important;
}
.stMarkdown p, .stMarkdown span, .stMarkdown div {
    color: #0F172A;
}

/* ── slider label size ──────────────────────────────────────────────── */
[data-testid="stSlider"] label p {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    font-family: 'Source Code Pro', monospace !important;
}

/* ── force light backgrounds on all inputs ──────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] > div > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
[data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div,
[data-baseweb="select"] > div > div > div,
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[data-baseweb="menu"] > div,
[data-baseweb="menu"] li,
[data-baseweb="menu"] ul,
[role="listbox"],
[role="listbox"] li,
[role="option"],
input, select, textarea,
[data-testid="stSlider"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-testid="stTextInput"] > div > div,
[data-testid="stNumberInput"] > div > div {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] input,
[data-baseweb="menu"] li,
[data-baseweb="menu"] li span,
[data-baseweb="menu"] li div,
[role="option"],
[role="option"] span,
[role="option"] div,
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] p {
    color: #0F172A !important;
}
[data-baseweb="select"] svg {
    fill: #64748B !important;
}
/* highlighted / focused option */
[data-baseweb="menu"] li[aria-selected="true"],
[data-baseweb="menu"] li:hover {
    background-color: #EFF6FF !important;
    color: #0F172A !important;
}

/* ── expander arrow fix ─────────────────────────────────────────────── */
[data-testid="stExpander"] summary {
    padding: 0.6rem 0.8rem !important;
    gap: 0.5rem !important;
}
[data-testid="stExpander"] summary svg {
    flex-shrink: 0;
    width: 16px;
    height: 16px;
}
[data-testid="stExpander"] details {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── KPI metric cards (command center) ──────────────────────────────── */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: #0F172A !important;
}
.kpi-card {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.kpi-card:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.15);
    transform: translateY(-2px);
}
.kpi-card .kpi-label {
    font-family: 'Source Code Pro', monospace;
    font-size: 0.72rem; font-weight: 600; color: #64748B;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0 0 0.4rem 0;
}
.kpi-card .kpi-value {
    font-family: 'Source Code Pro', monospace;
    font-size: 1.8rem; font-weight: 700; color: #0F172A;
    line-height: 1.1;
}
.kpi-card .kpi-sub {
    font-size: 0.78rem; color: #94A3B8; margin-top: 0.2rem;
}

/* ── general hover on card class ────────────────────────────────────── */
.card {
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.card:hover {
    box-shadow: 0 8px 24px rgba(37,99,235,0.14);
    transform: translateY(-2px);
}
.anomaly-item {
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.anomaly-item:hover {
    box-shadow: 0 4px 14px rgba(37,99,235,0.14);
    transform: translateY(-2px);
}

/* ── chart box wrapper ──────────────────────────────────────────────── */
.chart-box {
    border: 1.5px solid #1E293B;
    border-radius: 10px;
    padding: 0.6rem 0.7rem 0.3rem 0.7rem;
    background: #FFFFFF;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: box-shadow 0.2s ease;
}
.chart-box:hover {
    box-shadow: 0 6px 20px rgba(37,99,235,0.12);
}

/* ── anomaly banner ─────────────────────────────────────────────────── */
.anomaly-banner {
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem 1.4rem; border-radius: 10px;
    margin-bottom: 0.8rem;
}
.anomaly-banner.alert {
    background: linear-gradient(135deg, #FEF2F2, #FFF7ED);
    border: 1.5px solid #FECACA;
}
.anomaly-banner.clear {
    background: linear-gradient(135deg, #F0FDF4, #ECFDF5);
    border: 1.5px solid #BBF7D0;
}
.anomaly-icon {
    display: flex; align-items: center; justify-content: center;
    width: 42px; height: 42px; border-radius: 50%;
    font-size: 1.3rem; font-weight: 800;
    font-family: 'Source Code Pro', monospace;
    flex-shrink: 0;
}
.anomaly-icon.red { background: #DC2626; color: white; }
.anomaly-icon.green { background: #16A34A; color: white; }
.anomaly-count {
    font-family: 'Source Code Pro', monospace;
    font-size: 2rem; font-weight: 700; color: #0F172A;
    line-height: 1;
}
.anomaly-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem; color: #64748B; font-weight: 500;
}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

if "current_view" not in st.session_state:
    st.session_state.current_view = "hospital"
if "playing" not in st.session_state:
    st.session_state.playing = False

# ── Sentinel AI branding (logo + title) ───────────────────────────────
import base64 as _b64
from pathlib import Path as _Path

_logo_path = _Path(__file__).parent / "assets" / "sentinel_logo.svg"
if _logo_path.exists():
    _logo_b64 = _b64.b64encode(_logo_path.read_bytes()).decode()
    _logo_html = (
        f'<img src="data:image/svg+xml;base64,{_logo_b64}" '
        f'style="height:2.2rem;width:auto;margin-right:0.5rem;" />'
    )
else:
    _logo_html = ""

st.markdown(
    '<div style="display:flex;align-items:center;gap:0.7rem;padding:0.3rem 0 0.2rem 0;">'
    f'{_logo_html}'
    '<div style="font-family:\'Source Code Pro\',monospace;font-weight:700;'
    'font-size:1.45rem;color:#0F172A;letter-spacing:0.04em;">'
    'SENTINEL AI</div>'
    '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.78rem;'
    'color:#64748B;border-left:2px solid #E2E8F0;padding-left:0.7rem;">'
    'Syndromic Surveillance Dashboard</div></div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

df  = load_simulation_data()
emb = load_embeddings()
cat = build_categorical_matrix(str(FEWSHOT_DIR))       # kept for Symptom Analysis tab
oh, _symp_names = build_onehot_matrix(str(FEWSHOT_DIR))
detail_types  = _build_detail_types(str(FEWSHOT_DIR))
novelty_flags = _build_novelty_flags(str(FEWSHOT_DIR))

import json as _json
_df_json       = df.to_json()
_emb_bytes     = emb.astype(np.float64).tobytes()
_oh_bytes      = oh.astype(np.float64).tobytes()
_oh_shape      = oh.shape
_detail_json   = _json.dumps(detail_types)
_novelty_json  = _json.dumps([bool(v) for v in novelty_flags])
clustering = precompute_all_days(
    _df_json, _emb_bytes, _oh_bytes, _oh_shape, _detail_json, _novelty_json, NUM_DAYS
)


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTLY DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

_PLOTLY = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Space Grotesk, sans-serif", color="#0F172A", size=14),
    margin=dict(l=50, r=20, t=36, b=50),
    legend_font=dict(color="#0F172A", size=12),
    hoverlabel=dict(bgcolor="#FFFFFF", font_size=13, font_color="#0F172A",
                    bordercolor="#E2E8F0"),
)
_AXIS_FONT = dict(tickfont=dict(color="#0F172A", size=13), title_font=dict(color="#0F172A", size=14))


# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL CONTROLS — toggle bar + universal day slider
# ═══════════════════════════════════════════════════════════════════════════

_is_hospital = st.session_state.current_view == "hospital"

# Row 1: Toggle buttons only (slider moved into each view)
_btn1, _btn2, _btn3, _btn4 = st.columns(
    [1.2, 1.8, 1.5, 1.2], gap="medium"
)
with _btn1:
    if st.button("Hospital Hub", key="btn_hospital", use_container_width=True):
        st.session_state.current_view = "hospital"
        st.rerun()
with _btn2:
    if st.button("Public Health Dashboard", key="btn_command", use_container_width=True):
        st.session_state.current_view = "command"
        st.rerun()
with _btn3:
    if st.button("Conversation Demo", key="btn_conv_demo", use_container_width=True):
        st.session_state.current_view = "conv_demo"
        st.rerun()
with _btn4:
    st.link_button("GitHub Repository", "https://github.com/AJain271/Sentinel-AI-LLM-Based-Epidemic-Surveillance", help="Open the project repository")

# Default day — each view will override with its own slider
day = st.session_state.get("sim_day", 1)
play_clicked = False

# Indicator line
_VIEW_LABELS = {
    "hospital": "HOSPITAL HUB VIEW",
    "command": "PUBLIC HEALTH DASHBOARD",
    "conv_demo": "CONVERSATION DEMONSTRATION PIPELINE",
}
_indicator_text = _VIEW_LABELS.get(st.session_state.current_view, "SENTINEL AI")
st.markdown(
    f'<div style="font-family:Source Code Pro,monospace;font-size:0.70rem;'
    f'font-weight:600;color:#2563EB;letter-spacing:0.10em;'
    f'border-bottom:2px solid #2563EB;display:inline-block;'
    f'padding-bottom:2px;margin-bottom:0.5rem;">{_indicator_text}'
    f'  |  Day {day} ({DATE_LABELS.get(day, "")})</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  VIEW 1 — HOSPITAL HUB
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.current_view == "hospital":

    # ── Day slider + Play button ─────────────────────────────────────
    _h_sl, _h_pl = st.columns([6, 1])
    with _h_sl:
        day = st.slider("Simulation Day", 1, 7, value=1, key="sim_day", format="Day %d")
    with _h_pl:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        play_clicked = st.button("▶ Play", key="play_btn")
    if play_clicked and not st.session_state.get("_playing", False):
        st.session_state["_playing"] = True
        st.session_state["_play_day"] = day
    if st.session_state.get("_playing", False):
        _pd = st.session_state.get("_play_day", 1)
        if _pd <= 7:
            day = _pd
            st.markdown(
                f'<div style="font-size:0.85rem;color:#2563EB;font-weight:600;'
                f'font-family:Source Code Pro,monospace">'
                f'Playing Day {_pd} / 7  |  {DATE_LABELS.get(_pd, "")}</div>',
                unsafe_allow_html=True)
            if _pd < 7:
                st.session_state["_play_day"] = _pd + 1
                time.sleep(2)
                st.rerun()
            else:
                st.session_state.pop("_playing", None)
                st.session_state.pop("_play_day", None)
        else:
            st.session_state.pop("_playing", None)
            st.session_state.pop("_play_day", None)

    # ── Controls row: hospital dropdown + scale ──────────────────────
    ctrl1, ctrl2 = st.columns([4, 1])
    with ctrl1:
        hub_options = {
            f"{h['name']}  --  {h['city']}": h["hub_id"]
            for h in SC_HOSPITAL_HUBS
        }
        selected_hub_label = st.selectbox(
            "Select Hospital", list(hub_options.keys()), label_visibility="collapsed"
        )
        selected_hub_id = hub_options[selected_hub_label]
    with ctrl2:
        scale = st.selectbox("Scale", [1, 2, 5, 10], index=0, key="hub_scale",
                             format_func=lambda x: f"{x}x" if x > 1 else "Actual")

    hub_info = HUB_LOOKUP[selected_hub_id]
    day_df = df[(df["hub_id"] == selected_hub_id) & (df["day"] == day)]
    _day_subtitle = f"Day {day} ({DATE_LABELS.get(day, '')})"

    st.markdown(
        f"<h1 style='font-size:1.6rem;font-weight:700;margin-bottom:0'>"
        f"{hub_info['name']}</h1>"
        f"<p style='color:#64748B;margin-top:0;font-size:0.9rem'>"
        f"{hub_info['city']}, SC  |  {_day_subtitle}</p>",
        unsafe_allow_html=True,
    )

    # ── KPI cards ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    n_patients = len(day_df)
    day_cost = day_df["total_cost"].sum() * scale
    input_c  = day_df["input_cost"].sum() * scale
    output_c = day_df["output_cost"].sum() * scale
    n_flagged = int(day_df["Novelty_Flag"].sum())

    with c1:
        st.markdown(
            f"""<div class="card">
                <h3>Patient Volume</h3>
                <div class="big-num">{n_patients * scale}</div>
                <div class="sub">{'Actual' if scale == 1 else f'{scale}x projected'}</div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(
            f"""<div class="card">
                <h3>AI Processing Cost</h3>
                <div class="big-num">${day_cost:,.4f}</div>
                <div class="sub">Input ${input_c:,.4f}  |  Output ${output_c:,.4f}</div>
                <div class="badge">FEW-SHOT  |  GPT-4o</div>
            </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(
            f"""<div class="card">
                <h3>Novelty Flags</h3>
                <div class="big-num" style="color:{'#2563EB' if n_flagged else '#0F172A'}">{n_flagged * scale}</div>
                <div class="sub">Unmapped symptoms detected</div>
            </div>""", unsafe_allow_html=True)

    # ── Charts row ─────────────────────────────────────────────────────
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Patient Volume by Case Type</div>',
                    unsafe_allow_html=True)
        case_counts = day_df["case_type"].value_counts().reindex(
            ["novel_virus", "flu_like", "differential", "healthy"], fill_value=0
        ).reset_index()
        case_counts.columns = ["Case Type", "Count"]
        case_counts["Count"] = case_counts["Count"] * scale
        case_counts["Label"] = case_counts["Case Type"].map(CASE_DISPLAY)

        fig_vol = px.bar(case_counts, x="Label", y="Count", color="Case Type",
                         color_discrete_map=CASE_COLORS, text="Count")
        fig_vol.update_layout(**_PLOTLY, showlegend=False, yaxis_title="Patients",
                              xaxis_title="", height=400,
                              xaxis=_AXIS_FONT, yaxis=_AXIS_FONT)
        fig_vol.update_traces(textposition="outside", marker_line_width=0, width=0.55)
        fig_vol.update_xaxes(showgrid=False)
        fig_vol.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
        st.plotly_chart(fig_vol, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with chart2:
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Case Type Distribution</div>',
                    unsafe_allow_html=True)
        dist = day_df["case_type"].value_counts().reset_index()
        dist.columns = ["Case Type", "Count"]
        dist["Label"] = dist["Case Type"].map(CASE_DISPLAY)
        fig_pie = px.pie(dist, values="Count", names="Label",
                         color="Case Type", color_discrete_map=CASE_COLORS,
                         hole=0.45)
        fig_pie.update_layout(**_PLOTLY, height=320, showlegend=True,
                              legend=dict(font=dict(size=12, color="#0F172A")))
        fig_pie.update_traces(textinfo="percent+value",
                              textfont_size=12, textfont_color="#0F172A")
        st.plotly_chart(fig_pie, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Token Cost by Hospital ────────────────────────────────────────
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Token Cost by Hospital</div>',
                unsafe_allow_html=True)
    _day_all = df[df["day"] == day]
    _cost_by_hub = _day_all.groupby("hub_id").agg(
        Input=pd.NamedAgg(column="input_cost", aggfunc="sum"),
        Output=pd.NamedAgg(column="output_cost", aggfunc="sum"),
    ).reset_index()
    _cost_by_hub["Hospital"] = _cost_by_hub["hub_id"].map(
        {h["hub_id"]: h["name"] for h in SC_HOSPITAL_HUBS}
    )
    _cost_melt = _cost_by_hub.melt(id_vars="Hospital", value_vars=["Input", "Output"],
                                    var_name="Cost Type", value_name="USD")
    fig_cost = px.bar(_cost_melt, x="Hospital", y="USD", color="Cost Type",
                      barmode="group",
                      color_discrete_map={"Input": "#2563EB", "Output": "#F97316"},
                      text_auto=".4f")
    fig_cost.update_layout(**_PLOTLY, height=420, xaxis_title="", yaxis_title="Cost (USD)",
                           xaxis=dict(**_AXIS_FONT, tickangle=-30), yaxis=_AXIS_FONT,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                       xanchor="center", x=0.5,
                                       font=dict(size=12, color="#0F172A")))
    fig_cost.update_traces(textposition="outside")
    st.plotly_chart(fig_cost, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Anomaly Ticker ─────────────────────────────────────────────────
    flagged = day_df[day_df["Novelty_Flag"]].copy()
    _n_anomalies = sum(len(r["unmapped_anomalies"]) for _, r in flagged.iterrows()) if not flagged.empty else 0

    if flagged.empty:
        st.markdown(
            '<div class="anomaly-banner clear">'
            '<span class="anomaly-icon green">&#10003;</span>'
            '<span class="anomaly-count" style="color:#16A34A">0</span>'
            '<span class="anomaly-label">No novel symptoms detected</span>'
            '</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="anomaly-banner alert">'
            f'<span class="anomaly-icon red">!</span>'
            f'<span class="anomaly-count">{_n_anomalies}</span>'
            f'<span class="anomaly-label">Novel Symptom{"s" if _n_anomalies != 1 else ""} Detected'
            f'  |  {len(flagged)} patient{"s" if len(flagged) != 1 else ""}</span>'
            f'</div>',
            unsafe_allow_html=True)
        anomaly_cards = []
        for _, row in flagged.iterrows():
            terms  = row["unmapped_anomalies"]
            quotes = row["unmapped_quotes"]
            hospital = row.get("hospital", "")
            case_label = CASE_DISPLAY.get(row["case_type"], row["case_type"])
            day_num = row["day"]
            for t, q in zip(terms, quotes):
                anomaly_cards.append(
                    f'<div class="anomaly-item">'
                    f'<div class="term">{t}</div>'
                    f'<div class="quote">"{q}"</div>'
                    f'<div class="pid">{hospital}  |  Day {day_num}  |  {case_label}</div>'
                    f'</div>'
                )
        grid_html = '<div class="anomaly-scroll"><div class="anomaly-grid">' + ''.join(anomaly_cards) + '</div></div>'
        st.markdown(grid_html, unsafe_allow_html=True)

    # ── Cost Breakdown ─────────────────────────────────────────────────
    with st.expander("Cost Breakdown"):
        avg_prompt = day_df["prompt_tokens"].mean() if not day_df.empty else 0
        avg_comp   = day_df["completion_tokens"].mean() if not day_df.empty else 0
        st.markdown(f"""
| Metric | Value |
|--------|-------|
| Input rate | ${INPUT_COST_PER_M:.2f} / 1 M tokens |
| Output rate | ${OUTPUT_COST_PER_M:.2f} / 1 M tokens |
| Avg prompt tokens | {avg_prompt:,.0f} |
| Avg completion tokens | {avg_comp:,.0f} |
| Patients (actual) | {n_patients} |
| Scale factor | {scale}x |
| **Total projected cost** | **${day_cost:,.4f}** |
""")


# ═══════════════════════════════════════════════════════════════════════════
#  VIEW 2 — PUBLIC HEALTH DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.current_view == "command":
    st.markdown(
        "<h1 style='font-size:1.6rem;font-weight:700;margin-bottom:0'>"
        "Public Health Dashboard</h1>"
        "<p style='color:#64748B;margin-top:0;font-size:0.9rem'>"
        "South Carolina  |  Cumulative Outbreak Analysis</p>",
        unsafe_allow_html=True,
    )

    # ── Day slider + Play button ─────────────────────────────────────
    _c_sl, _c_pl = st.columns([6, 1])
    with _c_sl:
        day = st.slider("Simulation Day", 1, 7, value=1, key="sim_day", format="Day %d")
    with _c_pl:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        play_clicked = st.button("▶ Play", key="play_btn")
    if play_clicked and not st.session_state.get("_playing", False):
        st.session_state["_playing"] = True
        st.session_state["_play_day"] = day
    if st.session_state.get("_playing", False):
        _pd = st.session_state.get("_play_day", 1)
        if _pd <= 7:
            day = _pd
            st.markdown(
                f'<div style="font-size:0.85rem;color:#2563EB;font-weight:600;'
                f'font-family:Source Code Pro,monospace">'
                f'Playing Day {_pd} / 7  |  {DATE_LABELS.get(_pd, "")}</div>',
                unsafe_allow_html=True)
            if _pd < 7:
                st.session_state["_play_day"] = _pd + 1
                time.sleep(2)
                st.rerun()
            else:
                st.session_state.pop("_playing", None)
                st.session_state.pop("_play_day", None)
        else:
            st.session_state.pop("_playing", None)
            st.session_state.pop("_play_day", None)

    res = clustering[day]
    cum_df = df.iloc[res["idx"]].copy()
    cum_df["_cluster"] = res["labels"]
    cum_df["umap_1"]   = res["umap_2d"][:, 0]
    cum_df["umap_2"]   = res["umap_2d"][:, 1]
    outbreak_label     = res["outbreak_label"]
    label_names        = res.get("label_names", {})
    window_start       = res.get("window_start", max(1, day - WINDOW_SIZE + 1))
    window_end         = res.get("window_end", day)

    n_cum         = len(cum_df)
    n_flagged_cum = int(cum_df["Novelty_Flag"].sum())
    n_novel       = int((cum_df["case_type"] == "novel_virus").sum())

    # Display-friendly names for clusters
    _DISPLAY_NAMES = {
        "healthy": "Healthy", "novel_virus": "Novel Virus",
        "flu_like": "Flu-Like", "gastrointestinal": "GI",
        "dermatological": "Derm", "musculoskeletal": "MSK",
        "neurological": "Neuro", "noise": "Noise",
        "differential": "Differential", "mixed": "Mixed",
    }
    def _pretty(cid: int) -> str:
        raw = label_names.get(cid, f"cluster_{cid}")
        return _DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())

    cum_df["_label_str"] = cum_df["_cluster"].apply(
        lambda x: "Noise" if x == -1 else _pretty(x)
    )

    # ── KPI row ────────────────────────────────────────────────────────
    _window_label = (f"Day {window_start}" if window_start == window_end
                     else f"Days {window_start}-{window_end}")
    _window_dates = (DATE_LABELS.get(window_start, "") if window_start == window_end
                     else f"{DATE_LABELS.get(window_start, '')} - {DATE_LABELS.get(window_end, '')}")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Patients in Window</div>'
            f'<div class="kpi-value">{n_cum}</div>'
            f'</div>',
            unsafe_allow_html=True)
    with k2:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Window</div>'
            f'<div class="kpi-value">{_window_label}</div>'
            f'<div class="kpi-sub">{_window_dates}</div>'
            f'</div>',
            unsafe_allow_html=True)
    with k3:
        _flag_color = "#DC2626" if n_flagged_cum else "#0F172A"
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Novelty Flags</div>'
            f'<div class="kpi-value" style="color:{_flag_color}">{n_flagged_cum}</div>'
            f'</div>',
            unsafe_allow_html=True)
    with k4:
        if outbreak_label is not None:
            st.markdown(
                '<div class="kpi-card" style="border-color:#FECACA">'
                '<div class="kpi-label">Status</div>'
                '<div style="margin-top:0.3rem"><span class="status-badge outbreak">'
                'OUTBREAK DETECTED</span></div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="kpi-card" style="border-color:#BBF7D0">'
                '<div class="kpi-label">Status</div>'
                '<div style="margin-top:0.3rem"><span class="status-badge monitoring">'
                'MONITORING</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Day slider right above the tabs ──────────────────────────────
    st.markdown(
        '<div style="font-family:Source Code Pro,monospace;font-size:0.82rem;'
        'font-weight:600;color:#64748B;margin-bottom:0.2rem;">'
        'Use the Simulation Day slider above to step through the outbreak timeline.</div>',
        unsafe_allow_html=True,
    )

    # ── Sub-tabs: State Map / Cluster Evolution / Case Data / Symptom Analysis / Early Warning
    n_real_clusters = res["n_clusters"]
    tab_ew, tab_map, tab_umap, tab_data, tab_symp, tab_r0 = st.tabs([
        "Early Warning",
        "State Map",
        "UMAP Projection",
        "Case Data",
        "Symptom Analysis",
        "R0 Forecast",
    ])

    # ···· Tab 1: SC Map — individual patient dots ·····················
    with tab_map:
        st.markdown(
            f'<div style="font-size:0.82rem;color:#64748B;font-weight:600;'
            f'font-family:Source Code Pro,monospace;margin-bottom:0.5rem">'
            f'{n_real_clusters} clusters detected</div>',
            unsafe_allow_html=True)

        # ── Side-by-side: Map (left) + Bar chart (right) ─────────────
        _map_col, _bar_col = st.columns([3, 2], gap="medium")

        # Build map figure (shared, rendered in left column)
        fig_map = go.Figure()

        # Plot patients by case type (non-novel first, novel on top)
        _map_custom_ids = {}  # trace-index -> list of df indices for click-to-detail
        for ct in ["healthy", "differential", "flu_like", "novel_virus"]:
            ct_sub = cum_df[cum_df["case_type"] == ct]
            if ct_sub.empty:
                continue
            is_novel = ct == "novel_virus"
            label = CASE_DISPLAY.get(ct, ct)
            count = len(ct_sub)

            hover_text = []
            custom_ids = []
            for _, r in ct_sub.iterrows():
                flag_str = "  [FLAGGED]" if r["Novelty_Flag"] else ""
                hover_text.append(
                    f"<b>{label}{flag_str}</b><br>"
                    f"Hub: {r['hub_id']}  --  {r.get('hospital', '')}<br>"
                    f"Day {r['day']}"
                )
                custom_ids.append(int(r.name))

            fig_map.add_trace(go.Scattergeo(
                lat=ct_sub["lat"].values,
                lon=ct_sub["lng"].values,
                mode="markers",
                marker=dict(
                    size=12 if is_novel else 8,
                    color=CASE_COLORS.get(ct, "#999"),
                    opacity=0.85,
                    line=dict(width=0.3, color="white"),
                    symbol="circle",
                ),
                name=f"{label} ({count})",
                hovertext=hover_text,
                hoverinfo="text",
                customdata=custom_ids,
            ))

        # Flagged patients — red circle with white "!"
        flagged_pts = cum_df[cum_df["Novelty_Flag"]]
        if not flagged_pts.empty:
            fig_map.add_trace(go.Scattergeo(
                lat=flagged_pts["lat"].values,
                lon=flagged_pts["lng"].values,
                mode="markers+text",
                marker=dict(
                    size=20, color="#DC2626",
                    line=dict(width=1.5, color="white"),
                    symbol="circle", opacity=0.9,
                ),
                text=["!" for _ in range(len(flagged_pts))],
                textfont=dict(color="white", size=10, family="Arial Black"),
                textposition="middle center",
                name=f"Flagged ({len(flagged_pts)})",
                hovertext=[
                    f"<b>FLAGGED</b><br>{r.get('hospital','')}<br>"
                    f"Day {r['day']}  |  {CASE_DISPLAY.get(r['case_type'], r['case_type'])}"
                    for _, r in flagged_pts.iterrows()
                ],
                hoverinfo="text",
            ))

        # Hospital hub markers — blue circle with white "+"
        fig_map.add_trace(go.Scattergeo(
            lat=[h["lat"] for h in SC_HOSPITAL_HUBS],
            lon=[h["lng"] for h in SC_HOSPITAL_HUBS],
            mode="markers+text",
            marker=dict(
                size=22, color="#2563EB", symbol="circle",
                line=dict(width=2, color="white"), opacity=0.95,
            ),
            text=["+" for _ in SC_HOSPITAL_HUBS],
            textfont=dict(color="white", size=14, family="Arial Black"),
            textposition="middle center",
            name="Hospital Hub",
            hovertext=[f"<b>{h['name']}</b><br>{h['city']}, SC" for h in SC_HOSPITAL_HUBS],
            hoverinfo="text",
        ))

        fig_map.update_layout(
            geo=dict(
                scope="usa", projection_type="albers usa",
                showland=True, landcolor="rgb(243, 243, 243)",
                showlakes=True, lakecolor="rgb(204, 230, 255)",
                showrivers=False,
                showcountries=False, showsubunits=True,
                subunitcolor="rgb(80, 80, 80)", subunitwidth=2.5,
                center=dict(lat=SC_CENTER_LAT, lon=SC_CENTER_LNG),
                lonaxis=dict(range=[-84.0, -78.0]),
                lataxis=dict(range=[31.8, 35.5]),
                bgcolor="white",
            ),
            height=550,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#FFFFFF",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.08, xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=10, family="Space Grotesk, sans-serif", color="#0F172A"),
            ),
            hoverlabel=dict(bgcolor="#FFFFFF", font_size=13, font_color="#0F172A",
                            bordercolor="#E2E8F0"),
        )

        with _map_col:
            st.markdown(
                f'<div style="font-size:0.85rem;color:#64748B;margin-top:0.3rem;">'
                f'Showing <b>{n_cum}</b> patients across <b>{n_real_clusters}</b> clusters  |  '
                f'{_window_dates}</div>',
                unsafe_allow_html=True)
            map_event = st.plotly_chart(fig_map, use_container_width=True,
                                        on_select="rerun", selection_mode="points")

        with _bar_col:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="section-header">Cases in Window  |  {_window_label}</div>',
                unsafe_allow_html=True)
            _cc = cum_df["case_type"].value_counts().reindex(
                ["novel_virus", "flu_like", "differential", "healthy"], fill_value=0
            ).reset_index()
            _cc.columns = ["Case Type", "Count"]
            _cc["Label"] = _cc["Case Type"].map(CASE_DISPLAY)
            fig_bar_cmd = px.bar(
                _cc, x="Label", y="Count", color="Case Type",
                color_discrete_map=CASE_COLORS, text="Count",
            )
            fig_bar_cmd.update_layout(
                **_PLOTLY, showlegend=False, height=340,
                xaxis_title="", yaxis_title="Patients",
                xaxis=_AXIS_FONT, yaxis=_AXIS_FONT,
            )
            fig_bar_cmd.update_traces(textposition="outside", marker_line_width=0, width=0.55)
            fig_bar_cmd.update_xaxes(showgrid=False)
            fig_bar_cmd.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_bar_cmd, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Click-to-Detail panel ──────────────────────────────────────
        sel_points = (map_event.selection.points
                      if hasattr(map_event, "selection")
                      and hasattr(map_event.selection, "points")
                      else [])
        _clicked_ids = []
        for pt in sel_points:
            cd = pt.get("customdata")
            if cd is not None:
                if isinstance(cd, list):
                    _clicked_ids.extend(int(x) for x in cd)
                else:
                    _clicked_ids.append(int(cd))

        if _clicked_ids:
            _sel_rows = cum_df.loc[cum_df.index.isin(_clicked_ids)]
            if not _sel_rows.empty:
                st.markdown(
                    '<div class="section-header">Selected Patient Details</div>',
                    unsafe_allow_html=True)
                for _, _sr in _sel_rows.iterrows():
                    _ct_disp = CASE_DISPLAY.get(_sr["case_type"], _sr["case_type"])
                    _flag_badge = (
                        ' <span class="status-badge outbreak">FLAGGED</span>'
                        if _sr["Novelty_Flag"] else ''
                    )
                    _clust_name = _sr.get("_label_str", "")
                    _symp = _sr.get("unmapped_text", "")
                    _terms = _sr.get("unmapped_anomalies", [])
                    _quotes = _sr.get("unmapped_quotes", [])
                    _detail_html = (
                        f'<div class="card" style="margin-bottom:0.6rem">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'align-items:center;margin-bottom:0.4rem">'
                        f'<span style="font-family:Source Code Pro,monospace;'
                        f'font-size:0.85rem;font-weight:700;color:#0F172A">'
                        f'{_sr.get("hospital","")}  |  Day {_sr["day"]}'
                        f'</span>{_flag_badge}</div>'
                        f'<div style="font-size:0.82rem;color:#64748B">'
                        f'<b>Case Type:</b> {_ct_disp}  |  '
                        f'<b>Cluster:</b> {_clust_name}</div>'
                    )
                    if _symp:
                        _detail_html += (
                            f'<div style="font-size:0.80rem;color:#64748B;'
                            f'margin-top:0.3rem">'
                            f'<b>Novel Symptoms:</b> {_symp}</div>'
                        )
                    if _terms:
                        for _t, _q in zip(_terms, _quotes):
                            _detail_html += (
                                f'<div class="anomaly-item" style="margin-top:0.3rem">'
                                f'<div class="term">{_t}</div>'
                                f'<div class="quote">"{_q}"</div></div>'
                            )
                    _detail_html += '</div>'
                    st.markdown(_detail_html, unsafe_allow_html=True)

    # ···· Tab 2: UMAP Projection — formatted like plot_clusters
    with tab_umap:

        fig_umap = go.Figure()
        unique_labels = sorted(cum_df["_cluster"].unique())
        color_idx = 0

        # Draw each cluster as its own trace (like plot_clusters)
        for lbl in unique_labels:
            mask = cum_df["_cluster"] == lbl
            subset = cum_df[mask]
            count = len(subset)

            if lbl == -1:
                color = "#CCCCCC"
                legend_name = f"Noise ({count})"
                alpha = 0.4
                sz = 5
            elif outbreak_label is not None and lbl == outbreak_label:
                color = "#DC2626"
                legend_name = f"C{lbl} -- {_pretty(lbl)} [{count}]"
                alpha = 0.8
                sz = 9
            else:
                color = CLUSTER_PALETTE[color_idx % len(CLUSTER_PALETTE)]
                color_idx += 1
                legend_name = f"C{lbl} -- {_pretty(lbl)} ({count})"
                alpha = 0.7
                sz = 7

            fig_umap.add_trace(go.Scatter(
                x=subset["umap_1"], y=subset["umap_2"],
                mode="markers",
                marker=dict(
                    size=sz, color=color, opacity=alpha,
                    line=dict(width=0.3, color="white"),
                ),
                name=legend_name,
                hovertext=[
                    f"<b>{r['case_type']}</b><br>{r.get('hospital','')}<br>Day {r['day']}"
                    for _, r in subset.iterrows()
                ],
                hoverinfo="text",
            ))

        title_text = (f"UMAP + HDBSCAN (EOM)  |  {_window_label}  |  {_window_dates}  |  "
                      f"{n_real_clusters} cluster{'s' if n_real_clusters != 1 else ''}")

        fig_umap.update_layout(
            **_PLOTLY,
            height=600,
            title=dict(text=title_text, font=dict(size=14, color="#0F172A"), x=0.5),
            xaxis=dict(title="UMAP 1", showgrid=True, gridcolor="#F1F5F9",
                       zeroline=False, tickfont=dict(color="#0F172A"),
                       title_font=dict(color="#0F172A")),
            yaxis=dict(title="UMAP 2", showgrid=True, gridcolor="#F1F5F9",
                       zeroline=False, tickfont=dict(color="#0F172A"),
                       title_font=dict(color="#0F172A")),
            legend=dict(
                yanchor="top", y=0.98, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#E2E8F0", borderwidth=1,
                font=dict(size=10),
            ),
        )
        st.plotly_chart(fig_umap, width="stretch")

        # Cluster Metrics
        with st.expander("Cluster Metrics"):
            sil_str = f"{res['silhouette']:.4f}" if res['silhouette'] else "N/A"
            if outbreak_label is not None:
                outbreak_str = f"Cluster {outbreak_label} ({_pretty(outbreak_label)})"
            else:
                outbreak_str = "Not detected"
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Algorithm | UMAP + HDBSCAN (EOM) |
| Clusters detected | {res['n_clusters']} |
| Noise points | {res['n_noise']} |
| Silhouette score | {sil_str} |
| Outbreak cluster | {outbreak_str} |
| Novel virus cases | {n_novel} |
| Flagged cases | {n_flagged_cum} |
""")

    # ···· Tab 3: Case Data ·············································
    with tab_data:
        cluster_options = ["All"]
        if outbreak_label is not None:
            cluster_options.insert(0, "Outbreak Cluster")
        cluster_options.append("Noise")
        for lbl in sorted(set(cum_df["_cluster"].unique()) - {-1}):
            if lbl != outbreak_label:
                cluster_options.append(f"Cluster {lbl}")

        filter_choice = st.selectbox("Filter by cluster", cluster_options,
                                     label_visibility="collapsed")

        if filter_choice == "Outbreak Cluster" and outbreak_label is not None:
            show_df = cum_df[cum_df["_cluster"] == outbreak_label]
        elif filter_choice == "Noise":
            show_df = cum_df[cum_df["_cluster"] == -1]
        elif filter_choice.startswith("Cluster "):
            lbl_num = int(filter_choice.split()[-1])
            show_df = cum_df[cum_df["_cluster"] == lbl_num]
        else:
            show_df = cum_df

        display_cols = [
            "hospital", "day", "case_type",
            "_label_str", "Novelty_Flag", "unmapped_text",
        ]
        rename_map = {
            "hospital": "Hospital", "day": "Day",
            "case_type": "Case Type", "_label_str": "Cluster",
            "Novelty_Flag": "Flagged", "unmapped_text": "Novel Symptoms",
        }
        table_df = show_df[display_cols].rename(columns=rename_map).reset_index(drop=True)

        st.dataframe(
            table_df, width="stretch", height=420,
            column_config={
                "Novel Symptoms": st.column_config.TextColumn(width="large"),
            },
        )

        # Flagged anomaly details
        flagged_in_view = show_df[show_df["Novelty_Flag"]]
        if not flagged_in_view.empty:
            with st.expander(f"Flagged Cases  --  {len(flagged_in_view)} detected"):
                for _, row in flagged_in_view.iterrows():
                    terms  = row["unmapped_anomalies"]
                    quotes = row["unmapped_quotes"]
                    st.markdown(
                        f"**{row['hospital']}**  |  Day {row['day']}  |  "
                        f"{CASE_DISPLAY.get(row['case_type'], row['case_type'])}"
                    )
                    for t, q in zip(terms, quotes):
                        st.markdown(
                            f'<div class="anomaly-item">'
                            f'<div class="term">{t}</div>'
                            f'<div class="quote">"{q}"</div>'
                            f'</div>',
                            unsafe_allow_html=True)
                    st.markdown("---")

    # ···· Tab 4: Symptom Analysis ······································
    with tab_symp:
        _symptom_names = get_symptom_names(str(FEWSHOT_DIR))

        # Cluster selector
        _symp_cluster_opts = []
        for lbl in sorted(set(cum_df["_cluster"].unique()) - {-1}):
            _symp_cluster_opts.append(f"C{lbl} -- {_pretty(lbl)}")
        _symp_cluster_opts.insert(0, "All Patients")

        _symp_choice = st.selectbox("Select Cluster", _symp_cluster_opts,
                                     key="symp_cluster_sel")

        if _symp_choice == "All Patients":
            _symp_idx = cum_df.index.tolist()
        else:
            _symp_lbl = int(_symp_choice.split("C")[1].split(" ")[0])
            _symp_idx = cum_df[cum_df["_cluster"] == _symp_lbl].index.tolist()

        # Build prevalence from categorical matrix
        _symp_rows = cat[_symp_idx]  # (n_selected, 70)
        _symp_present = (_symp_rows >= 1).sum(axis=0)  # count of patients with symptom
        _symp_n = len(_symp_idx) if _symp_idx else 1
        _symp_pct = (_symp_present / _symp_n * 100).round(1)

        # Also compute overall prevalence for comparison
        _all_rows = cat[cum_df.index.tolist()]
        _all_present = (_all_rows >= 1).sum(axis=0)
        _all_n = len(cum_df)
        _all_pct = (_all_present / _all_n * 100).round(1)

        # Build DataFrame
        _symp_df = pd.DataFrame({
            "Symptom": _symptom_names,
            "Selected (%)": _symp_pct,
            "Overall (%)": _all_pct,
        }).sort_values("Selected (%)", ascending=True).tail(20)

        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="section-header">Symptom Prevalence  |  '
            f'{_symp_choice}  ({len(_symp_idx)} patients)</div>',
            unsafe_allow_html=True)

        fig_symp = go.Figure()
        fig_symp.add_trace(go.Bar(
            y=_symp_df["Symptom"], x=_symp_df["Overall (%)"],
            orientation="h", name="Overall",
            marker=dict(color="#CBD5E1", line=dict(width=0)),
            opacity=0.5,
        ))
        fig_symp.add_trace(go.Bar(
            y=_symp_df["Symptom"], x=_symp_df["Selected (%)"],
            orientation="h", name=_symp_choice,
            marker=dict(color="#2563EB", line=dict(width=0)),
            text=[f"{v:.0f}%" for v in _symp_df["Selected (%)"]],
            textposition="outside",
        ))
        _symp_layout = {k: v for k, v in _PLOTLY.items() if k != "margin"}
        fig_symp.update_layout(
            **_symp_layout,
            height=max(500, len(_symp_df) * 28),
            barmode="overlay",
            xaxis=dict(title="Prevalence (%)", range=[0, 105],
                       showgrid=True, gridcolor="#F1F5F9",
                       tickfont=dict(color="#0F172A", size=12),
                       title_font=dict(color="#0F172A", size=13)),
            yaxis=dict(title="", tickfont=dict(color="#0F172A", size=11),
                       automargin=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="center", x=0.5,
                        font=dict(size=12, color="#0F172A")),
            margin=dict(l=250, r=40, t=30, b=50),
        )
        st.plotly_chart(fig_symp, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

        # Detailed table
        with st.expander("Full Symptom Table"):
            _full_symp = pd.DataFrame({
                "Symptom": _symptom_names,
                "Count": _symp_present,
                "Prevalence (%)": _symp_pct,
            }).sort_values("Prevalence (%)", ascending=False).reset_index(drop=True)
            st.dataframe(_full_symp, width="stretch", height=400)

    # ···· Tab 5: R0 Forecast ··········································
    with tab_r0:
        st.markdown(
            '<div style="font-style:italic;font-size:0.80rem;color:#64748B;'
            'margin-bottom:0.8rem;">R<sub>0</sub> estimated from 7-day novel-virus '
            'incidence via exponential growth fitting. Small sample sizes may yield '
            'unstable estimates.</div>', unsafe_allow_html=True)

        # ── Controls row ──────────────────────────────────────────────
        _r0c1, _r0c2 = st.columns(2)
        with _r0c1:
            _gen_time = st.slider("Generation Time (days)", 3, 10, value=5,
                                  key="r0_gen_time")
        with _r0c2:
            _pop_size = st.select_slider(
                "Projected Population",
                options=[50_000, 100_000, 250_000, 500_000, 1_000_000],
                value=250_000, key="r0_pop",
                format_func=lambda x: f"{x:,}")

        # ── Aggregate novel_virus incidence by day ────────────────────
        _nv = df[df["case_type"] == "novel_virus"].groupby("day").size().reindex(
            range(1, NUM_DAYS + 1), fill_value=0)
        _days_arr = np.array(_nv.index)
        _counts_arr = np.array(_nv.values, dtype=float)

        # Only fit on days with counts > 0
        _pos_mask = _counts_arr > 0
        _r0_val = None
        _lam = None
        _doubling = None

        if _pos_mask.sum() >= 2:
            _fit = np.polyfit(_days_arr[_pos_mask],
                              np.log(_counts_arr[_pos_mask]), 1)
            _lam = _fit[0]  # growth rate
            _r0_val = float(np.exp(_lam * _gen_time))
            if _lam > 0:
                _doubling = float(np.log(2) / _lam)
            else:
                _doubling = None
        else:
            _lam = 0.0

        # ── KPI row ──────────────────────────────────────────────────
        _k1, _k2, _k3, _k4 = st.columns(4)
        with _k1:
            st.markdown(
                f'<div class="card"><h3>R<sub>0</sub> Estimate</h3>'
                f'<div class="big-num" style="color:{"#DC2626" if _r0_val and _r0_val > 1 else "#16A34A"}">'
                f'{_r0_val:.2f}</div>'
                f'<div class="sub">T<sub>g</sub> = {_gen_time} days</div></div>'
                if _r0_val else
                '<div class="card"><h3>R<sub>0</sub> Estimate</h3>'
                '<div class="big-num">N/A</div>'
                '<div class="sub">Insufficient data</div></div>',
                unsafe_allow_html=True)
        with _k2:
            _doub_str = f"{_doubling:.1f} days" if _doubling and _doubling > 0 else "N/A"
            st.markdown(
                f'<div class="card"><h3>Doubling Time</h3>'
                f'<div class="big-num">{_doub_str}</div>'
                f'<div class="sub">Cases double every</div></div>',
                unsafe_allow_html=True)
        with _k3:
            _total_nv = int(_counts_arr.sum())
            st.markdown(
                f'<div class="card"><h3>Total Novel Cases</h3>'
                f'<div class="big-num">{_total_nv}</div>'
                f'<div class="sub">Days 1 \u2013 {NUM_DAYS}</div></div>',
                unsafe_allow_html=True)
        with _k4:
            _growth_str = f"{_lam:.3f}" if _lam else "N/A"
            st.markdown(
                f'<div class="card"><h3>Growth Rate (\u03bb)</h3>'
                f'<div class="big-num">{_growth_str}</div>'
                f'<div class="sub">Exponential fit</div></div>',
                unsafe_allow_html=True)

        # ── Chart A: Incidence curve + fitted exponential ─────────────
        _r0_ch1, _r0_ch2 = st.columns(2)
        with _r0_ch1:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Daily Incidence  |  Novel Virus</div>',
                        unsafe_allow_html=True)
            fig_inc = go.Figure()
            fig_inc.add_trace(go.Bar(
                x=[DATE_LABELS.get(d, f"Day {d}") for d in _days_arr],
                y=_counts_arr, name="Observed",
                marker=dict(color="#2563EB", line=dict(width=0)),
                text=[int(c) for c in _counts_arr], textposition="outside",
            ))
            if _lam and _pos_mask.sum() >= 2:
                _fit_x = np.linspace(1, NUM_DAYS, 50)
                _fit_y = np.exp(_fit[1]) * np.exp(_lam * _fit_x)
                fig_inc.add_trace(go.Scatter(
                    x=[DATE_LABELS.get(int(round(d)), f"Day {int(round(d))}") for d in _fit_x],
                    y=_fit_y, mode="lines", name="Exp. Fit",
                    line=dict(color="#DC2626", dash="dash", width=2),
                ))
            fig_inc.update_layout(**_PLOTLY, height=400,
                                  xaxis_title="Day", yaxis_title="Cases",
                                  xaxis=_AXIS_FONT, yaxis=_AXIS_FONT,
                                  showlegend=True,
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="center", x=0.5,
                                              font=dict(size=12, color="#0F172A")))
            fig_inc.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_inc, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Chart B: SIR projection ──────────────────────────────────
        with _r0_ch2:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">SIR Projection  |  30-Day Forward</div>',
                        unsafe_allow_html=True)

            _N = _pop_size
            _gamma = 1.0 / _gen_time
            _beta = (_r0_val * _gamma) if _r0_val else 0.0
            _I0 = max(float(_counts_arr[0]), 1.0)
            _S0 = _N - _I0
            _R0_init = 0.0

            _sir_days = 30
            _S = np.zeros(_sir_days)
            _I = np.zeros(_sir_days)
            _R = np.zeros(_sir_days)
            _S[0], _I[0], _R[0] = _S0, _I0, _R0_init

            for t in range(1, _sir_days):
                _dS = -_beta * _S[t-1] * _I[t-1] / _N
                _dI = _beta * _S[t-1] * _I[t-1] / _N - _gamma * _I[t-1]
                _dR = _gamma * _I[t-1]
                _S[t] = max(_S[t-1] + _dS, 0)
                _I[t] = max(_I[t-1] + _dI, 0)
                _R[t] = min(_R[t-1] + _dR, _N)

            _peak_day = int(np.argmax(_I)) + 1
            _peak_inf = int(np.max(_I))

            fig_sir = go.Figure()
            fig_sir.add_trace(go.Scatter(
                x=list(range(1, _sir_days + 1)), y=_S,
                mode="lines", name="Susceptible",
                line=dict(color="#94A3B8", width=2),
            ))
            fig_sir.add_trace(go.Scatter(
                x=list(range(1, _sir_days + 1)), y=_I,
                mode="lines", name="Infected",
                line=dict(color="#DC2626", width=2.5),
                fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
            ))
            fig_sir.add_trace(go.Scatter(
                x=list(range(1, _sir_days + 1)), y=_R,
                mode="lines", name="Recovered",
                line=dict(color="#16A34A", width=2),
            ))
            fig_sir.update_layout(**_PLOTLY, height=400,
                                  xaxis_title="Day", yaxis_title="Population",
                                  xaxis=_AXIS_FONT, yaxis=_AXIS_FONT,
                                  showlegend=True,
                                  legend=dict(orientation="h", yanchor="bottom",
                                              y=1.02, xanchor="center", x=0.5,
                                              font=dict(size=12, color="#0F172A")))
            fig_sir.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_sir, width="stretch")

            # Peak info under chart
            st.markdown(
                f'<div style="text-align:center;font-size:0.85rem;color:#64748B;">'
                f'Projected peak: <b>Day {_peak_day}</b> with '
                f'<b>{_peak_inf:,}</b> concurrent infections  '
                f'(Pop. {_N:,})</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Per-hub R0 table ──────────────────────────────────────────
        with st.expander("R0 by Hospital Hub"):
            _hub_r0_rows = []
            for h in SC_HOSPITAL_HUBS:
                _h_nv = df[(df["case_type"] == "novel_virus") & (df["hub_id"] == h["hub_id"])
                          ].groupby("day").size().reindex(range(1, NUM_DAYS + 1), fill_value=0)
                _h_arr = np.array(_h_nv.values, dtype=float)
                _h_pos = _h_arr > 0
                if _h_pos.sum() >= 2:
                    _h_fit = np.polyfit(np.array(_h_nv.index)[_h_pos],
                                        np.log(_h_arr[_h_pos]), 1)
                    _h_lam = _h_fit[0]
                    _h_r0 = float(np.exp(_h_lam * _gen_time))
                    _h_doub = float(np.log(2) / _h_lam) if _h_lam > 0 else None
                else:
                    _h_r0, _h_lam, _h_doub = None, None, None
                _hub_r0_rows.append({
                    "Hospital": h["name"],
                    "City": h["city"],
                    "Total NV Cases": int(_h_arr.sum()),
                    "R0": round(_h_r0, 2) if _h_r0 else "N/A",
                    "Growth Rate": round(_h_lam, 4) if _h_lam else "N/A",
                    "Doubling (days)": round(_h_doub, 1) if _h_doub else "N/A",
                })
            st.dataframe(pd.DataFrame(_hub_r0_rows), width="stretch", hide_index=True)

    # ···· Tab 6: Early Warning — Novelty Flag Trend ···················
    with tab_ew:
        st.markdown(
            '<div style="font-style:italic;font-size:0.80rem;color:#64748B;'
            'margin-bottom:0.8rem;">The Early Warning Detection System tracks '
            'novelty flags (unmapped symptoms detected by the LLM extraction pipeline) '
            'over time. These flags provide an <b>earlier signal</b> than cluster-based '
            'outbreak detection — novel symptoms appear before UMAP+HDBSCAN can form '
            'a distinct outbreak cluster.</div>', unsafe_allow_html=True)

        # ── Compute novelty flag data across ALL 7 days ──────────────
        _ew_cum_flags = []       # cumulative flag count per day
        _ew_daily_flags = []     # new flags per day
        _ew_cum_patients = []    # cumulative flagged patients per day
        _ew_daily_patients = []  # new flagged patients per day
        _ew_outbreak_day = None  # first day outbreak is detected by HDBSCAN

        # Per-hub daily breakdown
        _hub_names_map = {h["hub_id"]: h["name"] for h in SC_HOSPITAL_HUBS}
        _ew_hub_daily = {h["hub_id"]: [] for h in SC_HOSPITAL_HUBS}

        _prev_flag_count = 0
        _prev_patient_count = 0
        for _ew_d in range(1, NUM_DAYS + 1):
            _ew_res = clustering[_ew_d]
            _ew_df = df.iloc[_ew_res["idx"]].copy()
            _ew_total_flags = int(_ew_df["Novelty_Flag"].sum())
            _ew_total_patients = int(_ew_df["Novelty_Flag"].sum())

            _ew_cum_flags.append(_ew_total_flags)
            _ew_daily_flags.append(max(0, _ew_total_flags - _prev_flag_count))
            _ew_cum_patients.append(_ew_total_patients)
            _ew_daily_patients.append(max(0, _ew_total_patients - _prev_patient_count))
            _prev_flag_count = _ew_total_flags
            _prev_patient_count = _ew_total_patients

            # Per-hub daily new flags
            _ew_day_only = _ew_df[_ew_df["day"] == _ew_d]
            for h in SC_HOSPITAL_HUBS:
                _h_day = _ew_day_only[_ew_day_only["hub_id"] == h["hub_id"]]
                _ew_hub_daily[h["hub_id"]].append(int(_h_day["Novelty_Flag"].sum()))

            # Check if HDBSCAN detected outbreak on this day
            if _ew_outbreak_day is None and _ew_res["outbreak_label"] is not None:
                _ew_outbreak_day = _ew_d

        # First day with any novelty flags
        _ew_first_flag_day = None
        for _fd, _fc in enumerate(_ew_daily_flags, start=1):
            if _fc > 0:
                _ew_first_flag_day = _fd
                break

        # Lead time: days between first flag and cluster detection
        _ew_lead_time = None
        if _ew_first_flag_day is not None and _ew_outbreak_day is not None:
            _ew_lead_time = _ew_outbreak_day - _ew_first_flag_day

        # ── KPI row ──────────────────────────────────────────────────
        _ew_k1, _ew_k2, _ew_k3, _ew_k4 = st.columns(4)
        with _ew_k1:
            _first_str = f"Day {_ew_first_flag_day}" if _ew_first_flag_day else "None"
            _first_date = DATE_LABELS.get(_ew_first_flag_day, "") if _ew_first_flag_day else ""
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">First Novel Flag</div>'
                f'<div class="kpi-value" style="color:#DC2626">{_first_str}</div>'
                f'<div class="kpi-sub">{_first_date}</div>'
                f'</div>', unsafe_allow_html=True)
        with _ew_k2:
            _outbreak_str = f"Day {_ew_outbreak_day}" if _ew_outbreak_day else "Not yet"
            _outbreak_date = DATE_LABELS.get(_ew_outbreak_day, "") if _ew_outbreak_day else ""
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">UMAP Cluster Detection</div>'
                f'<div class="kpi-value">{_outbreak_str}</div>'
                f'<div class="kpi-sub">{_outbreak_date}</div>'
                f'</div>', unsafe_allow_html=True)
        with _ew_k3:
            _lead_str = f"{_ew_lead_time} days" if _ew_lead_time is not None else "N/A"
            _lead_color = "#16A34A" if _ew_lead_time and _ew_lead_time > 0 else "#0F172A"
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Early Warning Lead Time</div>'
                f'<div class="kpi-value" style="color:{_lead_color}">{_lead_str}</div>'
                f'<div class="kpi-sub">Flags before cluster detection</div>'
                f'</div>', unsafe_allow_html=True)
        with _ew_k4:
            _cum_flags_now = _ew_cum_flags[day - 1] if day <= len(_ew_cum_flags) else 0
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">Cumulative Flags (Day {day})</div>'
                f'<div class="kpi-value" style="color:{"#DC2626" if _cum_flags_now else "#0F172A"}">{_cum_flags_now}</div>'
                f'<div class="kpi-sub">Through {DATE_LABELS.get(day, "")}</div>'
                f'</div>', unsafe_allow_html=True)

        # ── Chart A: Cumulative Novelty Flag Trend (LINE GRAPH) ──────
        _ew_ch1, _ew_ch2 = st.columns(2, gap="medium")
        with _ew_ch1:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">Cumulative Novelty Flags Over Time</div>',
                unsafe_allow_html=True)

            _ew_day_labels = [DATE_LABELS.get(d, f"Day {d}") for d in range(1, NUM_DAYS + 1)]
            fig_ew_line = go.Figure()
            fig_ew_line.add_trace(go.Scatter(
                x=_ew_day_labels,
                y=_ew_cum_flags,
                mode="lines+markers",
                name="Cumulative Novelty Flags",
                line=dict(color="#DC2626", width=3),
                marker=dict(size=10, color="#DC2626",
                            line=dict(width=2, color="white")),
                fill="tozeroy",
                fillcolor="rgba(220,38,38,0.08)",
            ))

            # Mark the current day
            if day <= NUM_DAYS:
                fig_ew_line.add_trace(go.Scatter(
                    x=[_ew_day_labels[day - 1]],
                    y=[_ew_cum_flags[day - 1]],
                    mode="markers",
                    marker=dict(size=16, color="#2563EB",
                                line=dict(width=3, color="white"),
                                symbol="circle"),
                    name=f"Current (Day {day})",
                    showlegend=True,
                ))

            # Annotate outbreak detection day
            if _ew_outbreak_day is not None:
                fig_ew_line.add_vline(
                    x=_ew_day_labels[_ew_outbreak_day - 1],
                    line_dash="dash", line_color="#2563EB", line_width=2,
                )
                fig_ew_line.add_annotation(
                    x=_ew_day_labels[_ew_outbreak_day - 1],
                    y=max(_ew_cum_flags) * 0.95,
                    text="UMAP Detects<br>Outbreak",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowcolor="#2563EB",
                    font=dict(size=11, color="#2563EB",
                              family="Source Code Pro, monospace"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#2563EB", borderwidth=1,
                    borderpad=4,
                )

            # Annotate first flag day
            if _ew_first_flag_day is not None:
                fig_ew_line.add_vline(
                    x=_ew_day_labels[_ew_first_flag_day - 1],
                    line_dash="dot", line_color="#DC2626", line_width=2,
                )
                fig_ew_line.add_annotation(
                    x=_ew_day_labels[_ew_first_flag_day - 1],
                    y=max(_ew_cum_flags) * 0.7,
                    text="First Novel<br>Flag",
                    showarrow=True, arrowhead=2, arrowsize=1,
                    arrowcolor="#DC2626",
                    font=dict(size=11, color="#DC2626",
                              family="Source Code Pro, monospace"),
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#DC2626", borderwidth=1,
                    borderpad=4,
                )

            fig_ew_line.update_layout(
                **_PLOTLY, height=420,
                xaxis_title="Day", yaxis_title="Cumulative Novelty Flags",
                xaxis=_AXIS_FONT, yaxis=_AXIS_FONT,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5,
                            font=dict(size=11, color="#0F172A")),
            )
            fig_ew_line.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_ew_line, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Chart B: Daily New Flags by Hospital Hub (stacked bar) ───
        with _ew_ch2:
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header">Daily New Novelty Flags by Hospital</div>',
                unsafe_allow_html=True)

            _hub_colors = ["#2563EB", "#F97316", "#16A34A", "#DC2626",
                           "#8B5CF6", "#EC4899", "#14B8A6"]
            fig_ew_bar = go.Figure()
            for _hi, h in enumerate(SC_HOSPITAL_HUBS):
                fig_ew_bar.add_trace(go.Bar(
                    x=_ew_day_labels,
                    y=_ew_hub_daily[h["hub_id"]],
                    name=h["name"].split(" ")[0],  # Short name
                    marker=dict(color=_hub_colors[_hi % len(_hub_colors)]),
                ))
            fig_ew_bar.update_layout(
                **_PLOTLY, height=420,
                barmode="stack",
                xaxis_title="Day", yaxis_title="New Flags",
                xaxis=_AXIS_FONT, yaxis=_AXIS_FONT,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5,
                            font=dict(size=10, color="#0F172A")),
            )
            fig_ew_bar.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_ew_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Early Warning Signal explanation ──────────────────────────
        if _ew_lead_time and _ew_lead_time > 0:
            st.markdown(
                f'<div class="anomaly-banner alert">'
                f'<span class="anomaly-icon red">!</span>'
                f'<div>'
                f'<span class="anomaly-count">{_ew_lead_time}-Day Early Warning</span><br>'
                f'<span class="anomaly-label">Novelty flags first appeared on '
                f'Day {_ew_first_flag_day} ({DATE_LABELS.get(_ew_first_flag_day, "")}), '
                f'but UMAP+HDBSCAN did not detect an outbreak cluster until '
                f'Day {_ew_outbreak_day} ({DATE_LABELS.get(_ew_outbreak_day, "")}). '
                f'The LLM extraction pipeline provided a {_ew_lead_time}-day head start.</span>'
                f'</div></div>', unsafe_allow_html=True)
        elif _ew_first_flag_day and _ew_outbreak_day is None:
            st.markdown(
                f'<div class="anomaly-banner alert">'
                f'<span class="anomaly-icon red">!</span>'
                f'<div>'
                f'<span class="anomaly-count">Flags Detected — No Cluster Yet</span><br>'
                f'<span class="anomaly-label">Novelty flags appeared on '
                f'Day {_ew_first_flag_day}, but UMAP+HDBSCAN has not yet formed '
                f'an outbreak cluster. The flag-based system is providing an early signal.</span>'
                f'</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="anomaly-banner clear">'
                '<span class="anomaly-icon green">&#10003;</span>'
                '<div>'
                '<span class="anomaly-count" style="color:#16A34A">No Flags</span><br>'
                '<span class="anomaly-label">No novelty flags detected yet in the current window.</span>'
                '</div></div>', unsafe_allow_html=True)

        # ── Novelty Flag Detail Table ─────────────────────────────────
        st.markdown(
            '<div class="section-header">Novelty Flag Details</div>',
            unsafe_allow_html=True)

        _ew_flagged = cum_df[cum_df["Novelty_Flag"]].copy()
        if not _ew_flagged.empty:
            _ew_flagged_sorted = _ew_flagged.sort_values("day", ascending=False)
            _ew_detail_rows = []
            for _, _fr in _ew_flagged_sorted.iterrows():
                _terms = _fr.get("unmapped_anomalies", [])
                _quotes = _fr.get("unmapped_quotes", [])
                for _ti, _term in enumerate(_terms):
                    _quote = _quotes[_ti] if _ti < len(_quotes) else ""
                    _ew_detail_rows.append({
                        "Day": _fr["day"],
                        "Date": DATE_LABELS.get(_fr["day"], ""),
                        "Hospital": _fr.get("hospital", ""),
                        "Case Type": CASE_DISPLAY.get(_fr["case_type"], _fr["case_type"]),
                        "Novel Symptom": _term,
                        "Patient Quote": _quote,
                    })
            if _ew_detail_rows:
                _ew_detail_df = pd.DataFrame(_ew_detail_rows)
                st.dataframe(
                    _ew_detail_df, width="stretch", height=400,
                    column_config={
                        "Patient Quote": st.column_config.TextColumn(width="large"),
                    },
                )
            else:
                st.info("No novelty symptoms with terms found in the current window.")
        else:
            st.info("No novelty-flagged patients in the current window.")


# ═══════════════════════════════════════════════════════════════════════════
#  VIEW 3 — CONVERSATION DEMONSTRATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

elif st.session_state.current_view == "conv_demo":
    st.markdown(
        "<h1 style='font-size:1.6rem;font-weight:700;margin-bottom:0'>"
        "Conversation Demonstration Pipeline</h1>"
        "<p style='color:#64748B;margin-top:0;font-size:0.9rem'>"
        "Generate a synthetic doctor–patient transcript using the v3 pipeline  "
        "(GPT-4o, one-shot generation)</p>",
        unsafe_allow_html=True,
    )

    # ── Add v3 generation directory to sys.path ──────────────────────
    _v3_dir = str(GENERATION_V3_DIR)
    if _v3_dir not in sys.path:
        sys.path.insert(0, _v3_dir)

    from symptom_library_v3 import (  # noqa: E402
        SYMPTOM_LIBRARY, FEASIBILITY_CLUSTERS, CASE_TYPES,
        sample_symptoms, NOVEL_VIRUS_HALLMARKS,
        FLU_FEASIBILITY_CLUSTERS, DIFFERENTIAL_CLUSTERS,
    )
    from scenario_builder_v3 import _extract_ccda_noise, _extract_demographics  # noqa: E402
    from generate_transcript_v3 import build_prompt, generate_transcript  # noqa: E402

    st.markdown("---")

    # ── Step 1: Case type + symptom sampling ─────────────────────────
    st.markdown("### Step 1 — Configure Scenario")
    _demo_c1, _demo_c2 = st.columns(2, gap="medium")

    # ── Build case-type-specific symptom pools ────────────────────────
    _COVID_CLUSTERS = ["upper_respiratory", "systemic", "sensory", "respiratory_red_flag"]

    with _demo_c1:
        _case_type = st.selectbox(
            "Case Type",
            ["novel_virus", "flu_like", "differential", "healthy"],
            index=0,
            key="demo_case_type",
        )
        _diff_sys = None
        if _case_type == "differential":
            _diff_sys = st.selectbox(
                "Differential System",
                sorted(DIFFERENTIAL_CLUSTERS),
                key="demo_diff_sys",
            )

        # Determine symptom pool based on case type (mirrors generator logic)
        if _case_type == "novel_virus":
            _available_pool = sorted({s for c in _COVID_CLUSTERS for s in FEASIBILITY_CLUSTERS[c]} | set(NOVEL_VIRUS_HALLMARKS))
            _max_present = min(10, len(_available_pool))
        elif _case_type == "flu_like":
            _available_pool = sorted({s for v in FLU_FEASIBILITY_CLUSTERS.values() for s in v})
            _max_present = min(10, len(_available_pool))
        elif _case_type == "differential":
            _sys_key = _diff_sys or DIFFERENTIAL_CLUSTERS[0]
            _available_pool = sorted(FEASIBILITY_CLUSTERS.get(_sys_key, []))
            _max_present = min(10, len(_available_pool))
        else:  # healthy
            _available_pool = []
            _max_present = 0

        _seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42, key="demo_seed")
        if _case_type != "healthy" and _max_present > 3:
            _n_present = st.slider("Present symptom count", 3, _max_present, min(5, _max_present), key="demo_n_present")
        elif _case_type != "healthy" and _max_present >= 1:
            _n_present = st.slider("Present symptom count", 1, _max_present, min(3, _max_present), key="demo_n_present")
        else:
            _n_present = 0
        _n_negated = st.slider("Negated symptom count", 1, 6, 3, key="demo_n_negated")

    with _demo_c2:
        # Novel-virus hallmark selection
        if _case_type == "novel_virus":
            st.markdown(
                '<div style="font-size:0.82rem;color:#64748B;margin-bottom:0.3rem;">'
                'Select which novel-virus hallmarks to include (auto-sampling picks 2-4):</div>',
                unsafe_allow_html=True)
            _hallmark_sel = st.multiselect(
                "Novel Virus Hallmarks",
                NOVEL_VIRUS_HALLMARKS,
                default=NOVEL_VIRUS_HALLMARKS[:3],
                key="demo_hallmarks",
            )

        if _case_type != "healthy" and _available_pool:
            _selected_symptoms = st.multiselect(
                "Override Present Symptoms (leave empty to auto-sample)",
                _available_pool,
                default=[],
                key="demo_symptoms",
            )
        else:
            _selected_symptoms = []

    # ── Sample symptoms button ───────────────────────────────────────
    if st.button("Sample Symptoms", key="demo_sample_btn"):
        import random as _rnd
        # Auto-increment seed on each click so repeated presses give new samples
        _effective_seed = _seed + st.session_state.get("_sample_click_count", 0)
        st.session_state["_sample_click_count"] = st.session_state.get("_sample_click_count", 0) + 1
        _rng = _rnd.Random(_effective_seed)
        _present, _chief, _negated, _diff_out = sample_symptoms(
            _rng,
            case_type=_case_type,
            n_present_min=min(_n_present, _max_present) if _case_type != "healthy" else 0,
            n_present_max=min(_n_present, _max_present) if _case_type != "healthy" else 0,
            n_negated_min=_n_negated,
            n_negated_max=_n_negated,
            differential_system=_diff_sys,
        )
        st.session_state["demo_sampled"] = {
            "present": _present,
            "chief": _chief,
            "negated": _negated,
            "diff_system": _diff_out,
        }

    # Display sampled symptoms
    if "demo_sampled" in st.session_state:
        _s = st.session_state["demo_sampled"]
        _sc1, _sc2 = st.columns(2, gap="medium")
        with _sc1:
            st.markdown(
                '<div class="card">'
                '<h3>Present Symptoms</h3>'
                '<div style="font-size:0.85rem;color:#0F172A">'
                + ''.join(f'<div style="padding:2px 0">• {sym}</div>' for sym in _s["present"])
                + '</div>'
                f'<div style="margin-top:0.5rem;font-size:0.82rem;color:#64748B">'
                f'<b>Chief Complaint:</b> {_s["chief"]}</div>'
                '</div>',
                unsafe_allow_html=True)
        with _sc2:
            _neg_html = ''.join(f'<div style="padding:2px 0">• {sym}</div>' for sym in _s["negated"]) if _s["negated"] else '<div style="color:#94A3B8">None</div>'
            _diff_html = f'<div style="margin-top:0.5rem;font-size:0.82rem;color:#64748B"><b>Differential System:</b> {_s["diff_system"]}</div>' if _s["diff_system"] else ''
            st.markdown(
                '<div class="card">'
                '<h3>Negated Symptoms</h3>'
                f'<div style="font-size:0.85rem;color:#0F172A">{_neg_html}</div>'
                f'{_diff_html}'
                '</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 2: CCDA noise source ────────────────────────────────────
    st.markdown("### Step 2 — Patient Background (CCDA)")

    # Search for ground-truth JSON files
    _gt_files = []
    _gt_base = GENERATION_V3_DIR.parent  # synthetic generation/
    if _gt_base.exists():
        _gt_files = sorted([f for f in _gt_base.iterdir() if f.name.endswith("_ground_truth.json")])

    _use_ccda = st.checkbox("Include CCDA patient background noise", value=bool(_gt_files), key="demo_use_ccda")
    _ccda_noise = {}
    _demographics = {}

    if _use_ccda and _gt_files:
        _gt_names = [f.name for f in _gt_files]
        _selected_gt_name = st.selectbox("Select Patient Record", _gt_names, key="demo_ccda")
        _gt_path = _gt_base / _selected_gt_name
        if _gt_path.exists():
            import json as _json2
            with open(_gt_path, "r", encoding="utf-8") as _f:
                _gt_data = _json2.load(_f)
            _ccda_noise = _extract_ccda_noise(_gt_data)
            _demographics = _extract_demographics(_gt_data)

            # Override CCDA location (Synthea defaults to MA) with SC hospital hub
            import random as _loc_rnd
            _loc_hub = _loc_rnd.choice(SC_HOSPITAL_HUBS)
            _demographics["city"] = _loc_hub["city"]
            _demographics["state"] = "SC"

            # Nice patient info card
            _dem = _demographics
            _conditions = _ccda_noise.get("conditions", [])
            _medications = _ccda_noise.get("medications", [])
            _allergies = _ccda_noise.get("allergies", [])
            _cond_html = ''.join(f'<span class="badge" style="margin:2px">{c}</span>' for c in _conditions[:10]) if _conditions else '<span style="color:#94A3B8">None recorded</span>'
            _med_html = ''.join(f'<span class="badge" style="margin:2px;background:#FEF3C7;border-color:#FDE68A;color:#92400E">{m}</span>' for m in _medications[:8]) if _medications else '<span style="color:#94A3B8">None recorded</span>'
            _allergy_html = ''.join(f'<span class="badge" style="margin:2px;background:#FEE2E2;border-color:#FECACA;color:#991B1B">{a}</span>' for a in _allergies) if _allergies else '<span style="color:#94A3B8">None recorded</span>'

            st.markdown(
                f'<div class="card">'
                f'<div style="display:flex;gap:2rem;flex-wrap:wrap;margin-bottom:0.8rem">'
                f'<div><span style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:0.05em">Name</span><br>'
                f'<span style="font-weight:700;font-size:1.1rem">{_dem.get("name","Unknown")}</span></div>'
                f'<div><span style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:0.05em">Age</span><br>'
                f'<span style="font-weight:700;font-size:1.1rem">{_dem.get("age","")}</span></div>'
                f'<div><span style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:0.05em">Gender</span><br>'
                f'<span style="font-weight:700;font-size:1.1rem">{_dem.get("gender","")}</span></div>'
                f'<div><span style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:0.05em">Race / Ethnicity</span><br>'
                f'<span style="font-weight:700;font-size:1.1rem">{_dem.get("race","")} / {_dem.get("ethnicity","")}</span></div>'
                f'<div><span style="font-size:0.72rem;color:#64748B;text-transform:uppercase;letter-spacing:0.05em">Location</span><br>'
                f'<span style="font-weight:700;font-size:1.1rem">{_dem.get("city","")}, {_dem.get("state","")}</span></div>'
                f'</div>'
                f'<div style="margin-bottom:0.5rem"><span style="font-size:0.78rem;font-weight:600;color:#64748B">CONDITIONS</span><br>{_cond_html}</div>'
                f'<div style="margin-bottom:0.5rem"><span style="font-size:0.78rem;font-weight:600;color:#64748B">MEDICATIONS</span><br>{_med_html}</div>'
                f'<div><span style="font-size:0.78rem;font-weight:600;color:#64748B">ALLERGIES</span><br>{_allergy_html}</div>'
                f'</div>',
                unsafe_allow_html=True)
    elif _use_ccda and not _gt_files:
        st.info("No ground-truth JSON files found. The transcript will be generated without patient background noise.")

    st.markdown("---")

    # ── Step 3: Build scenario & generate ────────────────────────────
    st.markdown("### Step 3 — Generate Transcript")

    _api_key_present = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    if not _api_key_present:
        st.warning("Set the `OPENAI_API_KEY` environment variable before generating.")

    # Persistent transcript placeholder
    if "generated_transcript" not in st.session_state:
        st.session_state["generated_transcript"] = None
        st.session_state["transcript_tokens"] = None
        st.session_state["transcript_scenario"] = None
        st.session_state["transcript_prompts"] = None

    if st.button("Generate Transcript", key="demo_generate", disabled=not _api_key_present):
        import random as _rnd
        _rng = _rnd.Random(_seed)

        if _selected_symptoms:
            _present = list(_selected_symptoms)
            _chief = _present[0]
            _negated = []
        elif "demo_sampled" in st.session_state:
            _s = st.session_state["demo_sampled"]
            _present = _s["present"]
            _chief = _s["chief"]
            _negated = _s["negated"]
        else:
            _present, _chief, _negated, _diff_sys_out = sample_symptoms(
                _rng,
                case_type=_case_type,
                n_present_min=min(_n_present, _max_present) if _max_present > 0 else 3,
                n_present_max=min(_n_present, _max_present) if _max_present > 0 else 3,
                n_negated_min=_n_negated,
                n_negated_max=_n_negated,
                differential_system=_diff_sys,
            )

        # For novel_virus, inject user-selected hallmarks
        if _case_type == "novel_virus" and not _selected_symptoms:
            _hallmarks = st.session_state.get("demo_hallmarks", NOVEL_VIRUS_HALLMARKS[:3])
            # Remove any auto-sampled hallmarks and replace with user selection
            _present = [s for s in _present if s not in NOVEL_VIRUS_HALLMARKS]
            _present.extend(_hallmarks)

        _scenario = {
            "case_type": _case_type,
            "demographics": _demographics or {"name": "Demo Patient", "age": "45", "gender": "Female"},
            "chief_complaint": _chief,
            "present_symptoms": _present,
            "negated_symptoms": _negated,
            "ccda_noise": _ccda_noise,
            "location": {"hospital": "Prisma Health Greenville", "city": "Greenville", "state": "SC"},
            "seed": _seed,
        }
        if _diff_sys:
            _scenario["differential_system"] = _diff_sys

        with st.spinner("Calling GPT-4o — this may take 30-60 seconds …"):
            _sys_prompt, _usr_prompt = build_prompt(_scenario)
            _transcript, _token_usage = generate_transcript(_sys_prompt, _usr_prompt)

        st.session_state["generated_transcript"] = _transcript
        st.session_state["transcript_tokens"] = _token_usage
        st.session_state["transcript_scenario"] = _scenario
        st.session_state["transcript_prompts"] = (_sys_prompt, _usr_prompt)

    # ── Display results (persisted across reruns) ────────────────────
    if st.session_state.get("generated_transcript"):
        _transcript = st.session_state["generated_transcript"]
        _token_usage = st.session_state["transcript_tokens"]
        _scenario = st.session_state["transcript_scenario"]
        _prompts = st.session_state["transcript_prompts"]

        # Token / cost metrics
        _p_tok = _token_usage.get("prompt_tokens", 0)
        _c_tok = _token_usage.get("completion_tokens", 0)
        _cost = (_p_tok / 1_000_000) * INPUT_COST_PER_M + (_c_tok / 1_000_000) * OUTPUT_COST_PER_M
        _tk1, _tk2, _tk3 = st.columns(3)
        with _tk1:
            st.metric("Prompt Tokens", f"{_p_tok:,}")
        with _tk2:
            st.metric("Completion Tokens", f"{_c_tok:,}")
        with _tk3:
            st.metric("Estimated Cost", f"${_cost:.4f}")

        # Prompt viewer
        with st.expander("View Full Prompt"):
            st.markdown("**System Prompt**")
            st.markdown(
                '<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
                'border-radius:8px;padding:1rem;font-family:Source Code Pro,monospace;'
                'font-size:0.78rem;line-height:1.55;white-space:pre-wrap;'
                'color:#0F172A;max-height:400px;overflow-y:auto">'
                f'{_prompts[0]}'
                '</div>', unsafe_allow_html=True)
            st.markdown("**User Prompt**")
            st.markdown(
                '<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
                'border-radius:8px;padding:1rem;font-family:Source Code Pro,monospace;'
                'font-size:0.78rem;line-height:1.55;white-space:pre-wrap;'
                'color:#0F172A;max-height:400px;overflow-y:auto">'
                f'{_prompts[1]}'
                '</div>', unsafe_allow_html=True)

        # Scenario viewer
        with st.expander("View Scenario JSON"):
            import json as _json_disp
            _scenario_text = _json_disp.dumps(_scenario, indent=2, default=str)
            st.markdown(
                '<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
                'border-radius:8px;padding:1rem;font-family:Source Code Pro,monospace;'
                'font-size:0.78rem;line-height:1.55;white-space:pre-wrap;'
                'color:#0F172A;max-height:400px;overflow-y:auto">'
                f'{_scenario_text}'
                '</div>', unsafe_allow_html=True)

        # Transcript display
        st.markdown("### Generated Transcript")
        st.markdown(
            '<div style="background:#FFFFFF;border:1.5px solid #E2E8F0;'
            'border-radius:10px;padding:1.4rem 1.6rem;max-height:600px;'
            'overflow-y:auto;font-family:\'Source Code Pro\',monospace;'
            'font-size:0.82rem;line-height:1.6;white-space:pre-wrap;'
            'color:#0F172A;box-shadow:0 2px 8px rgba(0,0,0,0.06)">'
            f'{_transcript}'
            '</div>',
            unsafe_allow_html=True)
    else:
        # Empty placeholder
        st.markdown(
            '<div class="placeholder-panel">'
            '<h4>Generated Transcript</h4>'
            '<p>Configure the scenario above, then click <b>Generate Transcript</b>. '
            'The generated doctor-patient conversation will appear here.</p>'
            '</div>',
            unsafe_allow_html=True)
