"""
Syndromic Surveillance Dashboard
=================================
Streamlit application with two views:
  1. Hospital Hub View   — per-hospital patient volume, cost, anomaly ticker
  2. Command Center      — cumulative HDBSCAN clustering, SC map, t-SNE, drill-down
"""
from __future__ import annotations

import time
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
)
from data_loader import load_simulation_data, load_embeddings
from clustering_engine import precompute_all_days, build_categorical_matrix, _build_detail_types, WINDOW_SIZE

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
.main .block-container { padding-top: 0.5rem !important; max-width: 100%; }
[data-testid="stAppViewBlockContainer"] { padding-top: 0.5rem !important; }
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
    padding: 0.45rem 1.2rem;
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
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    transition: box-shadow 0.15s ease;
}
.kpi-card:hover {
    box-shadow: 0 4px 12px rgba(37,99,235,0.10);
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
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.card:hover {
    box-shadow: 0 4px 16px rgba(37,99,235,0.10);
    transform: translateY(-1px);
}
.anomaly-item {
    transition: box-shadow 0.15s ease, transform 0.1s ease;
}
.anomaly-item:hover {
    box-shadow: 0 2px 8px rgba(37,99,235,0.12);
    transform: translateY(-1px);
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


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

df  = load_simulation_data()
emb = load_embeddings()
cat = build_categorical_matrix(str(FEWSHOT_DIR))
detail_types = _build_detail_types(str(FEWSHOT_DIR))

import json as _json
_df_json   = df.to_json()
_emb_bytes = emb.astype(np.float32).tobytes()
_cat_bytes = cat.tobytes()
_cat_shape = cat.shape
_detail_json = _json.dumps(detail_types)
clustering = precompute_all_days(
    _df_json, _emb_bytes, _cat_bytes, _cat_shape, _detail_json, NUM_DAYS
)


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTLY DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════

_PLOTLY = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(family="Space Grotesk, sans-serif", color="#0F172A", size=13),
    margin=dict(l=40, r=20, t=36, b=40),
    title_font=dict(color="#0F172A"),
    legend_font=dict(color="#0F172A"),
    hoverlabel=dict(bgcolor="#FFFFFF", font_size=13, font_color="#0F172A",
                    bordercolor="#E2E8F0"),
)
_AXIS_FONT = dict(tickfont=dict(color="#0F172A"), title_font=dict(color="#0F172A"))


# ═══════════════════════════════════════════════════════════════════════════
#  STATIC TOGGLE BAR
# ═══════════════════════════════════════════════════════════════════════════

_is_hospital = st.session_state.current_view == "hospital"
_cls_h = "active" if _is_hospital else "inactive"
_cls_c = "active" if not _is_hospital else "inactive"

toggle_cols = st.columns([2, 10])
with toggle_cols[0]:
    bc1, bc2 = st.columns(2, gap="small")
    with bc1:
        if st.button("Hospital Hub", key="btn_hospital"):
            st.session_state.current_view = "hospital"
            st.rerun()
    with bc2:
        if st.button("Command Center", key="btn_command"):
            st.session_state.current_view = "command"
            st.rerun()

# Thin indicator line
_indicator_text = "HOSPITAL HUB VIEW" if _is_hospital else "PUBLIC HEALTH COMMAND CENTER"
st.markdown(
    f'<div style="font-family:Source Code Pro,monospace;font-size:0.70rem;'
    f'font-weight:600;color:#2563EB;letter-spacing:0.10em;'
    f'border-bottom:2px solid #2563EB;display:inline-block;'
    f'padding-bottom:2px;margin-bottom:0.8rem;">{_indicator_text}</div>',
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════
#  VIEW 1 — HOSPITAL HUB
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.current_view == "hospital":

    # ── Controls row: hospital dropdown + day selector + scale ─────────
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])
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
        day_mode = st.selectbox("Time Range", ["Single Day", "Full Week"], key="day_mode")
    with ctrl3:
        scale = st.selectbox("Scale", [1, 2, 5, 10], index=0, key="hub_scale",
                             format_func=lambda x: f"{x}x" if x > 1 else "Actual")

    hub_info = HUB_LOOKUP[selected_hub_id]

    if day_mode == "Single Day":
        selected_day = st.slider("Day", 1, 7, value=7, key="hub_day")
        day_df = df[(df["hub_id"] == selected_hub_id) & (df["day"] == selected_day)]
        _day_subtitle = f"Day {selected_day} ({DATE_LABELS.get(selected_day, '')})"
    else:
        day_df = df[df["hub_id"] == selected_hub_id]
        _day_subtitle = "Full Week  |  Days 1-7"

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
        fig_vol.update_layout(**_PLOTLY, showlegend=False, yaxis_title="",
                              xaxis_title="", height=300,
                              xaxis=_AXIS_FONT, yaxis=_AXIS_FONT)
        fig_vol.update_traces(textposition="outside", marker_line_width=0, width=0.55)
        fig_vol.update_xaxes(showgrid=False)
        fig_vol.update_yaxes(showgrid=False, showticklabels=False)
        st.plotly_chart(fig_vol, width="stretch")

    with chart2:
        if day_mode == "Full Week":
            st.markdown('<div class="section-header">Daily Volume Trend</div>',
                        unsafe_allow_html=True)
            trend = day_df.groupby("day").size().reindex(range(1, 8), fill_value=0)
            trend = trend.reset_index()
            trend.columns = ["Day", "Patients"]
            trend["Patients"] = trend["Patients"] * scale
            trend["Date"] = trend["Day"].map(DATE_LABELS)
            fig_trend = px.line(trend, x="Date", y="Patients", markers=True)
            fig_trend.update_traces(line_color="#2563EB", marker_size=8)
            fig_trend.update_layout(**_PLOTLY, height=300, yaxis_title="Patients",
                                    xaxis_title="",
                                    xaxis=_AXIS_FONT, yaxis=_AXIS_FONT)
            fig_trend.update_xaxes(showgrid=False)
            fig_trend.update_yaxes(showgrid=True, gridcolor="#F1F5F9")
            st.plotly_chart(fig_trend, width="stretch")
        else:
            st.markdown('<div class="section-header">Case Type Distribution</div>',
                        unsafe_allow_html=True)
            dist = day_df["case_type"].value_counts().reset_index()
            dist.columns = ["Case Type", "Count"]
            dist["Label"] = dist["Case Type"].map(CASE_DISPLAY)
            fig_pie = px.pie(dist, values="Count", names="Label",
                             color="Case Type", color_discrete_map=CASE_COLORS,
                             hole=0.45)
            fig_pie.update_layout(**_PLOTLY, height=300, showlegend=True,
                                  legend=dict(font=dict(size=11, color="#0F172A")))
            fig_pie.update_traces(textinfo="percent+value",
                                  textfont_size=11, textfont_color="#0F172A")
            st.plotly_chart(fig_pie, width="stretch")

    # ── Anomaly Ticker ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Live Anomaly Ticker</div>',
                unsafe_allow_html=True)
    flagged = day_df[day_df["Novelty_Flag"]].copy()

    if flagged.empty:
        st.markdown(
            '<p style="color:#64748B;text-align:center;padding:2rem 0">'
            'No novel symptoms detected for this selection.</p>',
            unsafe_allow_html=True)
    else:
        # Build all anomaly cards as HTML grid
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

    # ── Trend Forecasting Placeholder ──────────────────────────────────
    st.markdown(
        """<div class="placeholder-panel">
            <h4>Trend Forecasting</h4>
            <p>Predictive modeling and volume trend analysis -- coming soon</p>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  VIEW 2 — PUBLIC HEALTH COMMAND CENTER
# ═══════════════════════════════════════════════════════════════════════════

else:
    st.markdown(
        "<h1 style='font-size:1.6rem;font-weight:700;margin-bottom:0'>"
        "Public Health Command Center</h1>"
        "<p style='color:#64748B;margin-top:0;font-size:0.9rem'>"
        "South Carolina  |  Rolling Window Outbreak Analysis</p>",
        unsafe_allow_html=True,
    )

    # ── Day slider + play button ───────────────────────────────────────
    sl_col, play_col = st.columns([8, 1])
    with sl_col:
        day = st.slider("Simulation Day", 1, 7, value=1, key="cmd_day",
                         format="Day %d")
    with play_col:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        play_clicked = st.button("Play", key="play_btn")

    # Auto-play: advance one day per rerun, with a 4-second pause
    if play_clicked and not st.session_state.get("_playing", False):
        st.session_state["_playing"] = True
        st.session_state["_play_day"] = st.session_state.get("cmd_day", 1)

    if st.session_state.get("_playing", False):
        play_day = st.session_state.get("_play_day", 1)
        if play_day <= 7:
            day = play_day
            # show progress
            st.markdown(
                f'<div style="font-size:0.85rem;color:#2563EB;font-weight:600;'
                f'font-family:Source Code Pro,monospace">'
                f'Playing Day {play_day} / 7  |  {DATE_LABELS.get(play_day, "")}</div>',
                unsafe_allow_html=True)
            # schedule next day
            if play_day < 7:
                st.session_state["_play_day"] = play_day + 1
                time.sleep(2)
                st.rerun()
            else:
                # end of play
                st.session_state.pop("_playing", None)
                st.session_state.pop("_play_day", None)
        else:
            st.session_state.pop("_playing", None)
            st.session_state.pop("_play_day", None)
    else:
        day = st.session_state.get("cmd_day", 1)

    res = clustering[day]
    cum_df = df.iloc[res["idx"]].copy()
    cum_df["_cluster"] = res["labels"]
    cum_df["tsne_1"]   = res["tsne"][:, 0]
    cum_df["tsne_2"]   = res["tsne"][:, 1]
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

    # ── Sub-tabs: State Map (with cluster count) / Cluster Evolution / Case Data
    n_real_clusters = res["n_clusters"]
    tab_map, tab_tsne, tab_data = st.tabs([
        f"State Map  ({n_real_clusters} clusters)",
        "Cluster Evolution",
        "Case Data",
    ])

    # ···· Tab 1: SC Map — individual patient dots ·····················
    with tab_map:

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
                    size=8 if is_novel else 5,
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
                    size=16, color="#DC2626",
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
                size=18, color="#2563EB", symbol="circle",
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
                subunitcolor="rgb(200, 200, 200)", subunitwidth=0.5,
                center=dict(lat=SC_CENTER_LAT, lon=SC_CENTER_LNG),
                lonaxis=dict(range=[-84.0, -78.0]),
                lataxis=dict(range=[31.8, 35.5]),
                bgcolor="white",
            ),
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#FFFFFF",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.05, xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                font=dict(size=11, family="Space Grotesk, sans-serif", color="#0F172A"),
            ),
            hoverlabel=dict(bgcolor="#FFFFFF", font_size=13, font_color="#0F172A",
                            bordercolor="#E2E8F0"),
        )
        map_event = st.plotly_chart(fig_map, width="stretch",
                                    on_select="rerun", selection_mode="points")

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

    # ···· Tab 2: Cluster Evolution (t-SNE) — formatted like plot_clusters
    with tab_tsne:

        fig_tsne = go.Figure()
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

            fig_tsne.add_trace(go.Scatter(
                x=subset["tsne_1"], y=subset["tsne_2"],
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

        title_text = (f"Two-Stage HDBSCAN  |  {_window_label}  |  {_window_dates}  |  "
                      f"{n_real_clusters} cluster{'s' if n_real_clusters != 1 else ''}")

        fig_tsne.update_layout(
            **_PLOTLY,
            height=600,
            title=dict(text=title_text, font=dict(size=14, color="#0F172A"), x=0.5),
            xaxis=dict(title="t-SNE 1", showgrid=True, gridcolor="#F1F5F9",
                       zeroline=False, tickfont=dict(color="#0F172A"),
                       title_font=dict(color="#0F172A")),
            yaxis=dict(title="t-SNE 2", showgrid=True, gridcolor="#F1F5F9",
                       zeroline=False, tickfont=dict(color="#0F172A"),
                       title_font=dict(color="#0F172A")),
            legend=dict(
                yanchor="top", y=0.98, xanchor="left", x=0.01,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#E2E8F0", borderwidth=1,
                font=dict(size=10),
            ),
        )
        st.plotly_chart(fig_tsne, width="stretch")

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
| Algorithm | Two-Stage HDBSCAN |
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

    # ── Trend Forecasting Placeholder ──────────────────────────────────
    st.markdown(
        """<div class="placeholder-panel">
            <h4>Epidemiological Trend Forecasting</h4>
            <p>Predictive modeling and trend analysis across hospital network -- coming soon</p>
        </div>""", unsafe_allow_html=True)
