"""
Disease Simulation Dashboard – South Carolina
===============================================
Streamlit app that visualises the simulated disease-outbreak timeline
across 7 SC hospital hubs.  Cases are geo-jittered around each hub and
colour-coded by type (Novel, Flu, Distractor, Healthy).

Run:  streamlit run disease_simulation_dashboard.py
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Hub definitions (same as sc_hospital_hubs.py)
# ──────────────────────────────────────────────────────────────────────────────

SC_HOSPITAL_HUBS = [
    {"hub_id": "H1", "name": "AnMed Health Medical Center",    "city": "Anderson",     "lat": 34.526, "lng": -82.646},
    {"hub_id": "H2", "name": "Prisma Health Greenville",       "city": "Greenville",   "lat": 34.819, "lng": -82.416},
    {"hub_id": "H3", "name": "Prisma Health Richland",         "city": "Columbia",     "lat": 34.032, "lng": -81.033},
    {"hub_id": "H4", "name": "Aiken Regional Medical Center",  "city": "Aiken",        "lat": 33.565, "lng": -81.763},
    {"hub_id": "H5", "name": "MUSC Health",                    "city": "Charleston",   "lat": 32.785, "lng": -79.947},
    {"hub_id": "H6", "name": "McLeod Regional Medical Center", "city": "Florence",     "lat": 34.192, "lng": -79.765},
    {"hub_id": "H7", "name": "Grand Strand Medical Center",    "city": "Myrtle Beach", "lat": 33.748, "lng": -78.847},
]

# Map CSV filenames → hub_id
_CSV_TO_HUB = {
    "Simulated Timeline -  AnMed (Anderson.csv":              "H1",
    "Simulated Timeline - Prisma (Greenville).csv":           "H2",
    "Simulated Timeline - Prisma (Richland).csv":             "H3",
    "Simulated Timeline - Aiken Regional.csv":                "H4",
    "Simulated Timeline - MUSC Health (Charleston).csv":      "H5",
    "Simulated Timeline - McLeod Regional Medical Center.csv":"H6",
    "Simulated Timeline - Grand Strand Medical Center.csv":   "H7",
}

HUB_LOOKUP = {h["hub_id"]: h for h in SC_HOSPITAL_HUBS}

# Colour palette for case types
CASE_COLOURS = {
    "Novel (Signal)": "#e63946",   # red
    "Flu":            "#457b9d",   # blue
    "Distractor":     "#f4a261",   # orange
    "Healthy":        "#2a9d8f",   # teal
}

# Time period labels in chronological order
TIME_PERIODS = [
    "Mar 01–03",
    "Mar 04–07",
    "Mar 08–11",
    "Mar 12–14",
]


# ──────────────────────────────────────────────────────────────────────────────
# Geo-jitter helper
# ──────────────────────────────────────────────────────────────────────────────

def jitter_point(rng: random.Random, lat: float, lng: float,
                 radius_km: float = 20.0):
    """Return a uniformly-distributed random point within *radius_km*."""
    km_per_deg_lat = 111.0
    km_per_deg_lng = 111.0 * math.cos(math.radians(lat))
    angle = rng.uniform(0, 2 * math.pi)
    r = radius_km * math.sqrt(rng.random())
    return (
        round(lat + (r * math.sin(angle)) / km_per_deg_lat, 5),
        round(lng + (r * math.cos(angle)) / km_per_deg_lng, 5),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data(seed: int = 42, radius_km: float = 20.0):
    """Read every CSV, expand counts into individual case rows with jittered
    lat/lng, and return a single DataFrame."""

    data_dir = Path(__file__).parent / "Simualted Timeline"
    rng = random.Random(seed)
    rows: list[dict] = []

    for csv_name, hub_id in _CSV_TO_HUB.items():
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            st.warning(f"Missing file: {csv_path}")
            continue

        hub = HUB_LOOKUP[hub_id]
        raw = pd.read_csv(csv_path)

        # Skip the "Totals" row
        raw = raw[~raw.iloc[:, 0].str.lower().str.startswith("total")]

        for row_idx, row in raw.iterrows():
            date_range_raw = row.iloc[0]               # e.g. "01–03 (Baseline)"
            time_label = TIME_PERIODS[row_idx] if row_idx < len(TIME_PERIODS) else f"Period {row_idx}"

            for case_type in ["Novel (Signal)", "Flu", "Distractor", "Healthy"]:
                count_raw = row[case_type]
                # Handle approximate values like "~43"
                count_str = str(count_raw).replace("~", "").strip()
                try:
                    count = int(float(count_str))
                except (ValueError, TypeError):
                    count = 0

                for _ in range(count):
                    jlat, jlng = jitter_point(rng, hub["lat"], hub["lng"], radius_km)
                    rows.append({
                        "hub_id":     hub_id,
                        "hospital":   hub["name"],
                        "city":       hub["city"],
                        "time_period": time_label,
                        "phase":      date_range_raw,
                        "case_type":  case_type,
                        "lat":        jlat,
                        "lng":        jlng,
                    })

    df = pd.DataFrame(rows)
    # Ensure chronological ordering for time_period
    df["time_period"] = pd.Categorical(df["time_period"], categories=TIME_PERIODS, ordered=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="SC Disease Simulation", layout="wide")
    st.title("🦠 South Carolina Disease Simulation Dashboard")
    st.markdown(
        "Visualise the simulated outbreak across **7 hospital hubs** in SC.  "
        "Use the **time slider** to step through the 4 time periods in March."
    )

    df = load_data()

    if df.empty:
        st.error("No data loaded – check the CSV folder path.")
        return

    # ── Sidebar controls ─────────────────────────────────────────────────────
    st.sidebar.header("Controls")

    period_idx = st.sidebar.slider(
        "Time Period",
        min_value=0,
        max_value=len(TIME_PERIODS) - 1,
        value=0,
        format="",
    )
    selected_period = TIME_PERIODS[period_idx]
    st.sidebar.markdown(f"**Selected:** {selected_period}")

    cumulative = st.sidebar.checkbox("Show cumulative (all periods up to selected)", value=False)

    selected_types = st.sidebar.multiselect(
        "Case Types to Display",
        options=list(CASE_COLOURS.keys()),
        default=list(CASE_COLOURS.keys()),
    )

    # ── Filter data ──────────────────────────────────────────────────────────
    if cumulative:
        cutoff = TIME_PERIODS.index(selected_period)
        valid_periods = TIME_PERIODS[: cutoff + 1]
        df_filtered = df[
            (df["time_period"].isin(valid_periods)) & (df["case_type"].isin(selected_types))
        ]
    else:
        df_filtered = df[
            (df["time_period"] == selected_period) & (df["case_type"].isin(selected_types))
        ]

    # ── KPI row ──────────────────────────────────────────────────────────────
    cols = st.columns(4)
    for i, ct in enumerate(CASE_COLOURS):
        count = int((df_filtered["case_type"] == ct).sum()) if ct in selected_types else 0
        cols[i].metric(ct, count)

    # ── Map ──────────────────────────────────────────────────────────────────
    st.subheader(f"Case Map – {selected_period}" + (" (cumulative)" if cumulative else ""))

    if df_filtered.empty:
        st.info("No cases to display for this selection.")
    else:
        fig_map = px.scatter_mapbox(
            df_filtered,
            lat="lat",
            lon="lng",
            color="case_type",
            color_discrete_map=CASE_COLOURS,
            hover_data=["hospital", "city", "phase", "case_type"],
            category_orders={"case_type": list(CASE_COLOURS.keys())},
            zoom=6.3,
            center={"lat": 33.8, "lon": -80.9},
            height=600,
            opacity=0.7,
        )
        fig_map.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=0, r=0, t=0, b=0),
            legend_title_text="Case Type",
        )
        # Add hospital hub markers
        hub_df = pd.DataFrame(SC_HOSPITAL_HUBS)
        fig_map.add_trace(go.Scattermapbox(
            lat=hub_df["lat"],
            lon=hub_df["lng"],
            mode="markers+text",
            marker=dict(size=12, color="black", symbol="hospital"),
            text=hub_df["city"],
            textposition="top center",
            name="Hospital Hub",
            hovertext=hub_df["name"],
        ))
        st.plotly_chart(fig_map, use_container_width=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)

    # 1) Novel uptick over time (all hubs stacked)
    with chart_col1:
        st.subheader("Novel (Signal) Cases Over Time")
        novel_df = df[df["case_type"] == "Novel (Signal)"]
        novel_counts = (
            novel_df.groupby(["time_period", "hospital"])
            .size()
            .reset_index(name="count")
        )
        fig_novel = px.bar(
            novel_counts,
            x="time_period",
            y="count",
            color="hospital",
            barmode="stack",
            labels={"time_period": "Time Period", "count": "Novel Cases"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_novel.update_layout(xaxis_tickangle=-30, legend_title_text="Hospital")
        st.plotly_chart(fig_novel, use_container_width=True)

    # 2) Case mix per time period (grouped bar)
    with chart_col2:
        st.subheader("All Case Types Over Time")
        all_counts = (
            df.groupby(["time_period", "case_type"])
            .size()
            .reset_index(name="count")
        )
        fig_all = px.bar(
            all_counts,
            x="time_period",
            y="count",
            color="case_type",
            barmode="group",
            color_discrete_map=CASE_COLOURS,
            category_orders={"case_type": list(CASE_COLOURS.keys())},
            labels={"time_period": "Time Period", "count": "Cases"},
        )
        fig_all.update_layout(xaxis_tickangle=-30, legend_title_text="Case Type")
        st.plotly_chart(fig_all, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    # 3) Novel proportion over time (line)
    with chart_col3:
        st.subheader("Novel Case Proportion Over Time")
        period_totals = df.groupby("time_period").size().reset_index(name="total")
        novel_totals = novel_df.groupby("time_period").size().reset_index(name="novel")
        prop = period_totals.merge(novel_totals, on="time_period", how="left").fillna(0)
        prop["proportion"] = prop["novel"] / prop["total"]
        fig_prop = px.line(
            prop,
            x="time_period",
            y="proportion",
            markers=True,
            labels={"time_period": "Time Period", "proportion": "Novel / Total"},
        )
        fig_prop.update_traces(line_color="#e63946", line_width=3)
        fig_prop.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30)
        st.plotly_chart(fig_prop, use_container_width=True)

    # 4) Per-hub heatmap of Novel cases
    with chart_col4:
        st.subheader("Novel Cases Heatmap (Hub × Period)")
        heat = (
            novel_df.groupby(["hospital", "time_period"])
            .size()
            .reset_index(name="count")
        )
        heat_pivot = heat.pivot(index="hospital", columns="time_period", values="count").fillna(0)
        # Reorder columns
        heat_pivot = heat_pivot.reindex(columns=TIME_PERIODS, fill_value=0)
        fig_heat = px.imshow(
            heat_pivot.values,
            x=list(heat_pivot.columns),
            y=list(heat_pivot.index),
            color_continuous_scale="Reds",
            labels=dict(x="Time Period", y="Hospital", color="Novel Cases"),
            aspect="auto",
        )
        fig_heat.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_heat, use_container_width=True)

    # 5) Cumulative novel curve
    st.subheader("Cumulative Novel Cases Over Time")
    novel_by_hub_period = (
        novel_df.groupby(["time_period", "hospital"])
        .size()
        .reset_index(name="count")
    )
    # Compute cumulative per hospital
    cum_rows = []
    for hosp in novel_by_hub_period["hospital"].unique():
        sub = novel_by_hub_period[novel_by_hub_period["hospital"] == hosp].sort_values("time_period")
        sub = sub.copy()
        sub["cumulative"] = sub["count"].cumsum()
        cum_rows.append(sub)
    if cum_rows:
        cum_df = pd.concat(cum_rows)
        fig_cum = px.line(
            cum_df,
            x="time_period",
            y="cumulative",
            color="hospital",
            markers=True,
            labels={"time_period": "Time Period", "cumulative": "Cumulative Novel Cases"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_cum.update_layout(xaxis_tickangle=-30, legend_title_text="Hospital")
        st.plotly_chart(fig_cum, use_container_width=True)

    # ── Summary table ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Summary Table")
    summary = (
        df.groupby(["hospital", "time_period", "case_type"])
        .size()
        .reset_index(name="count")
        .pivot_table(index=["hospital", "time_period"], columns="case_type", values="count", fill_value=0)
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
