"""
Generate 7 maps of South Carolina — one per outbreak day — showing
patient cases scattered around their assigned hospital hubs.

Outputs:
  • 7 individual PNG maps  (day1_map.png … day7_map.png)
  • 1 combined 7-panel figure  (outbreak_7day_maps.png)

Uses Plotly for map rendering with SC county boundaries from the
built-in US-states GeoJSON.
"""

from __future__ import annotations

import json
import csv
import sys
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Paths ───────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
MASTER_CSV = _ROOT / "LLM Symptom Extraction Full Run" / "results" / "master_results.csv"
OUTPUT_DIR = _THIS_DIR / "maps"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Hub definitions (for star markers) ──────────────────────────────────────
SC_HOSPITAL_HUBS = [
    {"hub_id": "H1", "name": "AnMed Health",     "city": "Anderson",     "lat": 34.526, "lng": -82.646},
    {"hub_id": "H2", "name": "Prisma Greenville", "city": "Greenville",  "lat": 34.819, "lng": -82.416},
    {"hub_id": "H3", "name": "Prisma Richland",   "city": "Columbia",    "lat": 34.032, "lng": -81.033},
    {"hub_id": "H4", "name": "Aiken Regional",    "city": "Aiken",       "lat": 33.565, "lng": -81.763},
    {"hub_id": "H5", "name": "MUSC Health",        "city": "Charleston",  "lat": 32.785, "lng": -79.947},
    {"hub_id": "H6", "name": "McLeod Regional",   "city": "Florence",    "lat": 34.192, "lng": -79.765},
    {"hub_id": "H7", "name": "Grand Strand",      "city": "Myrtle Beach","lat": 33.748, "lng": -78.847},
]

# ─── Case-type colours ───────────────────────────────────────────────────────
CASE_COLORS = {
    "novel_virus":  "#e31a1c",   # bright red
    "flu_like":     "#1f78b4",   # blue
    "healthy":      "#33a02c",   # green
    "differential": "#6a3d9a",   # purple
}

CASE_LABELS = {
    "novel_virus":  "Novel Virus",
    "flu_like":     "Flu-like",
    "healthy":      "Healthy",
    "differential": "Differential",
}

OUTBREAK_START = date(2025, 12, 26)

# ─── SC boundary (simplified polygon) ────────────────────────────────────────
# Approximate SC outline for a clean background when geo-libraries unavailable.
# We'll use plotly's built-in mapbox/scattergeo with USA state layer instead.


def load_data():
    """Load master CSV and group rows by day."""
    rows = []
    with open(MASTER_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("lat") or not row.get("timestamp"):
                continue
            row["lat"] = float(row["lat"])
            row["lng"] = float(row["lng"])
            rows.append(row)

    by_day = defaultdict(list)
    for r in rows:
        by_day[r["timestamp"]].append(r)

    return rows, by_day


def _day_label(day_str: str) -> str:
    """Convert '2025-12-26' → 'Day 1 — Dec 26'."""
    d = date.fromisoformat(day_str)
    day_num = (d - OUTBREAK_START).days + 1
    return f"Day {day_num} — {d.strftime('%b %d')}"


def _count_novel(cases):
    return sum(1 for c in cases if c["case_type"] == "novel_virus")


def make_day_map(day_str: str, cases: list, *, save_path: Path | None = None,
                 show: bool = False) -> go.Figure:
    """Build a single SC map figure for one outbreak day."""

    day_label = _day_label(day_str)
    n_novel = _count_novel(cases)
    n_total = len(cases)

    fig = go.Figure()

    # ── Patient scatter points (by case type) ────────────────────────────
    for ct in ["healthy", "flu_like", "differential", "novel_virus"]:
        subset = [c for c in cases if c["case_type"] == ct]
        if not subset:
            continue

        hover_text = []
        for c in subset:
            diff = f" ({c.get('differential_system', '')})" if ct == "differential" else ""
            hover_text.append(
                f"<b>{CASE_LABELS.get(ct, ct)}{diff}</b><br>"
                f"Hub: {c['hub_id']} — {c.get('hospital', '')}<br>"
                f"File: {c['filename'][:40]}…"
            )

        fig.add_trace(go.Scattergeo(
            lat=[c["lat"] for c in subset],
            lon=[c["lng"] for c in subset],
            mode="markers",
            marker=dict(
                size=8 if ct == "novel_virus" else 6,
                color=CASE_COLORS.get(ct, "#999"),
                opacity=0.85,
                line=dict(width=0.5, color="white"),
                symbol="circle",
            ),
            name=f"{CASE_LABELS.get(ct, ct)} ({len(subset)})",
            hovertext=hover_text,
            hoverinfo="text",
        ))

    # ── Hub star markers ─────────────────────────────────────────────────
    fig.add_trace(go.Scattergeo(
        lat=[h["lat"] for h in SC_HOSPITAL_HUBS],
        lon=[h["lng"] for h in SC_HOSPITAL_HUBS],
        mode="markers+text",
        marker=dict(size=14, color="black", symbol="star", line=dict(width=1, color="gold")),
        text=[h["hub_id"] for h in SC_HOSPITAL_HUBS],
        textposition="top center",
        textfont=dict(size=9, color="black", family="Arial Black"),
        name="Hospital Hubs",
        hovertext=[f"<b>{h['name']}</b><br>{h['city']}, SC" for h in SC_HOSPITAL_HUBS],
        hoverinfo="text",
    ))

    # ── Layout (zoom to SC) ──────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{day_label}</b>"
                f"<br><span style='font-size:13px; color:#555'>"
                f"{n_total} cases  •  {n_novel} novel virus</span>"
            ),
            x=0.5,
            font=dict(size=18),
        ),
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="rgb(243, 243, 243)",
            showlakes=True,
            lakecolor="rgb(204, 230, 255)",
            showrivers=True,
            rivercolor="rgb(204, 230, 255)",
            showcountries=False,
            showsubunits=True,
            subunitcolor="rgb(180, 180, 180)",
            subunitwidth=1,
            center=dict(lat=33.8, lon=-80.9),
            lonaxis=dict(range=[-84.0, -78.0]),
            lataxis=dict(range=[31.8, 35.5]),
            bgcolor="white",
        ),
        legend=dict(
            yanchor="top", y=0.95,
            xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        width=900,
        height=650,
        paper_bgcolor="white",
    )

    if save_path:
        fig.write_image(str(save_path), scale=2)
        print(f"  Saved: {save_path.name}")

    if show:
        fig.show()

    return fig


def make_combined_figure(all_days: dict, save_path: Path | None = None):
    """Create a single image with all 7 day maps arranged in a grid."""

    sorted_days = sorted(all_days.keys())

    # 2 rows × 4 cols (7 maps + 1 empty cell for legend)
    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[_day_label(d) for d in sorted_days] + [""],
        specs=[
            [{"type": "scattergeo"}, {"type": "scattergeo"}, {"type": "scattergeo"}, {"type": "scattergeo"}],
            [{"type": "scattergeo"}, {"type": "scattergeo"}, {"type": "scattergeo"}, {"type": "scattergeo"}],
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    geo_settings = dict(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        showlakes=True,
        lakecolor="rgb(204, 230, 255)",
        showcountries=False,
        showsubunits=True,
        subunitcolor="rgb(200, 200, 200)",
        subunitwidth=0.5,
        center=dict(lat=33.8, lon=-80.9),
        lonaxis=dict(range=[-84.0, -78.0]),
        lataxis=dict(range=[31.8, 35.5]),
        bgcolor="white",
    )

    shown_legend = set()

    for i, day_str in enumerate(sorted_days):
        row = i // 4 + 1
        col = i % 4 + 1
        geo_key = f"geo{i + 1}" if i > 0 else "geo"

        cases = all_days[day_str]
        n_novel = _count_novel(cases)

        for ct in ["healthy", "flu_like", "differential", "novel_virus"]:
            subset = [c for c in cases if c["case_type"] == ct]
            if not subset:
                continue

            legend_name = CASE_LABELS.get(ct, ct)
            show_legend = legend_name not in shown_legend
            if show_legend:
                shown_legend.add(legend_name)

            fig.add_trace(
                go.Scattergeo(
                    lat=[c["lat"] for c in subset],
                    lon=[c["lng"] for c in subset],
                    mode="markers",
                    marker=dict(
                        size=5 if ct != "novel_virus" else 7,
                        color=CASE_COLORS.get(ct, "#999"),
                        opacity=0.8,
                        line=dict(width=0.3, color="white"),
                    ),
                    name=legend_name,
                    showlegend=show_legend,
                    hoverinfo="skip",
                    geo=geo_key,
                ),
                row=row, col=col,
            )

        # Hub markers
        fig.add_trace(
            go.Scattergeo(
                lat=[h["lat"] for h in SC_HOSPITAL_HUBS],
                lon=[h["lng"] for h in SC_HOSPITAL_HUBS],
                mode="markers",
                marker=dict(size=8, color="black", symbol="star",
                            line=dict(width=0.5, color="gold")),
                name="Hospital Hub",
                showlegend=("Hospital Hub" not in shown_legend),
                hoverinfo="skip",
                geo=geo_key,
            ),
            row=row, col=col,
        )
        shown_legend.add("Hospital Hub")

        fig.update_layout(**{geo_key: geo_settings})

    # Update the 8th subplot (empty) to hide it
    fig.update_layout(
        geo8=dict(visible=False),
    )

    # Annotate novel counts on each subplot title
    for i, day_str in enumerate(sorted_days):
        n = _count_novel(all_days[day_str])
        total = len(all_days[day_str])
        ann_idx = i  # subtitle annotation index
        if ann_idx < len(fig.layout.annotations):
            current = fig.layout.annotations[ann_idx].text
            fig.layout.annotations[ann_idx].text = (
                f"<b>{current}</b><br>"
                f"<span style='font-size:10px; color:#666'>"
                f"{total} cases • {n} novel</span>"
            )

    fig.update_layout(
        title=dict(
            text=(
                "<b>7-Day Outbreak Simulation — South Carolina Hospital Hubs</b>"
                "<br><span style='font-size:13px; color:#555'>"
                "Dec 26, 2025 → Jan 1, 2026  •  500 patients across 7 hubs</span>"
            ),
            x=0.5,
            font=dict(size=20),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.05,
            xanchor="center", x=0.5,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.9)",
        ),
        width=1800,
        height=900,
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=100, b=60),
    )

    if save_path:
        fig.write_image(str(save_path), scale=2)
        print(f"  Saved: {save_path.name}")

    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  7-DAY OUTBREAK MAP GENERATOR")
    print("═" * 60)

    all_rows, by_day = load_data()
    sorted_days = sorted(by_day.keys())

    print(f"\n  Loaded {len(all_rows)} patients across {len(sorted_days)} days")
    for d in sorted_days:
        n = _count_novel(by_day[d])
        print(f"    {_day_label(d):20s}  {len(by_day[d]):3d} cases  ({n} novel)")
    print()

    # ── Individual day maps ──────────────────────────────────────────────
    print("  Generating individual day maps…")
    for i, day_str in enumerate(sorted_days, 1):
        out = OUTPUT_DIR / f"day{i}_map.png"
        make_day_map(day_str, by_day[day_str], save_path=out)

    # ── Combined 7-panel figure ──────────────────────────────────────────
    print("\n  Generating combined 7-panel figure…")
    combined_path = OUTPUT_DIR / "outbreak_7day_maps.png"
    make_combined_figure(by_day, save_path=combined_path)

    # ── Also save an interactive HTML version ────────────────────────────
    print("\n  Generating interactive HTML version…")
    html_path = OUTPUT_DIR / "outbreak_7day_maps.html"
    fig = make_combined_figure(by_day)
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"  Saved: {html_path.name}")

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
