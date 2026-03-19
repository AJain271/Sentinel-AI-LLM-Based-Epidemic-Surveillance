"""
Batch 500 — Individual Chart Export
Saves each chart as a separate PNG in Batch 500/charts/
"""

import csv, json, re
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
CSV_PATH  = BASE / "transcript_index.csv"
TX_DIR    = BASE / "transcripts"
META_DIR  = BASE / "metadata"
CHART_DIR = BASE / "charts"
CHART_DIR.mkdir(exist_ok=True)

# ── load CSV ─────────────────────────────────────────────────────────
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# ── load metadata JSONs ──────────────────────────────────────────────
meta_data = []
for r in rows:
    mp = META_DIR / r["metadata_filename"]
    if mp.exists():
        with open(mp, encoding="utf-8") as f:
            meta_data.append(json.load(f))
    else:
        meta_data.append(None)

print(f"Loaded {len(rows)} rows, {sum(1 for m in meta_data if m)} metadata JSONs")

# ── derived arrays ───────────────────────────────────────────────────
case_types = [r["case_type"] for r in rows]
subtypes   = [r["differential_subtype"] for r in rows]
present    = [int(r["present_symptom_count"]) for r in rows]
negated    = [int(r["negated_symptom_count"]) for r in rows]

ages    = [m["demographics"]["age"] for m in meta_data if m]
genders = [m["demographics"]["gender"] for m in meta_data if m]

total_lines = [m["transcript_stats"]["total_lines"] for m in meta_data if m]
doc_lines   = [m["transcript_stats"]["doctor_lines"] for m in meta_data if m]
pat_lines   = [m["transcript_stats"]["patient_lines"] for m in meta_data if m]

word_counts = []
for r in rows:
    tp = TX_DIR / r["filename"]
    if tp.exists():
        word_counts.append(len(tp.read_text(encoding="utf-8").split()))
    else:
        word_counts.append(0)

prompt_tok = [int(r["prompt_tokens"]) for r in rows if r["prompt_tokens"]]
comp_tok   = [int(r["completion_tokens"]) for r in rows if r["completion_tokens"]]
total_tok  = [int(r["total_tokens"]) for r in rows if r["total_tokens"]]
gen_times  = [float(r["generation_time_seconds"]) for r in rows]

hubs = [m["location"]["hospital"] for m in meta_data if m]

all_negated_names = []
all_present_names = []
type_present_names = {}   # case_type -> list of symptom strings
for row, m in zip(rows, meta_data):
    if m:
        ct = row["case_type"]
        for s in m.get("negated_symptoms", []):
            all_negated_names.append(re.sub(r"\s*\(.*?\)", "", s).strip())
        for s in m.get("present_symptoms", []):
            clean = re.sub(r"\s*\(.*?\)", "", s).strip()
            all_present_names.append(clean)
            type_present_names.setdefault(ct, []).append(clean)

type_words = {}
for ct, wc in zip(case_types, word_counts):
    type_words.setdefault(ct, []).append(wc)

sub_present = {}
for r in rows:
    if r["differential_subtype"]:
        sub_present.setdefault(r["differential_subtype"], []).append(
            int(r["present_symptom_count"]))

# ── colours (unified blue palette) ────────────────────────────────────
TYPE_COLORS = {
    "healthy": "#93C5FD", "flu_like": "#3B82F6",
    "novel_virus": "#1E40AF", "differential": "#1E3A5F",
}
SUB_COLORS = {
    "musculoskeletal": "#1E3A5F", "gastrointestinal": "#2563EB",
    "dermatological": "#60A5FA", "neurological": "#93C5FD",
}
TYPE_ORDER = ["healthy", "flu_like", "novel_virus", "differential"]
TYPE_LABELS = {
    "healthy": "Healthy", "flu_like": "Influenza",
    "novel_virus": "Novel Virus", "differential": "Varied Cases",
}

# Shared style
_BLUE_PRIMARY   = "#2563EB"
_BLUE_DARK      = "#1E3A5F"
_BLUE_MED       = "#3B82F6"
_BLUE_LIGHT     = "#93C5FD"
_BLUE_PALE      = "#DBEAFE"
_ACCENT_LINE    = "#1E40AF"
_BG             = "#FFFFFF"
_TEXT           = "#0F172A"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": _TEXT,
    "xtick.color": _TEXT,
    "ytick.color": _TEXT,
    "figure.facecolor": _BG,
    "axes.facecolor": _BG,
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "grid.alpha": 0.6,
})

DPI = 200
saved = 0

def save(fig, name):
    global saved
    path = CHART_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    saved += 1
    print(f"  [{saved:2d}] {path.name}")

# ══════════════════════════════════════════════════════════════════════
#  1. Case-type distribution (pie + legend)
# ══════════════════════════════════════════════════════════════════════
ct_counts = Counter(case_types)
ct_order  = sorted(ct_counts.keys(), key=lambda k: -ct_counts[k])
sizes     = [ct_counts[k] for k in ct_order]
total_n   = sum(sizes)

PIE_COLORS = ["#1E3A5F", "#2563EB", "#60A5FA", "#BFDBFE"]

fig, ax = plt.subplots(figsize=(9, 7), facecolor=_BG)
ax.set_facecolor(_BG)

wedges, _ = ax.pie(
    sizes,
    colors=PIE_COLORS[: len(sizes)],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 3},
    radius=1.0,
    pctdistance=0.70,
)

# ── % + count inside each slice ───────────────────────────────────
for wedge, key, val in zip(wedges, ct_order, sizes):
    pct     = val / total_n * 100
    ang     = (wedge.theta1 + wedge.theta2) / 2
    x       = 0.70 * np.cos(np.deg2rad(ang))
    y       = 0.70 * np.sin(np.deg2rad(ang))
    color   = "white" if pct > 12 else _BLUE_DARK
    ax.text(x, y, f"{pct:.1f}%\nn={val}",
            ha="center", va="center",
            fontsize=14, fontweight="bold",
            color=color, linespacing=1.5)

# ── legend ────────────────────────────────────────────────────────
import matplotlib.patches as mpatches
patches = [mpatches.Patch(facecolor=PIE_COLORS[i], edgecolor="white",
                           label=TYPE_LABELS[k])
           for i, k in enumerate(ct_order)]
ax.legend(handles=patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.08), ncol=2,
          fontsize=13, frameon=False,
          handlelength=1.5, handleheight=1.3,
          columnspacing=2.0, labelcolor=_TEXT)

ax.set_title("Case Type Distribution", fontweight="bold", fontsize=17,
             color=_TEXT, pad=20)
fig.tight_layout()
save(fig, "01_case_type_distribution")

# ══════════════════════════════════════════════════════════════════════
#  2. Differential subtype breakdown (bar)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
sub_counts = {k: v for k, v in Counter(subtypes).items() if k}
sub_labels = sorted(sub_counts.keys())
sub_vals   = [sub_counts[s] for s in sub_labels]
sub_cols   = [SUB_COLORS.get(s, "#999") for s in sub_labels]
bars = ax.bar(sub_labels, sub_vals, color=sub_cols, edgecolor="white", linewidth=1.2)
ax.bar_label(bars, fontsize=11, fontweight="bold", color=_TEXT)
ax.set_title("Differential Subtypes", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
save(fig, "02_differential_subtypes")

# ══════════════════════════════════════════════════════════════════════
#  3. Symptom counts table (present vs negated by case type)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.axis("off")

present_by_type = {t: [] for t in TYPE_ORDER}
negated_by_type = {t: [] for t in TYPE_ORDER}
for ct, p, n in zip(case_types, present, negated):
    present_by_type[ct].append(p)
    negated_by_type[ct].append(n)

col_headers = ["Case Type", "Present — Mean", "Present — Median",
               "Negated — Mean", "Negated — Median"]
table_data = []
for t in TYPE_ORDER:
    p_vals = present_by_type[t]
    n_vals = negated_by_type[t]
    table_data.append([
        TYPE_LABELS[t],
        f"{np.mean(p_vals):.1f}",
        f"{np.median(p_vals):.0f}",
        f"{np.mean(n_vals):.1f}",
        f"{np.median(n_vals):.0f}",
    ])

tbl = ax.table(
    cellText=table_data, colLabels=col_headers,
    cellLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2.4)

# Style header row
for col in range(len(col_headers)):
    cell = tbl[0, col]
    cell.set_facecolor(_BLUE_DARK)
    cell.set_text_props(color="white", fontweight="bold")

# Style data rows with alternating shades
for row in range(1, len(table_data) + 1):
    bg = _BLUE_PALE if row % 2 == 0 else "#FFFFFF"
    for col in range(len(col_headers)):
        cell = tbl[row, col]
        cell.set_facecolor(bg)
        cell.set_text_props(color=_TEXT)
        if col == 0:
            cell.set_text_props(color=_TEXT, fontweight="bold")

tbl.auto_set_column_width(list(range(len(col_headers))))
ax.set_title("Symptom Counts by Case Type", fontweight="bold", fontsize=14,
             color=_TEXT, pad=12)
fig.tight_layout()
save(fig, "03_symptom_counts_table")

# (chart 04 slot kept free — was negated box plot, now replaced by table above)

# ══════════════════════════════════════════════════════════════════════
#  5. Word count distribution (histogram)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4, 3))
ax.hist(word_counts, bins=30, color=_BLUE_PRIMARY, edgecolor="white", alpha=0.85)
ax.axvline(np.median(word_counts), color=_BLUE_DARK, ls="--", lw=2,
           label=f"median = {int(np.median(word_counts))}")
ax.axvline(np.mean(word_counts), color=_BLUE_LIGHT, ls=":", lw=2,
           label=f"mean = {int(np.mean(word_counts))}")
ax.set_title("Transcript Word Count Distribution", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Words"); ax.set_ylabel("Frequency")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "05_word_count_distribution")

# ══════════════════════════════════════════════════════════════════════
#  6. Word count by case type (box)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
bp3 = ax.boxplot([type_words.get(t, []) for t in TYPE_ORDER],
                 tick_labels=[TYPE_LABELS[t] for t in TYPE_ORDER], patch_artist=True, widths=0.6)
for patch, t in zip(bp3["boxes"], TYPE_ORDER):
    patch.set_facecolor(TYPE_COLORS[t]); patch.set_alpha(0.7)
ax.set_title("Word Count by Case Type", fontweight="bold", fontsize=14)
ax.set_ylabel("Words")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
save(fig, "06_word_count_by_type")

# ══════════════════════════════════════════════════════════════════════
#  7. Doctor vs Patient lines (scatter)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(doc_lines, pat_lines, alpha=0.5, s=20, c=_BLUE_PRIMARY, edgecolors="white", linewidth=0.3)
mx = max(max(doc_lines), max(pat_lines)) + 2
ax.plot([0, mx], [0, mx], ls="--", color=_BLUE_DARK, alpha=0.5, label="1:1 line")
ax.set_title("Doctor vs Patient Lines", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Doctor lines"); ax.set_ylabel("Patient lines")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "07_doctor_vs_patient_lines")

# ══════════════════════════════════════════════════════════════════════
#  8. Total lines per transcript (histogram)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(total_lines, bins=30, color=_BLUE_MED, edgecolor="white", alpha=0.85)
ax.axvline(np.median(total_lines), color=_BLUE_DARK, ls="--", lw=2,
           label=f"median = {int(np.median(total_lines))}")
ax.set_title("Total Lines per Transcript", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Lines"); ax.set_ylabel("Frequency")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "08_total_lines_distribution")

# ══════════════════════════════════════════════════════════════════════
#  9. Age distribution (histogram)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(ages, bins=20, color=_BLUE_PRIMARY, edgecolor="white", alpha=0.85)
ax.axvline(np.median(ages), color=_BLUE_DARK, ls="--", lw=2,
           label=f"median = {int(np.median(ages))}")
ax.set_title("Patient Age Distribution", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Age"); ax.set_ylabel("Frequency")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "09_age_distribution")

# ══════════════════════════════════════════════════════════════════════
#  10. Gender split (bar)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5, 5))
g_counts = Counter(genders)
g_labels = list(g_counts.keys())
g_vals   = list(g_counts.values())
g_colors = [_BLUE_DARK if g == "M" else _BLUE_LIGHT for g in g_labels]
bars_g = ax.bar(g_labels, g_vals, color=g_colors, edgecolor="white", linewidth=1.2)
ax.bar_label(bars_g, fontsize=13, fontweight="bold", color=_TEXT)
ax.set_title("Gender Distribution", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_ylabel("Count")
fig.tight_layout()
save(fig, "10_gender_distribution")

# ══════════════════════════════════════════════════════════════════════
#  11. Hospital hub distribution (horizontal bar)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
hub_counts = Counter(hubs)
hub_sorted = hub_counts.most_common()
hub_labels = [h for h, _ in hub_sorted]
hub_vals   = [v for _, v in hub_sorted]
bars_h = ax.barh(hub_labels, hub_vals, color=_BLUE_PRIMARY, edgecolor="white", linewidth=1.2)
ax.bar_label(bars_h, fontsize=10, fontweight="bold", color=_TEXT)
ax.set_title("Hospital Hub Distribution", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Count")
ax.invert_yaxis()
fig.tight_layout()
save(fig, "11_hospital_hub_distribution")

# ══════════════════════════════════════════════════════════════════════
#  12. Generation time distribution (histogram)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(gen_times, bins=30, color=_BLUE_MED, edgecolor="white", alpha=0.85)
ax.axvline(np.median(gen_times), color=_BLUE_DARK, ls="--", lw=2,
           label=f"median = {np.median(gen_times):.1f}s")
ax.set_title("Generation Time per Transcript", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Seconds"); ax.set_ylabel("Frequency")
ax.legend(fontsize=10)
fig.tight_layout()
save(fig, "12_generation_time")

# ══════════════════════════════════════════════════════════════════════
#  13. Total tokens distribution (histogram)
# ══════════════════════════════════════════════════════════════════════
if total_tok:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(total_tok, bins=30, color=_BLUE_PRIMARY, edgecolor="white", alpha=0.85)
    ax.axvline(np.median(total_tok), color=_BLUE_DARK, ls="--", lw=2,
               label=f"median = {int(np.median(total_tok))}")
    ax.set_title("Total Tokens per API Call", fontweight="bold", fontsize=14, color=_TEXT)
    ax.set_xlabel("Tokens"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)
    fig.tight_layout()
    save(fig, "13_total_tokens_distribution")

# ══════════════════════════════════════════════════════════════════════
#  14. Prompt vs Completion tokens (scatter)
# ══════════════════════════════════════════════════════════════════════
if prompt_tok and comp_tok:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(prompt_tok, comp_tok, alpha=0.5, s=20, c=_BLUE_PRIMARY, edgecolors="white", linewidth=0.3)
    ax.set_title("Prompt vs Completion Tokens", fontweight="bold", fontsize=14, color=_TEXT)
    ax.set_xlabel("Prompt tokens"); ax.set_ylabel("Completion tokens")
    fig.tight_layout()
    save(fig, "14_prompt_vs_completion_tokens")

# ══════════════════════════════════════════════════════════════════════
#  15. Top 20 present symptoms (horizontal bar)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
pres_top = Counter(all_present_names).most_common(20)
pres_labels = [s for s, _ in pres_top]
pres_vals   = [v for _, v in pres_top]
ax.barh(pres_labels[::-1], pres_vals[::-1], color=_BLUE_PRIMARY, edgecolor="white", linewidth=1.2)
ax.set_title("Top 20 Present Symptoms", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Occurrences")
ax.tick_params(axis="y", labelsize=9)
fig.tight_layout()
save(fig, "15_top20_present_symptoms")

# ══════════════════════════════════════════════════════════════════════
#  16. Top 20 negated symptoms (horizontal bar)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 7))
neg_top = Counter(all_negated_names).most_common(20)
neg_labels = [s for s, _ in neg_top]
neg_vals   = [v for _, v in neg_top]
ax.barh(neg_labels[::-1], neg_vals[::-1], color=_BLUE_LIGHT, edgecolor="white", linewidth=1.2)
ax.set_title("Top 20 Negated Symptoms", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Occurrences")
ax.tick_params(axis="y", labelsize=9)
fig.tight_layout()
save(fig, "16_top20_negated_symptoms")

# ══════════════════════════════════════════════════════════════════════
#  17. Present symptoms by differential subtype (box)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
sub_order = sorted(sub_present.keys())
bp4 = ax.boxplot([sub_present[s] for s in sub_order],
                 tick_labels=sub_order, patch_artist=True, widths=0.6)
for patch, s in zip(bp4["boxes"], sub_order):
    patch.set_facecolor(SUB_COLORS.get(s, "#999")); patch.set_alpha(0.7)
ax.set_title("Present Symptoms by Differential Subtype", fontweight="bold", fontsize=14)
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=25, labelsize=9)
fig.tight_layout()
save(fig, "17_present_by_differential_subtype")

# ══════════════════════════════════════════════════════════════════════
#  18. Word count vs present symptoms (scatter)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
colors_scatter = [TYPE_COLORS.get(ct, "#999") for ct in case_types]
ax.scatter(present, word_counts, alpha=0.55, s=22, c=colors_scatter, edgecolors="white", linewidth=0.3)
ax.set_title("Word Count vs Present Symptoms", fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlabel("Present symptom count"); ax.set_ylabel("Word count")
for t in TYPE_ORDER:
    ax.scatter([], [], c=TYPE_COLORS[t], label=TYPE_LABELS[t], s=40)
ax.legend(fontsize=9)
fig.tight_layout()
save(fig, "18_word_count_vs_present_symptoms")

# ══════════════════════════════════════════════════════════════════════
#  19. Summary statistics (text figure)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("off")

ct_c = Counter(case_types)
g_c  = Counter(genders)
stats_text = (
    f"{'='*55}\n"
    f"  BATCH 500 — SUMMARY STATISTICS\n"
    f"{'='*55}\n\n"
    f"  Total transcripts:       {len(rows)}\n"
    f"  Unique CCDAs used:       {len(set(r['ccda_source'] for r in rows))}\n\n"
    f"  Word count:  min={min(word_counts)}  max={max(word_counts)}  "
    f"mean={np.mean(word_counts):.0f}  median={np.median(word_counts):.0f}\n"
    f"  Total lines: min={min(total_lines)}  max={max(total_lines)}  "
    f"mean={np.mean(total_lines):.0f}  median={np.median(total_lines):.0f}\n"
    f"  Doc lines:   min={min(doc_lines)}  max={max(doc_lines)}  "
    f"mean={np.mean(doc_lines):.0f}  median={np.median(doc_lines):.0f}\n"
    f"  Pat lines:   min={min(pat_lines)}  max={max(pat_lines)}  "
    f"mean={np.mean(pat_lines):.0f}  median={np.median(pat_lines):.0f}\n\n"
    f"  Ages:        min={min(ages)}  max={max(ages)}  "
    f"mean={np.mean(ages):.0f}  median={np.median(ages):.0f}\n"
    f"  Genders:     {dict(g_c)}\n\n"
    f"  Present symptoms:  min={min(present)}  max={max(present)}  "
    f"mean={np.mean(present):.1f}  median={np.median(present):.0f}\n"
    f"  Negated symptoms:  min={min(negated)}  max={max(negated)}  "
    f"mean={np.mean(negated):.1f}  median={np.median(negated):.0f}\n\n"
    f"  Gen time (s):  min={min(gen_times):.1f}  max={max(gen_times):.1f}  "
    f"mean={np.mean(gen_times):.1f}  median={np.median(gen_times):.1f}\n"
    f"  Total runtime: {sum(gen_times)/60:.1f} minutes\n"
)
if total_tok:
    stats_text += (
        f"\n  Tokens (491 rows w/ data):\n"
        f"    Prompt:     mean={np.mean(prompt_tok):.0f}  median={np.median(prompt_tok):.0f}\n"
        f"    Completion: mean={np.mean(comp_tok):.0f}  median={np.median(comp_tok):.0f}\n"
        f"    Total:      mean={np.mean(total_tok):.0f}  median={np.median(total_tok):.0f}\n"
        f"    Grand total: {sum(total_tok):,} tokens\n"
    )
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=11, fontfamily="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=_BLUE_PALE, edgecolor=_BLUE_LIGHT))
fig.tight_layout()
save(fig, "19_summary_statistics")

# ══════════════════════════════════════════════════════════════════════
#  20. Quality checks (text figure)
# ══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("off")

checks = []
checks.append(("Total = 500", len(rows) == 500, f"{len(rows)} rows"))
unique_ccda = len(set(r["ccda_source"] for r in rows))
checks.append(("Unique CCDAs", unique_ccda == len(rows), f"{unique_ccda} unique"))

ct_ok = (ct_c.get("healthy", 0) == 252 and ct_c.get("flu_like", 0) == 35 and
         ct_c.get("novel_virus", 0) == 50 and ct_c.get("differential", 0) == 163)
checks.append(("Distribution matches plan", ct_ok,
               f"H={ct_c.get('healthy',0)} F={ct_c.get('flu_like',0)} "
               f"NV={ct_c.get('novel_virus',0)} D={ct_c.get('differential',0)}"))

h_pres = [int(r["present_symptom_count"]) for r in rows if r["case_type"] == "healthy"]
checks.append(("Healthy: 0 present symptoms", all(p == 0 for p in h_pres),
               f"min={min(h_pres)} max={max(h_pres)}"))

nv_pres = [int(r["present_symptom_count"]) for r in rows if r["case_type"] == "novel_virus"]
checks.append(("Novel virus: >=5 present", all(p >= 5 for p in nv_pres),
               f"min={min(nv_pres)} max={max(nv_pres)}"))

checks.append(("All transcripts non-empty", min(word_counts) > 50,
               f"min words={min(word_counts)}"))

meta_exist = sum(1 for m in meta_data if m)
checks.append(("All metadata files exist", meta_exist == len(rows),
               f"{meta_exist}/{len(rows)}"))

unique_fn = len(set(r["filename"] for r in rows))
checks.append(("Unique filenames", unique_fn == len(rows), f"{unique_fn} unique"))

sub_c = {k: v for k, v in Counter(subtypes).items() if k}
sub_ok = (sub_c.get("musculoskeletal", 0) == 63 and sub_c.get("gastrointestinal", 0) == 55
          and sub_c.get("dermatological", 0) == 29 and sub_c.get("neurological", 0) == 16)
checks.append(("Subtype counts match", sub_ok,
               f"MSK={sub_c.get('musculoskeletal',0)} GI={sub_c.get('gastrointestinal',0)} "
               f"Derm={sub_c.get('dermatological',0)} Neuro={sub_c.get('neurological',0)}"))

ratio = np.mean(doc_lines) / np.mean(pat_lines) if np.mean(pat_lines) > 0 else 0
checks.append(("Doc/Pat line ratio ~1.0", 0.7 < ratio < 1.4, f"ratio={ratio:.2f}"))

check_text = f"{'='*55}\n  QUALITY CHECKS\n{'='*55}\n\n"
for label, passed, detail in checks:
    icon = "PASS" if passed else "FAIL"
    check_text += f"  [{icon}]  {label}\n           {detail}\n\n"
passed_count = sum(1 for _, p, _ in checks if p)
check_text += f"\n  Result: {passed_count}/{len(checks)} checks passed"

bg = _BLUE_PALE if passed_count == len(checks) else "#FFF3E0"
ec = _BLUE_PRIMARY if passed_count == len(checks) else "#FFA726"
ax.text(0.02, 0.98, check_text, transform=ax.transAxes,
        fontsize=11, fontfamily="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=bg, edgecolor=ec))
fig.tight_layout()
save(fig, "20_quality_checks")

# ══════════════════════════════════════════════════════════════════════
#  21. Top 10 present symptoms — whole dataset
# ══════════════════════════════════════════════════════════════════════
top10_all = Counter(all_present_names).most_common(10)
labels_all = [s.title() for s, _ in top10_all]
vals_all   = [v for _, v in top10_all]
total_pats = len([m for m in meta_data if m])

# gradient: darkest bar = highest count (bottom of chart = highest)
cmap_all   = plt.get_cmap("Blues")
grad_all   = [cmap_all(0.40 + 0.55 * i / max(len(vals_all) - 1, 1))
              for i in range(len(vals_all))]  # light→dark left to right in value
grad_all_r = grad_all[::-1]  # reversed so highest bar (bottom of barh) is darkest

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(labels_all[::-1], vals_all[::-1], color=grad_all, linewidth=0)
for bar, val in zip(bars, vals_all[::-1]):
    pct = val / total_pats * 100
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
            f"{val}  ({pct:.1f}%)", va="center", fontsize=10, color=_TEXT)
ax.set_xlabel("Number of Transcripts")
ax.set_title("Top 10 Present Symptoms — Full Dataset",
             fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlim(0, max(vals_all) * 1.22)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False)
fig.tight_layout()
save(fig, "21_top10_symptoms_all")

# ══════════════════════════════════════════════════════════════════════
#  22. Top 10 present symptoms — Novel Virus cases only
# ══════════════════════════════════════════════════════════════════════
nv_syms = type_present_names.get("novel_virus", [])
nv_count = sum(1 for r in rows if r["case_type"] == "novel_virus")
top10_nv = Counter(nv_syms).most_common(10)
labels_nv = [s.title() for s, _ in top10_nv]
vals_nv   = [v for _, v in top10_nv]

cmap_nv  = plt.get_cmap("Blues")
grad_nv  = [cmap_nv(0.40 + 0.55 * i / max(len(vals_nv) - 1, 1))
             for i in range(len(vals_nv))]

fig, ax = plt.subplots(figsize=(10, 6))
bars_nv = ax.barh(labels_nv[::-1], vals_nv[::-1], color=grad_nv, linewidth=0)
for bar, val in zip(bars_nv, vals_nv[::-1]):
    pct = val / nv_count * 100
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val}  ({pct:.1f}%)", va="center", fontsize=10, color=_TEXT)
ax.set_xlabel("Number of Transcripts")
ax.set_title(f"Top 10 Present Symptoms — Novel Virus Cases (n={nv_count})",
             fontweight="bold", fontsize=14, color=_TEXT)
ax.set_xlim(0, max(vals_nv) * 1.3)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(left=False)
fig.tight_layout()
save(fig, "22_top10_symptoms_novel_virus")

print(f"\nDone — {saved} charts saved to {CHART_DIR}")
