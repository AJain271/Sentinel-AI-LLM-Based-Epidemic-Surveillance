"""
Batch 500 — Initial Data Visualization & Quality Checks
Generates a multi-page dashboard saved as batch_500_dashboard.png
"""

import csv, json, os, re
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── paths ────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
CSV_PATH  = BASE / "transcript_index.csv"
TX_DIR    = BASE / "transcripts"
META_DIR  = BASE / "metadata"
OUT_PATH  = BASE / "batch_500_dashboard.png"

# ── load CSV ─────────────────────────────────────────────────────────
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} rows from CSV")

# ── load metadata JSONs ──────────────────────────────────────────────
meta_data = []
for r in rows:
    mp = META_DIR / r["metadata_filename"]
    if mp.exists():
        with open(mp, encoding="utf-8") as f:
            meta_data.append(json.load(f))
    else:
        meta_data.append(None)

print(f"Loaded {sum(1 for m in meta_data if m)} metadata JSONs")

# ── derived arrays ───────────────────────────────────────────────────
case_types = ["Healthy","Influenza","Novel Virus","Varied Cases"]
subtypes   = [r["differential_subtype"] for r in rows]
present    = [int(r["present_symptom_count"]) for r in rows]
negated    = [int(r["negated_symptom_count"]) for r in rows]
total_symp = [p + n for p, n in zip(present, negated)]

# ages & genders from metadata
ages    = [m["demographics"]["age"] for m in meta_data if m]
genders = [m["demographics"]["gender"] for m in meta_data if m]

# transcript line counts
total_lines  = [m["transcript_stats"]["total_lines"] for m in meta_data if m]
doc_lines    = [m["transcript_stats"]["doctor_lines"] for m in meta_data if m]
pat_lines    = [m["transcript_stats"]["patient_lines"] for m in meta_data if m]

# word counts from actual transcripts
word_counts = []
char_counts = []
for r in rows:
    tp = TX_DIR / r["filename"]
    if tp.exists():
        text = tp.read_text(encoding="utf-8")
        word_counts.append(len(text.split()))
        char_counts.append(len(text))
    else:
        word_counts.append(0)
        char_counts.append(0)

# tokens (skip blank test rows)
prompt_tok  = [int(r["prompt_tokens"]) for r in rows if r["prompt_tokens"]]
comp_tok    = [int(r["completion_tokens"]) for r in rows if r["completion_tokens"]]
total_tok   = [int(r["total_tokens"]) for r in rows if r["total_tokens"]]
gen_times   = [float(r["generation_time_seconds"]) for r in rows]

# location hubs
hubs = [m["location"]["hospital"] for m in meta_data if m]

# negated symptom names (for all case types)
all_negated_names = []
for m in meta_data:
    if m:
        for s in m.get("negated_symptoms", []):
            # strip parenthetical description
            clean = re.sub(r"\s*\(.*?\)", "", s).strip()
            all_negated_names.append(clean)

# present symptom names
all_present_names = []
for m in meta_data:
    if m:
        for s in m.get("present_symptoms", []):
            clean = re.sub(r"\s*\(.*?\)", "", s).strip()
            all_present_names.append(clean)

# per-type word counts
type_words = {}
for ct, wc in zip(case_types, word_counts):
    type_words.setdefault(ct, []).append(wc)

# per-subtype present symptom counts (differential only)
sub_present = {}
for r in rows:
    if r["differential_subtype"]:
        sub_present.setdefault(r["differential_subtype"], []).append(
            int(r["present_symptom_count"])
        )

# ── colour palette ───────────────────────────────────────────────────
TYPE_COLORS = {
    "Healthy":      "#4CAF50",
    "Influenza":     "#FF9800",
    "Novel Virus":  "#F44336",
    "Varied Cases": "#2196F3",
}
SUB_COLORS = {
    "Musculoskeletal":  "#1565C0",
    "Gastrointestinal": "#7B1FA2",
    "Dermatological":   "#00897B",
    "Neurological":     "#E65100",
}

# ══════════════════════════════════════════════════════════════════════
#  BUILD DASHBOARD
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(28, 36), facecolor="white")
fig.suptitle("Batch 500 — Synthetic Transcript Dashboard",
             fontsize=24, fontweight="bold", y=0.995)

gs = gridspec.GridSpec(6, 4, figure=fig, hspace=0.38, wspace=0.32,
                       left=0.05, right=0.97, top=0.975, bottom=0.025)

# ── 1. Case-type distribution (pie) ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ct_counts = Counter(case_types)
labels_ct = list(ct_counts.keys())
sizes_ct  = list(ct_counts.values())
colors_ct = [TYPE_COLORS.get(l, "#999") for l in labels_ct]
wedges, texts, autotexts = ax1.pie(
    sizes_ct, labels=[f"{l}\n({v})" for l, v in zip(labels_ct, sizes_ct)],
    colors=colors_ct, autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 9})
ax1.set_title("Case Type Distribution", fontweight="bold")

# ── 2. Differential subtype breakdown (bar) ──────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sub_counts = {k: v for k, v in Counter(subtypes).items() if k}
sub_labels = sorted(sub_counts.keys())
sub_vals   = [sub_counts[s] for s in sub_labels]
sub_cols   = [SUB_COLORS.get(s, "#999") for s in sub_labels]
bars = ax2.bar(sub_labels, sub_vals, color=sub_cols, edgecolor="white")
ax2.bar_label(bars, fontsize=10, fontweight="bold")
ax2.set_title("Differential Subtypes", fontweight="bold")
ax2.set_ylabel("Count")
ax2.tick_params(axis="x", rotation=25)

# ── 3. Present symptom count by case type (box) ─────────────────────
ax3 = fig.add_subplot(gs[0, 2])
type_order = ["Healthy", "Influenza", "Novel Virus", "Varied Cases"]
present_by_type = {t: [] for t in type_order}
for ct, p in zip(case_types, present):
    present_by_type[ct].append(p)
bp_data = [present_by_type[t] for t in type_order]
bp = ax3.boxplot(bp_data, labels=type_order, patch_artist=True, widths=0.6)
for patch, t in zip(bp["boxes"], type_order):
    patch.set_facecolor(TYPE_COLORS[t])
    patch.set_alpha(0.7)
ax3.set_title("Present Symptoms by Case Type", fontweight="bold")
ax3.set_ylabel("Count")
ax3.tick_params(axis="x", rotation=20)

# ── 4. Negated symptom count by case type (box) ─────────────────────
ax4 = fig.add_subplot(gs[0, 3])
negated_by_type = {t: [] for t in type_order}
for ct, n in zip(case_types, negated):
    negated_by_type[ct].append(n)
bp_data2 = [negated_by_type[t] for t in type_order]
bp2 = ax4.boxplot(bp_data2, labels=type_order, patch_artist=True, widths=0.6)
for patch, t in zip(bp2["boxes"], type_order):
    patch.set_facecolor(TYPE_COLORS[t])
    patch.set_alpha(0.7)
ax4.set_title("Negated Symptoms by Case Type", fontweight="bold")
ax4.set_ylabel("Count")
ax4.tick_params(axis="x", rotation=20)

# ── 5. Word count distribution (histogram) ──────────────────────────
ax5 = fig.add_subplot(gs[1, 0])
ax5.hist(word_counts, bins=30, color="#607D8B", edgecolor="white", alpha=0.85)
ax5.axvline(np.median(word_counts), color="red", linestyle="--", label=f"median={int(np.median(word_counts))}")
ax5.axvline(np.mean(word_counts), color="blue", linestyle="--", label=f"mean={int(np.mean(word_counts))}")
ax5.set_title("Transcript Word Count Distribution", fontweight="bold")
ax5.set_xlabel("Words")
ax5.set_ylabel("Frequency")
ax5.legend(fontsize=8)

# ── 6. Word count by case type (box) ────────────────────────────────
ax6 = fig.add_subplot(gs[1, 1])
wc_data = [type_words.get(t, []) for t in type_order]
bp3 = ax6.boxplot(wc_data, labels=type_order, patch_artist=True, widths=0.6)
for patch, t in zip(bp3["boxes"], type_order):
    patch.set_facecolor(TYPE_COLORS[t])
    patch.set_alpha(0.7)
ax6.set_title("Word Count by Case Type", fontweight="bold")
ax6.set_ylabel("Words")
ax6.tick_params(axis="x", rotation=20)

# ── 7. Transcript line counts (doc vs patient) ──────────────────────
ax7 = fig.add_subplot(gs[1, 2])
ax7.scatter(doc_lines, pat_lines, alpha=0.4, s=15, c="#37474F")
max_val = max(max(doc_lines), max(pat_lines)) + 2
ax7.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="1:1 line")
ax7.set_title("Doctor vs Patient Lines", fontweight="bold")
ax7.set_xlabel("Doctor lines")
ax7.set_ylabel("Patient lines")
ax7.legend(fontsize=8)

# ── 8. Total lines histogram ────────────────────────────────────────
ax8 = fig.add_subplot(gs[1, 3])
ax8.hist(total_lines, bins=30, color="#78909C", edgecolor="white", alpha=0.85)
ax8.axvline(np.median(total_lines), color="red", linestyle="--",
            label=f"median={int(np.median(total_lines))}")
ax8.set_title("Total Lines per Transcript", fontweight="bold")
ax8.set_xlabel("Lines")
ax8.set_ylabel("Frequency")
ax8.legend(fontsize=8)

# ── 9. Age distribution ─────────────────────────────────────────────
ax9 = fig.add_subplot(gs[2, 0])
ax9.hist(ages, bins=20, color="#8BC34A", edgecolor="white", alpha=0.85)
ax9.axvline(np.median(ages), color="red", linestyle="--",
            label=f"median={int(np.median(ages))}")
ax9.set_title("Patient Age Distribution", fontweight="bold")
ax9.set_xlabel("Age")
ax9.set_ylabel("Frequency")
ax9.legend(fontsize=8)

# ── 10. Gender split ─────────────────────────────────────────────────
ax10 = fig.add_subplot(gs[2, 1])
g_counts = Counter(genders)
g_labels = list(g_counts.keys())
g_vals   = list(g_counts.values())
g_colors = ["#42A5F5" if g == "M" else "#EF5350" for g in g_labels]
bars_g = ax10.bar(g_labels, g_vals, color=g_colors, edgecolor="white")
ax10.bar_label(bars_g, fontsize=12, fontweight="bold")
ax10.set_title("Gender Distribution", fontweight="bold")
ax10.set_ylabel("Count")

# ── 11. Hospital hub distribution ────────────────────────────────────
ax11 = fig.add_subplot(gs[2, 2:4])
hub_counts = Counter(hubs)
hub_sorted = hub_counts.most_common()
hub_labels = [h for h, _ in hub_sorted]
hub_vals   = [v for _, v in hub_sorted]
bars_h = ax11.barh(hub_labels, hub_vals, color="#5C6BC0", edgecolor="white")
ax11.bar_label(bars_h, fontsize=9, fontweight="bold")
ax11.set_title("Hospital Hub Distribution", fontweight="bold")
ax11.set_xlabel("Count")
ax11.invert_yaxis()

# ── 12. Generation time distribution ────────────────────────────────
ax12 = fig.add_subplot(gs[3, 0])
ax12.hist(gen_times, bins=30, color="#FF7043", edgecolor="white", alpha=0.85)
ax12.axvline(np.median(gen_times), color="blue", linestyle="--",
             label=f"median={np.median(gen_times):.1f}s")
ax12.set_title("Generation Time per Transcript", fontweight="bold")
ax12.set_xlabel("Seconds")
ax12.set_ylabel("Frequency")
ax12.legend(fontsize=8)

# ── 13. Total tokens distribution ───────────────────────────────────
ax13 = fig.add_subplot(gs[3, 1])
if total_tok:
    ax13.hist(total_tok, bins=30, color="#AB47BC", edgecolor="white", alpha=0.85)
    ax13.axvline(np.median(total_tok), color="red", linestyle="--",
                 label=f"median={int(np.median(total_tok))}")
    ax13.set_title("Total Tokens per Call", fontweight="bold")
    ax13.set_xlabel("Tokens")
    ax13.set_ylabel("Frequency")
    ax13.legend(fontsize=8)

# ── 14. Prompt vs Completion tokens scatter ─────────────────────────
ax14 = fig.add_subplot(gs[3, 2])
if prompt_tok and comp_tok:
    ax14.scatter(prompt_tok, comp_tok, alpha=0.4, s=15, c="#7E57C2")
    ax14.set_title("Prompt vs Completion Tokens", fontweight="bold")
    ax14.set_xlabel("Prompt tokens")
    ax14.set_ylabel("Completion tokens")

# ── 15. Top 20 present symptoms ─────────────────────────────────────
ax15 = fig.add_subplot(gs[3, 3])
pres_top = Counter(all_present_names).most_common(20)
pres_labels = [s for s, _ in pres_top]
pres_vals   = [v for _, v in pres_top]
ax15.barh(pres_labels[::-1], pres_vals[::-1], color="#26A69A", edgecolor="white")
ax15.set_title("Top 20 Present Symptoms", fontweight="bold")
ax15.set_xlabel("Occurrences")
ax15.tick_params(axis="y", labelsize=7)

# ── 16. Top 20 negated symptoms ─────────────────────────────────────
ax16 = fig.add_subplot(gs[4, 0:2])
neg_top = Counter(all_negated_names).most_common(20)
neg_labels = [s for s, _ in neg_top]
neg_vals   = [v for _, v in neg_top]
ax16.barh(neg_labels[::-1], neg_vals[::-1], color="#EF5350", edgecolor="white")
ax16.set_title("Top 20 Negated Symptoms", fontweight="bold")
ax16.set_xlabel("Occurrences")
ax16.tick_params(axis="y", labelsize=7)

# ── 17. Present symptoms by differential subtype (box) ──────────────
ax17 = fig.add_subplot(gs[4, 2])
sub_order = sorted(sub_present.keys())
bp4 = ax17.boxplot([sub_present[s] for s in sub_order],
                   labels=sub_order, patch_artist=True, widths=0.6)
for patch, s in zip(bp4["boxes"], sub_order):
    patch.set_facecolor(SUB_COLORS.get(s, "#999"))
    patch.set_alpha(0.7)
ax17.set_title("Present Symptoms\nby Differential Subtype", fontweight="bold")
ax17.set_ylabel("Count")
ax17.tick_params(axis="x", rotation=25, labelsize=8)

# ── 18. Word count vs present symptoms scatter ──────────────────────
ax18 = fig.add_subplot(gs[4, 3])
colors_scatter = [TYPE_COLORS.get(ct, "#999") for ct in case_types]
ax18.scatter(present, word_counts, alpha=0.45, s=18, c=colors_scatter)
ax18.set_title("Word Count vs Present Symptoms", fontweight="bold")
ax18.set_xlabel("Present symptom count")
ax18.set_ylabel("Word count")
# legend
for t in type_order:
    ax18.scatter([], [], c=TYPE_COLORS[t], label=t, s=30)
ax18.legend(fontsize=7, loc="upper left")

# ── 19. Summary stats text panel ────────────────────────────────────
ax19 = fig.add_subplot(gs[5, 0:2])
ax19.axis("off")
stats_text = (
    f"{'═'*50}\n"
    f"  BATCH 500 — SUMMARY STATISTICS\n"
    f"{'═'*50}\n\n"
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
    f"  Genders:     {dict(Counter(genders))}\n\n"
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
ax19.text(0.02, 0.98, stats_text, transform=ax19.transAxes,
          fontsize=9.5, fontfamily="monospace", verticalalignment="top",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

# ── 20. Quality checks panel ────────────────────────────────────────
ax20 = fig.add_subplot(gs[5, 2:4])
ax20.axis("off")

# Run checks
checks = []
# 1. Total count
checks.append(("Total = 500", len(rows) == 500, f"{len(rows)} rows"))
# 2. No duplicate CCDAs
unique_ccda = len(set(r["ccda_source"] for r in rows))
checks.append(("Unique CCDAs", unique_ccda == len(rows), f"{unique_ccda} unique"))
# 3. Distribution matches plan
ct_ok = (ct_counts.get("Healthy", 0) == 252 and
         ct_counts.get("Influenza", 0) == 35 and
         ct_counts.get("Novel Virus", 0) == 50 and
         ct_counts.get("Varied Cases", 0) == 163)
checks.append(("Distribution matches plan", ct_ok,
               f"H={ct_counts.get('Healthy',0)} F={ct_counts.get('Influenza',0)} "
               f"NV={ct_counts.get('Novel Virus',0)} D={ct_counts.get('Varied Cases',0)}"))
# 4. Healthy have 0 present symptoms
h_present = [int(r["present_symptom_count"]) for r in rows if r["case_type"] == "Healthy"]
checks.append(("Healthy: 0 present symptoms", all(p == 0 for p in h_present),
               f"min={min(h_present)} max={max(h_present)}"))
# 5. Novel virus >= 5 present (hallmarks)
nv_present = [int(r["present_symptom_count"]) for r in rows if r["case_type"] == "Novel Virus"]
checks.append(("Novel virus: ≥5 present", all(p >= 5 for p in nv_present),
               f"min={min(nv_present)} max={max(nv_present)}"))
# 6. All transcripts non-empty
min_wc = min(word_counts)
checks.append(("All transcripts non-empty", min_wc > 50,
               f"min words={min_wc}"))
# 7. All metadata files exist
meta_exist = sum(1 for m in meta_data if m)
checks.append(("All metadata files exist", meta_exist == len(rows),
               f"{meta_exist}/{len(rows)}"))
# 8. No duplicate filenames
unique_fn = len(set(r["filename"] for r in rows))
checks.append(("Unique filenames", unique_fn == len(rows),
               f"{unique_fn} unique"))
# 9. Differential subtype counts
sub_ok = (sub_counts.get("Musculoskeletal", 0) == 63 and
          sub_counts.get("Gastrointestinal", 0) == 55 and
          sub_counts.get("Dermatological", 0) == 29 and
          sub_counts.get("Neurological", 0) == 16)
checks.append(("Subtype counts match", sub_ok,
               f"MSK={sub_counts.get('Musculoskeletal',0)} GI={sub_counts.get('Gastrointestinal',0)} "
               f"Derm={sub_counts.get('Dermatological',0)} Neuro={sub_counts.get('Neurological',0)}"))
# 10. Doctor/patient line balance
ratio = np.mean(doc_lines) / np.mean(pat_lines) if np.mean(pat_lines) > 0 else 0
checks.append(("Doc/Pat line ratio ~1.0", 0.7 < ratio < 1.4,
               f"ratio={ratio:.2f}"))

check_text = f"{'═'*50}\n  QUALITY CHECKS\n{'═'*50}\n\n"
for label, passed, detail in checks:
    icon = "✓ PASS" if passed else "✗ FAIL"
    check_text += f"  [{icon}]  {label}\n           {detail}\n\n"

passed_count = sum(1 for _, p, _ in checks if p)
check_text += f"\n  Result: {passed_count}/{len(checks)} checks passed"

ax20.text(0.02, 0.98, check_text, transform=ax20.transAxes,
          fontsize=9.5, fontfamily="monospace", verticalalignment="top",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9" if passed_count == len(checks) else "#FFF3E0",
                    edgecolor="#66BB6A" if passed_count == len(checks) else "#FFA726"))

# ── save ─────────────────────────────────────────────────────────────
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nDashboard saved → {OUT_PATH}")
plt.close()
