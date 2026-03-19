"""
Dataset Characteristic Visualizations for Batch 500 Transcript Index
Generates a multi-panel figure with key dataset statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "transcript_index.csv"
df = pd.read_csv(CSV_PATH)

# ── Styling ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

COLORS_CASE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
COLORS_DIFF = ["#8172B3", "#64B5CD", "#CCB974", "#72B5A4", "#E18D96"]

# Friendly labels for case_type
CASE_LABELS = {
    "healthy": "Healthy",
    "differential": "Differential Dx",
    "flu_like": "Flu-Like",
    "novel_virus": "Novel Virus",
}

fig = plt.figure(figsize=(20, 14))
fig.suptitle("Batch 500 — Synthetic Transcript Dataset Overview  (n = {})".format(len(df)),
             fontsize=17, fontweight="bold", y=0.98)

# ── 1. Pie chart: Case Type Distribution ──────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
case_counts = df["case_type"].value_counts()
labels = [CASE_LABELS.get(c, c) for c in case_counts.index]
wedges, texts, autotexts = ax1.pie(
    case_counts, labels=labels, autopct=lambda p: f"{p:.1f}%\n({int(round(p*len(df)/100))})",
    colors=COLORS_CASE, startangle=140, textprops={"fontsize": 10},
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
for at in autotexts:
    at.set_fontsize(9)
ax1.set_title("Case Type Distribution")

# ── 2. Bar chart: Differential Subtypes ───────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
diff_df = df[df["case_type"] == "differential"]
sub_counts = diff_df["differential_subtype"].value_counts()
sub_labels = [s.capitalize() for s in sub_counts.index]
bars = ax2.barh(sub_labels, sub_counts.values, color=COLORS_DIFF[:len(sub_counts)],
                edgecolor="white", linewidth=1.2)
ax2.bar_label(bars, fmt="%d", padding=4, fontsize=10)
ax2.set_xlabel("Count")
ax2.set_title("Differential Diagnosis Subtypes")
ax2.invert_yaxis()

# ── 3. Histogram: Total Token Count ──────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
tokens = df["total_tokens"].dropna()
ax3.hist(tokens, bins=25, color="#4C72B0", edgecolor="white", linewidth=0.8, alpha=0.85)
ax3.axvline(tokens.mean(), color="#C44E52", ls="--", lw=2, label=f"Mean = {tokens.mean():.0f}")
ax3.axvline(tokens.median(), color="#DD8452", ls=":", lw=2, label=f"Median = {tokens.median():.0f}")
ax3.legend(fontsize=9)
ax3.set_xlabel("Total Tokens")
ax3.set_ylabel("Frequency")
ax3.set_title(f"Total Token Distribution  (n = {len(tokens)})")

# ── 4. Box plot: Token Counts by Case Type ────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
case_order = ["healthy", "flu_like", "novel_virus", "differential"]
token_groups = [df.loc[(df["case_type"] == ct) & df["total_tokens"].notna(), "total_tokens"].values
                for ct in case_order]
bp = ax4.boxplot(token_groups, patch_artist=True, tick_labels=[CASE_LABELS[c] for c in case_order],
                 medianprops=dict(color="black", lw=1.5))
for patch, color in zip(bp["boxes"], COLORS_CASE):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel("Total Tokens")
ax4.set_title("Token Count by Case Type")

# ── 5. Grouped bar: Present vs Negated Symptom Counts ────────────────────
ax5 = fig.add_subplot(2, 3, 5)
sym_stats = df.groupby("case_type")[["present_symptom_count", "negated_symptom_count"]].mean()
sym_stats = sym_stats.reindex(case_order)
x = np.arange(len(case_order))
w = 0.35
b1 = ax5.bar(x - w/2, sym_stats["present_symptom_count"], w, label="Present", color="#55A868", edgecolor="white")
b2 = ax5.bar(x + w/2, sym_stats["negated_symptom_count"], w, label="Negated", color="#C44E52", edgecolor="white")
ax5.bar_label(b1, fmt="%.1f", padding=2, fontsize=9)
ax5.bar_label(b2, fmt="%.1f", padding=2, fontsize=9)
ax5.set_xticks(x)
ax5.set_xticklabels([CASE_LABELS[c] for c in case_order])
ax5.set_ylabel("Mean Symptom Count")
ax5.set_title("Avg Present vs Negated Symptoms")
ax5.legend(fontsize=9)

# ── 6. Histogram: Generation Time ────────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
gen_time = df["generation_time_seconds"].dropna()
ax6.hist(gen_time, bins=30, color="#8172B3", edgecolor="white", linewidth=0.8, alpha=0.85)
ax6.axvline(gen_time.mean(), color="#C44E52", ls="--", lw=2, label=f"Mean = {gen_time.mean():.1f}s")
ax6.axvline(gen_time.median(), color="#DD8452", ls=":", lw=2, label=f"Median = {gen_time.median():.1f}s")
ax6.legend(fontsize=9)
ax6.set_xlabel("Generation Time (seconds)")
ax6.set_ylabel("Frequency")
ax6.set_title("Generation Time Distribution")

plt.tight_layout(rect=[0, 0, 1, 0.94])

OUT = Path(__file__).parent / "dataset_overview.png"
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT}")
plt.show()
