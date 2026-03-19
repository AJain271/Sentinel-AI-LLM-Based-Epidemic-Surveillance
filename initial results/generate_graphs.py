"""
Science Fair Publication-Quality Graphs
Generates polished figures for the SNHS Symposium project.
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import silhouette_samples

# --- STYLE ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
})

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(SCRIPT_DIR, "clustering_results.csv"))
with open(os.path.join(SCRIPT_DIR, "..", "Clustering", "cluster_metrics.json"), 'r') as f:
    metrics = json.load(f)

SYMPTOM_COLS = [c for c in df.columns if c not in [
    'Filename', 'Predicted_Category', 'Cluster', 'TSNE_1', 'TSNE_2', 'Silhouette'
]]

CATEGORY_COLORS = {
    'RES': '#3498db', 'CAR': '#e74c3c', 'MSK': '#2ecc71',
    'GAS': '#f39c12', 'DER': '#9b59b6'
}

# Blue palette for clusters
CLUSTER_BLUES = ['#bdc3c7', '#08306b', '#2171b5', '#6baed6', '#9ecae1', '#c6dbef']

OUT = os.path.join(SCRIPT_DIR, "graphs")
os.makedirs(OUT, exist_ok=True)

# =============================================================
# GRAPH 1: DBSCAN Cluster Map (Blue Theme)
# =============================================================
print("1/7  DBSCAN Cluster Map...", flush=True)

fig, ax = plt.subplots(figsize=(10, 8))

# Noise first (grey), then clusters in blue shades
unique_clusters = sorted(df['Cluster'].unique())
for cl in unique_clusters:
    mask = df['Cluster'] == cl
    sub = df[mask]
    if cl == -1:
        ax.scatter(sub['TSNE_1'], sub['TSNE_2'],
                   c='#d5d8dc', s=25, alpha=0.4, label='Noise', zorder=1,
                   edgecolors='#aab7b8', linewidths=0.3)
    else:
        color = CLUSTER_BLUES[cl + 1] if cl + 1 < len(CLUSTER_BLUES) else CLUSTER_BLUES[-1]
        ax.scatter(sub['TSNE_1'], sub['TSNE_2'],
                   c=color, s=55, alpha=0.85, label=f'Cluster {cl}', zorder=2,
                   edgecolors='#2c3e50', linewidths=0.4)

ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")
ax.set_title("DBSCAN Patient Clustering\n(t-SNE Projection of Symptom Vectors)")
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=11)
ax.grid(True, alpha=0.15)
sns.despine()
fig.savefig(f"{OUT}/1_dbscan_cluster_map.png")
plt.close()

# =============================================================
# GRAPH 2: Silhouette & Davies-Bouldin Metrics
# =============================================================
print("2/7  Clustering Quality Metrics...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Silhouette gauge ---
sil = metrics.get('silhouette_coefficient', 0)
ax = axes[0]
categories = ['Poor\n(-1 to 0)', 'Weak\n(0 to 0.25)', 'Fair\n(0.25 to 0.5)',
              'Good\n(0.5 to 0.75)', 'Strong\n(0.75 to 1)']
cat_colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60']
bar_vals = [0.25, 0.25, 0.25, 0.25, 0.25]  # each segment is 0.25 wide on a -1 to 1 scale

# Simpler: bar chart showing the score vs reference
ax.barh(['Score'], [sil], color='#2980b9', height=0.5, edgecolor='#1a5276', linewidth=1.5)
ax.axvline(x=0, color='#e74c3c', linewidth=1.5, linestyle='--', alpha=0.7)
ax.axvline(x=0.25, color='#f39c12', linewidth=1, linestyle=':', alpha=0.5)
ax.axvline(x=0.50, color='#2ecc71', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlim(-0.2, 1.0)
ax.set_title("Silhouette Coefficient")
ax.text(sil + 0.02, 0, f'{sil:.3f}', va='center', fontweight='bold', fontsize=14, color='#2c3e50')
# Region labels
ax.text(-0.1, -0.35, 'Poor', ha='center', fontsize=9, color='#e74c3c')
ax.text(0.125, -0.35, 'Weak', ha='center', fontsize=9, color='#f39c12')
ax.text(0.375, -0.35, 'Fair', ha='center', fontsize=9, color='#f1c40f')
ax.text(0.625, -0.35, 'Good', ha='center', fontsize=9, color='#2ecc71')
ax.text(0.875, -0.35, 'Strong', ha='center', fontsize=9, color='#27ae60')
ax.set_xlabel("Score (closer to 1 = better)")
sns.despine(ax=ax)

# --- Davies-Bouldin gauge ---
dbi = metrics.get('davies_bouldin_index', 0)
ax2 = axes[1]
ax2.barh(['Score'], [dbi], color='#e67e22', height=0.5, edgecolor='#a04000', linewidth=1.5)
ax2.axvline(x=1.0, color='#f39c12', linewidth=1, linestyle=':', alpha=0.5)
ax2.axvline(x=2.0, color='#e74c3c', linewidth=1, linestyle=':', alpha=0.5)
ax2.set_xlim(0, 3.0)
ax2.set_title("Davies-Bouldin Index")
ax2.text(dbi + 0.05, 0, f'{dbi:.3f}', va='center', fontweight='bold', fontsize=14, color='#2c3e50')
ax2.text(0.5, -0.35, 'Good', ha='center', fontsize=9, color='#2ecc71')
ax2.text(1.5, -0.35, 'Fair', ha='center', fontsize=9, color='#f39c12')
ax2.text(2.5, -0.35, 'Poor', ha='center', fontsize=9, color='#e74c3c')
ax2.set_xlabel("Score (closer to 0 = better)")
sns.despine(ax=ax2)

plt.suptitle("Clustering Quality Metrics", fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
fig.savefig(f"{OUT}/2_clustering_metrics.png")
plt.close()

# =============================================================
# GRAPH 3: Per-Sample Silhouette Plot (classic knife plot)
# =============================================================
print("3/7  Silhouette Knife Plot...", flush=True)

df_scored = df[df['Cluster'] != -1].copy()

if 'Silhouette' in df_scored.columns and df_scored['Silhouette'].notna().any():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_lower = 0
    unique_cl = sorted(df_scored['Cluster'].unique())
    
    for i, cl in enumerate(unique_cl):
        cl_sil = df_scored[df_scored['Cluster'] == cl]['Silhouette'].sort_values()
        size = len(cl_sil)
        y_upper = y_lower + size
        
        color = CLUSTER_BLUES[cl + 1] if cl + 1 < len(CLUSTER_BLUES) else CLUSTER_BLUES[-1]
        ax.barh(range(y_lower, y_upper), cl_sil.values,
                height=1.0, color=color, edgecolor='none', alpha=0.85)
        ax.text(-0.02, y_lower + size / 2, f'Cluster {cl}',
                fontweight='bold', va='center', ha='right', fontsize=11)
        y_lower = y_upper + 2
    
    ax.axvline(x=sil, color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'Mean = {sil:.3f}')
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Patients (sorted within cluster)")
    ax.set_title("Per-Patient Silhouette Analysis")
    ax.set_yticks([])
    ax.legend(fontsize=11)
    sns.despine(left=True)
    fig.savefig(f"{OUT}/3_silhouette_knife.png")
    plt.close()

# =============================================================
# GRAPH 4: Symptom Prevalence by Cluster (Heatmap)
# =============================================================
print("4/7  Symptom Prevalence Heatmap...", flush=True)

df_no_noise = df[df['Cluster'] != -1]
cluster_means = df_no_noise.groupby('Cluster')[SYMPTOM_COLS].mean()

fig, ax = plt.subplots(figsize=(14, 8))
blue_cmap = LinearSegmentedColormap.from_list('custom_blue', ['#f7fbff', '#08306b'])
sns.heatmap(
    cluster_means.T, annot=True, fmt='.2f', cmap=blue_cmap,
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Mean Score (0=Not Mentioned, 1=Negated, 2=Present)'},
    ax=ax
)
ax.set_title("Symptom Prevalence by Cluster\n(Mean extraction score per symptom)")
ax.set_xlabel("Cluster")
ax.set_ylabel("")
ax.set_xticklabels([f"Cluster {int(x.get_text())}" for x in ax.get_xticklabels()])
plt.tight_layout()
fig.savefig(f"{OUT}/4_symptom_heatmap_by_cluster.png")
plt.close()

# =============================================================
# GRAPH 5: Cluster vs Global Deviation (What makes each cluster unique)
# =============================================================
print("5/7  Cluster Feature Deviation...", flush=True)

global_mean = df[SYMPTOM_COLS].mean()
n_clusters = len(cluster_means)

fig, axes = plt.subplots(1, n_clusters, figsize=(6 * n_clusters, 8), sharey=True)
if n_clusters == 1:
    axes = [axes]

for idx, (cl, row) in enumerate(cluster_means.iterrows()):
    ax = axes[idx]
    diff = (row - global_mean).sort_values()
    colors = ['#2980b9' if v > 0 else '#c0392b' for v in diff.values]
    
    ax.barh(diff.index, diff.values, color=colors, edgecolor='white', linewidth=0.3)
    ax.axvline(0, color='#2c3e50', linewidth=0.8)
    ax.set_title(f"Cluster {int(cl)}\n({len(df_no_noise[df_no_noise['Cluster'] == cl])} patients)")
    ax.set_xlabel("Deviation from Global Mean")
    ax.grid(axis='x', alpha=0.15)
    sns.despine(ax=ax)

plt.suptitle("What Makes Each Cluster Unique?\n(Symptom deviation from the population average)",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT}/5_cluster_deviation.png")
plt.close()

# =============================================================
# GRAPH 6: Symptom Co-Occurrence Matrix
# =============================================================
print("6/7  Symptom Co-Occurrence Matrix...", flush=True)

binary_present = (df[SYMPTOM_COLS] >= 2).astype(int)
corr = binary_present.corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
rdbu = LinearSegmentedColormap.from_list('custom_rdbu',
    ['#c0392b', '#f5b7b1', '#fdfefe', '#aed6f1', '#2471a3'])

sns.heatmap(
    corr, mask=mask, annot=True, fmt='.2f', cmap=rdbu,
    center=0, vmin=-0.5, vmax=0.5,
    linewidths=0.5, linecolor='white',
    square=True, ax=ax,
    cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8}
)
ax.set_title("Symptom Co-Occurrence Correlation\n(Based on binary presence across 269 patients)")
plt.tight_layout()
fig.savefig(f"{OUT}/6_symptom_cooccurrence.png")
plt.close()

# =============================================================
# GRAPH 7: Overall Symptom Distribution (Present vs Negated vs Not Mentioned)
# =============================================================
print("7/7  Overall Symptom Distribution...", flush=True)

present_counts = (df[SYMPTOM_COLS] == 2).sum()
negated_counts = (df[SYMPTOM_COLS] == 1).sum()
absent_counts  = (df[SYMPTOM_COLS] == 0).sum()

# Sort by total mentions (present + negated)
order = (present_counts + negated_counts).sort_values(ascending=True).index

fig, ax = plt.subplots(figsize=(12, 9))

y = np.arange(len(order))
h = 0.7

ax.barh(y, present_counts.reindex(order), height=h, label='Present (2)',
        color='#2980b9', edgecolor='white', linewidth=0.3)
ax.barh(y, negated_counts.reindex(order), height=h, left=present_counts.reindex(order),
        label='Negated (1)', color='#e74c3c', edgecolor='white', linewidth=0.3)

ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=11)
ax.set_xlabel("Count (out of 269 patients)")
ax.set_title("Overall Symptom Distribution\n(How often each symptom was detected by the LLM)")
ax.legend(frameon=True, fancybox=True, fontsize=11, loc='lower right')
ax.grid(axis='x', alpha=0.15)
sns.despine()
plt.tight_layout()
fig.savefig(f"{OUT}/7_symptom_distribution.png")
plt.close()

print(f"\nAll 7 graphs saved to '{OUT}/' folder!", flush=True)
print("Files:", flush=True)
for f in sorted(os.listdir(OUT)):
    print(f"  - {OUT}/{f}", flush=True)
