"""Compute per-symptom F1 (present class) for all three methods + novel symptom analysis + heatmap."""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symptom_checklist import KNOWN_SYMPTOM_LIST

meta_dir = Path(__file__).resolve().parent.parent.parent / "Synthetic Transcript Generation" / "Synthetic Transcripts" / "Batch 500" / "metadata"
v3_results = Path(__file__).resolve().parent / "results"
OUT_CSV = Path(__file__).resolve().parent / "results" / "per_symptom_f1.csv"
HEATMAP_PATH = Path(__file__).resolve().parent / "results" / "per_symptom_f1_heatmap.png"

def find_metadata(transcript_filename):
    name = transcript_filename.replace('.json','').replace('.txt','')
    meta_name = name.replace('_SYNTHETIC_', '_METADATA_') + '.json'
    meta_path = meta_dir / meta_name
    if meta_path.exists():
        return meta_path
    return None

def build_ground_truth(metadata_path):
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    present = set(meta.get('present_symptoms', []))
    negated = set(meta.get('negated_symptoms', []))
    truth = {}
    for symptom in KNOWN_SYMPTOM_LIST:
        if symptom in present:
            truth[symptom] = 1
        elif symptom in negated:
            truth[symptom] = -1
        else:
            truth[symptom] = 0
    return truth

# ── Collect per-symptom predictions and ground truth ──
symptom_data = {
    m: {s: {'y_true': [], 'y_pred': []} for s in KNOWN_SYMPTOM_LIST}
    for m in ['zeroshot', 'fewshot', 'regex']
}

for trial in ['trial_1', 'trial_2', 'trial_3']:
    for method in ['zeroshot', 'fewshot', 'regex']:
        method_dir = v3_results / trial / method
        if not method_dir.exists():
            continue
        for jf in sorted(method_dir.glob('*.json')):
            extraction = json.loads(jf.read_text(encoding='utf-8'))
            fname = extraction.get('filename', jf.stem + '.txt')
            meta_path = find_metadata(fname)
            if meta_path is None:
                continue
            truth = build_ground_truth(meta_path)
            scores = extraction.get('checklist_scores', {})
            for symptom in KNOWN_SYMPTOM_LIST:
                gt = truth[symptom]
                pred = scores.get(symptom, 0)
                symptom_data[method][symptom]['y_true'].append(gt)
                symptom_data[method][symptom]['y_pred'].append(pred)

# ── Compute per-symptom binary F1 for present class ──
results = []
for symptom in KNOWN_SYMPTOM_LIST:
    row = {'symptom': symptom}
    for method, label in [('zeroshot','ZS'), ('fewshot','FS'), ('regex','RX')]:
        yt = symptom_data[method][symptom]['y_true']
        yp = symptom_data[method][symptom]['y_pred']
        yt_bin = [1 if y == 1 else 0 for y in yt]
        yp_bin = [1 if y == 1 else 0 for y in yp]
        n_pos = sum(yt_bin)
        n_pred_pos = sum(yp_bin)
        if n_pos > 0:
            f1 = f1_score(yt_bin, yp_bin, zero_division=0)
            prec = precision_score(yt_bin, yp_bin, zero_division=0)
            rec = recall_score(yt_bin, yp_bin, zero_division=0)
        else:
            f1 = prec = rec = None
        row[f'{label}_f1'] = f1
        row[f'{label}_precision'] = prec
        row[f'{label}_recall'] = rec
        row[f'{label}_pred_pos'] = n_pred_pos
        if method == 'zeroshot':
            row['n_true_pos'] = n_pos
            row['n_samples'] = len(yt)
    results.append(row)

# Sort by ZS F1 ascending (None last mapped to -1)
results.sort(key=lambda r: (0 if r['n_true_pos'] == 0 else 1,
                            r['ZS_f1'] if r['ZS_f1'] is not None else -1))

# ── Write CSV ──
csv_cols = [
    'symptom', 'n_true_pos', 'n_samples',
    'ZS_f1', 'ZS_precision', 'ZS_recall', 'ZS_pred_pos',
    'FS_f1', 'FS_precision', 'FS_recall', 'FS_pred_pos',
    'RX_f1', 'RX_precision', 'RX_recall', 'RX_pred_pos',
]
with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=csv_cols)
    writer.writeheader()
    for r in results:
        out = {}
        for c in csv_cols:
            v = r.get(c)
            if v is None:
                out[c] = ''
            elif isinstance(v, float):
                out[c] = round(v, 4)
            else:
                out[c] = v
        writer.writerow(out)

print(f"Saved: {OUT_CSV}")

# ── Print table ──
header = f"{'Symptom':<60} {'ZS-F1':>7} {'ZS-P':>6} {'ZS-R':>6}   {'FS-F1':>7} {'FS-P':>6} {'FS-R':>6}   {'RX-F1':>7} {'RX-P':>6} {'RX-R':>6}   {'#Pos':>5}"
print("\n" + header)
print('-' * len(header))

for r in results:
    n = r['n_true_pos']
    s = r['symptom']
    if n == 0:
        print(f"{s:<60} {'n/a':>7} {'n/a':>6} {'n/a':>6}   {'n/a':>7} {'n/a':>6} {'n/a':>6}   {'n/a':>7} {'n/a':>6} {'n/a':>6}   {n:>5}")
    else:
        print(f"{s:<60} {r['ZS_f1']:>7.4f} {r['ZS_precision']:>6.3f} {r['ZS_recall']:>6.3f}   {r['FS_f1']:>7.4f} {r['FS_precision']:>6.3f} {r['FS_recall']:>6.3f}   {r['RX_f1']:>7.4f} {r['RX_precision']:>6.3f} {r['RX_recall']:>6.3f}   {n:>5}")

# ── Summary ──
print("\n\n=== SUMMARY (symptoms with >0 positives only) ===")
with_pos = [r for r in results if r['n_true_pos'] > 0]
for method, label in [('zeroshot','ZS'), ('fewshot','FS'), ('regex','RX')]:
    f1s = [r[f'{label}_f1'] for r in with_pos]
    avg = sum(f1s) / len(f1s)
    below_80 = sum(1 for f in f1s if f < 0.80)
    below_50 = sum(1 for f in f1s if f < 0.50)
    perfect = sum(1 for f in f1s if f == 1.0)
    print(f"  {label}: Mean F1={avg:.4f}  Perfect(1.0)={perfect}  <0.80={below_80}  <0.50={below_50}  (of {len(with_pos)} symptoms)")

# ── Novel symptom analysis from master_results.csv + holdout ──
print("\n\n=== NOVEL SYMPTOM DETECTION ===")
master_csv = v3_results / "master_results.csv"
holdout_csv = v3_results / "novel_holdout" / "holdout_results.csv"

master_rows = []
with open(master_csv, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        master_rows.append(r)

holdout_rows = []
if holdout_csv.exists():
    with open(holdout_csv, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            holdout_rows.append(r)

for method in ['zeroshot', 'fewshot', 'regex']:
    # Benchmark novel cases
    bench = [r for r in master_rows if r['method'] == method and r['case_type'] == 'novel_virus']
    # Holdout novel cases (no regex in holdout)
    hold = [r for r in holdout_rows if r.get('method', '') == method]

    print(f"\n  {method.upper()}:")

    for label, data in [("Benchmark", bench), ("Holdout", hold)]:
        if not data:
            continue
        n = len(data)
        total_novel = sum(int(r['novel_count']) for r in data)
        total_matched = sum(int(r['novel_matched']) for r in data)
        total_fps = sum(int(r['novel_false_positives']) for r in data)
        recalls = [float(r['novel_recall']) for r in data if r['novel_recall'] not in ('', 'None', None)]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0
        overall_recall = total_matched / total_novel if total_novel > 0 else 0
        print(f"    {label} (n={n}): recall={overall_recall:.4f}  matched={total_matched}/{total_novel}  FPs={total_fps}")

    # Combined
    combined = bench + hold
    if combined:
        total_novel = sum(int(r['novel_count']) for r in combined)
        total_matched = sum(int(r['novel_matched']) for r in combined)
        overall_recall = total_matched / total_novel if total_novel > 0 else 0
        print(f"    COMBINED  (n={len(combined)}): recall={overall_recall:.4f}  matched={total_matched}/{total_novel}")

# ── Heatmap (split across multiple images) ──
print("\n\nGenerating heatmaps...")

# Only symptoms with positives, sorted by avg ZS+FS F1 descending
with_pos = [r for r in results if r['n_true_pos'] > 0]
with_pos.sort(key=lambda r: (r['ZS_f1'] + r['FS_f1']) / 2, reverse=True)

method_labels = ['Zero-Shot', 'Few-Shot', 'Regex']
method_keys = ['ZS', 'FS', 'RX']

# Split into 3 pages
N_PAGES = 3
page_size = len(with_pos) // N_PAGES
remainder = len(with_pos) % N_PAGES
pages = []
start = 0
for p in range(N_PAGES):
    end = start + page_size + (1 if p < remainder else 0)
    pages.append(with_pos[start:end])
    start = end

for page_idx, page_rows in enumerate(pages, 1):
    names = [r['symptom'] for r in page_rows]
    n = len(names)
    mat = np.full((n, 3), np.nan)
    for i, r in enumerate(page_rows):
        for j, mk in enumerate(method_keys):
            v = r[f'{mk}_f1']
            if v is not None:
                mat[i, j] = v

    fig_height = max(6, n * 0.45)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color='#e0e0e0')
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(3))
    ax.set_xticklabels(method_labels, fontsize=11, fontweight='bold',
                       ha='center')
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=9)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for i in range(n):
        for j in range(3):
            val = mat[i, j]
            if np.isnan(val):
                ax.text(j, i, "n/a", ha='center', va='center', fontsize=8, color='#888888')
            else:
                color = 'white' if val > 0.65 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8,
                        fontweight='bold', color=color)

    ax.set_title(f"Per-Symptom F1 (Present Class) — Page {page_idx}/{len(pages)}",
                 fontsize=13, fontweight='bold', pad=20)
    plt.colorbar(im, ax=ax, label='F1 Score', shrink=0.5, pad=0.08)
    plt.tight_layout()
    out_path = v3_results / f"per_symptom_f1_heatmap_p{page_idx}.png"
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f"Saved heatmap: {out_path}")
    plt.close()

# ── Macro F1 bar chart (from master_results.csv, benchmark only) ──
print("\nGenerating macro F1 / per-class charts...")

# Aggregate flat y_true / y_pred across all trials for each method
all_yt = {m: [] for m in ['zeroshot', 'fewshot', 'regex']}
all_yp = {m: [] for m in ['zeroshot', 'fewshot', 'regex']}
for method in ['zeroshot', 'fewshot', 'regex']:
    for symptom in KNOWN_SYMPTOM_LIST:
        all_yt[method].extend(symptom_data[method][symptom]['y_true'])
        all_yp[method].extend(symptom_data[method][symptom]['y_pred'])

method_display = {'zeroshot': 'Zero-Shot', 'fewshot': 'Few-Shot', 'regex': 'Regex'}
methods_list = ['zeroshot', 'fewshot', 'regex']
bar_colors = ['#2196F3', '#4CAF50', '#FF9800']

# Compute macro F1 and per-class F1 + accuracy
CLASS_LABELS = {-1: 'Negated', 0: 'Not Present', 1: 'Present'}
class_order = [-1, 0, 1]

macro_f1s = []
per_class_f1 = {c: [] for c in class_order}
per_class_acc = {c: [] for c in class_order}

for method in methods_list:
    yt = np.array(all_yt[method])
    yp = np.array(all_yp[method])
    macro_f1s.append(f1_score(yt, yp, labels=class_order, average='macro', zero_division=0))

    for cls in class_order:
        yt_bin = (yt == cls).astype(int)
        yp_bin = (yp == cls).astype(int)
        per_class_f1[cls].append(f1_score(yt_bin, yp_bin, zero_division=0))
        # accuracy for this class = (TP + TN) / total
        per_class_acc[cls].append(((yt_bin == yp_bin).sum()) / len(yt_bin))

# --- Chart 1: Macro F1 bar chart ---
fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(methods_list))
bars = ax.bar(x, macro_f1s, color=bar_colors, width=0.55, edgecolor='black', linewidth=0.5)
for b, v in zip(bars, macro_f1s):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.4f}",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([method_display[m] for m in methods_list], fontsize=11)
ax.set_ylabel('Macro F1', fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title('Macro F1 by Extraction Method\n(Benchmark — 3 trials × 100 transcripts)', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
macro_path = v3_results / "macro_f1_by_method.png"
plt.savefig(macro_path, dpi=180, bbox_inches='tight')
print(f"Saved: {macro_path}")
plt.close()

# --- Chart 2: Per-class F1 grouped bar chart ---
fig, ax = plt.subplots(figsize=(7, 4.5))
n_methods = len(methods_list)
n_classes = len(class_order)
width = 0.22
x = np.arange(n_classes)
for i, method in enumerate(methods_list):
    vals = [per_class_f1[cls][i] for cls in class_order]
    offset = (i - (n_methods - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width=width, label=method_display[method],
                  color=bar_colors[i], edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{v:.3f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([CLASS_LABELS[c] for c in class_order], fontsize=11)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title('Per-Class F1 by Extraction Method', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
class_f1_path = v3_results / "per_class_f1_by_method.png"
plt.savefig(class_f1_path, dpi=180, bbox_inches='tight')
print(f"Saved: {class_f1_path}")
plt.close()

# --- Chart 3: Per-class accuracy grouped bar chart ---
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, method in enumerate(methods_list):
    vals = [per_class_acc[cls][i] for cls in class_order]
    offset = (i - (n_methods - 1) / 2) * width
    bars = ax.bar(x + offset, vals, width=width, label=method_display[method],
                  color=bar_colors[i], edgecolor='black', linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{v:.3f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([CLASS_LABELS[c] for c in class_order], fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title('Per-Class Accuracy by Extraction Method', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
class_acc_path = v3_results / "per_class_accuracy_by_method.png"
plt.savefig(class_acc_path, dpi=180, bbox_inches='tight')
print(f"Saved: {class_acc_path}")
plt.close()

# Print per-class summary
print("\n\n=== PER-CLASS METRICS (across all 70 symptoms × 300 samples) ===")
print(f"{'Method':<12} {'Macro F1':>9}  {'F1-Neg':>7} {'F1-NotP':>8} {'F1-Pres':>8}  {'Acc-Neg':>8} {'Acc-NotP':>9} {'Acc-Pres':>9}")
print('-' * 85)
for i, method in enumerate(methods_list):
    print(f"{method_display[method]:<12} {macro_f1s[i]:>9.4f}  "
          f"{per_class_f1[-1][i]:>7.4f} {per_class_f1[0][i]:>8.4f} {per_class_f1[1][i]:>8.4f}  "
          f"{per_class_acc[-1][i]:>8.4f} {per_class_acc[0][i]:>9.4f} {per_class_acc[1][i]:>9.4f}")

# ── Chart: F1 by Case Type (pooled per trial → mean ± σ) ──
print("\nGenerating F1-by-case-type chart...")

CASE_TYPE_LABELS = {
    'differential': 'Differential',
    'flu_like': 'Influenza',
    'healthy': 'Healthy',
    'novel_virus': 'Novel Virus',
}
red_shades_ct = ['#B71C1C', '#E53935', '#EF9A9A']

# Gather per-trial per-method per-case-type F1-present from raw JSONs
case_types_all = sorted(set(r.get('case_type', '') for r in master_rows if r.get('case_type', '') in CASE_TYPE_LABELS))
# {method: {case_type: {trial: [y_true, y_pred]}}}
ct_trial_data = {m: {ct: {} for ct in case_types_all} for m in methods_list}

for trial_name in ['trial_1', 'trial_2', 'trial_3']:
    for method in methods_list:
        method_dir = v3_results / trial_name / method
        if not method_dir.exists():
            continue
        for jf in sorted(method_dir.glob('*.json')):
            extraction = json.loads(jf.read_text(encoding='utf-8'))
            fname = extraction.get('filename', jf.stem + '.txt')
            # Determine case type from master_rows
            ct_match = None
            for mr in master_rows:
                if mr['filename'] == fname and mr['method'] == method and str(mr['trial']) == trial_name.split('_')[1]:
                    ct_match = mr.get('case_type', '')
                    break
            if ct_match not in CASE_TYPE_LABELS:
                continue
            meta_path = find_metadata(fname)
            if meta_path is None:
                continue
            truth = build_ground_truth(meta_path)
            scores = extraction.get('checklist_scores', {})
            if trial_name not in ct_trial_data[method][ct_match]:
                ct_trial_data[method][ct_match][trial_name] = [[], []]
            for symptom in KNOWN_SYMPTOM_LIST:
                gt = truth[symptom]
                pred = scores.get(symptom, 0)
                ct_trial_data[method][ct_match][trial_name][0].append(gt)
                ct_trial_data[method][ct_match][trial_name][1].append(pred)

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(case_types_all))
width = 0.22

for i, method in enumerate(methods_list):
    means_ct, stds_ct = [], []
    for ct in case_types_all:
        trial_f1s_ct = []
        for trial_name in ['trial_1', 'trial_2', 'trial_3']:
            td = ct_trial_data[method][ct].get(trial_name)
            if td is None or not td[0]:
                continue
            yt_arr = np.array(td[0])
            yp_arr = np.array(td[1])
            macro_cls = []
            for cls_val in [-1, 0, 1]:
                yt_b = (yt_arr == cls_val).astype(int)
                yp_b = (yp_arr == cls_val).astype(int)
                macro_cls.append(f1_score(yt_b, yp_b, zero_division=0))
            trial_f1s_ct.append(sum(macro_cls) / len(macro_cls))
        means_ct.append(np.mean(trial_f1s_ct) if trial_f1s_ct else 0)
        stds_ct.append(np.std(trial_f1s_ct, ddof=1) if len(trial_f1s_ct) > 1 else 0)
    offset = (i - (len(methods_list) - 1) / 2) * width
    bars = ax.bar(x + offset, means_ct, width, yerr=stds_ct, capsize=3,
                  label=method_display[method], color=red_shades_ct[i],
                  edgecolor='black', linewidth=0.5, error_kw={'linewidth': 1.0})
    for b, v, s in zip(bars, means_ct, stds_ct):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + s + 0.03,
                f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([CASE_TYPE_LABELS[ct] for ct in case_types_all], fontsize=13)
ax.set_ylim(0, 1.20)
ax.set_ylabel('Macro F1', fontsize=13)
ax.set_title('Macro F1 by Case Type × Extraction Method', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right', bbox_to_anchor=(0.99, 0.05))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
f1_ct_path = v3_results / "f1_by_case_type.png"
plt.savefig(f1_ct_path, dpi=180, bbox_inches='tight')
print(f"Saved: {f1_ct_path}")
plt.close()

# ── Chart: Average Novel Recall across 3 trials for 5 novel symptoms ──
print("Generating novel recall chart...")

from novel_detection_review import (
    NOVEL_HALLMARKS, HALLMARK_SHORT, HALLMARK_DESCRIPTIONS,
    match_hallmark, get_hallmarks_for_case, load_metadata,
)

hallmark_short_list = list(HALLMARK_SHORT.values())
hallmark_full_list = list(HALLMARK_SHORT.keys())

# Per trial per method per hallmark: recall
# {method: {trial: {hallmark: [hits, total]}}}
novel_trial_recall = {m: {t: {h: [0, 0] for h in hallmark_full_list}
                         for t in [1, 2, 3]} for m in ['zeroshot', 'fewshot', 'regex']}

for trial in [1, 2, 3]:
    for method in ['zeroshot', 'fewshot', 'regex']:
        method_dir = v3_results / f"trial_{trial}" / method
        if not method_dir.exists():
            continue
        for jf in sorted(method_dir.glob('NOVEL_*.json')):
            ext = json.loads(jf.read_text(encoding='utf-8'))
            fname = ext.get('filename', jf.stem + '.txt')
            meta = load_metadata(fname)
            if not meta:
                continue
            hallmarks = get_hallmarks_for_case(meta)
            unmapped = ext.get('unmapped_symptoms', [])
            for h in hallmarks:
                novel_trial_recall[method][trial][h][1] += 1
                for item in unmapped:
                    if match_hallmark(h, item):
                        novel_trial_recall[method][trial][h][0] += 1
                        break

# Also include holdout
holdout_dir = v3_results / "novel_holdout"
for trial in [1, 2, 3]:
    for method in ['zeroshot', 'fewshot']:
        method_dir = holdout_dir / f"trial_{trial}" / method
        if not method_dir.exists():
            continue
        for jf in sorted(method_dir.glob('NOVEL_*.json')):
            ext = json.loads(jf.read_text(encoding='utf-8'))
            fname = ext.get('filename', jf.stem + '.txt')
            meta = load_metadata(fname)
            if not meta:
                continue
            hallmarks = get_hallmarks_for_case(meta)
            unmapped = ext.get('unmapped_symptoms', [])
            for h in hallmarks:
                novel_trial_recall[method][trial][h][1] += 1
                for item in unmapped:
                    if match_hallmark(h, item):
                        novel_trial_recall[method][trial][h][0] += 1
                        break

# Compute mean ± 2SE across 3 trials for each method × hallmark
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(hallmark_short_list))
width = 0.25
blue_2 = ['#1565C0', '#42A5F5', '#90CAF9']
n_trials = 3

for i, method in enumerate(['zeroshot', 'fewshot', 'regex']):
    means_r, se2_r = [], []
    for h_full in hallmark_full_list:
        trial_recs = []
        for t in [1, 2, 3]:
            hits, total = novel_trial_recall[method][t][h_full]
            trial_recs.append(hits / total if total > 0 else np.nan)
        valid = [v for v in trial_recs if not np.isnan(v)]
        means_r.append(np.mean(valid) if valid else 0)
        sd = np.std(valid, ddof=1) if len(valid) > 1 else 0
        se2_r.append(2 * sd / np.sqrt(len(valid)) if len(valid) > 1 else 0)
    offset = (i - 1) * width
    bars = ax.bar(x + offset, means_r, width, yerr=se2_r, capsize=3,
                  label=method_display[method], color=blue_2[i],
                  edgecolor='black', linewidth=0.5, error_kw={'linewidth': 1.0})
    for b, v, err in zip(bars, means_r, se2_r):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + err + 0.03,
                f"{v:.0%}", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(hallmark_short_list, fontsize=10, ha='center')
ax.set_ylim(0, 1.25)
ax.set_ylabel('Recall', fontsize=12)
ax.set_title('Novel Symptom Recall by Hallmark (Mean ± 2SE Across 3 Trials)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.98, 0.98))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
novel_recall_path = v3_results / "novel_recall_by_hallmark.png"
plt.savefig(novel_recall_path, dpi=180, bbox_inches='tight')
print(f"Saved: {novel_recall_path}")
plt.close()