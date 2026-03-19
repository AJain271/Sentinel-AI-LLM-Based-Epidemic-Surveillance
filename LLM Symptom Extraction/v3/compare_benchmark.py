"""
Comprehensive cross-method comparison for v3 benchmark results.

Reads master_results.csv and generates:
  - Summary tables (printed + JSON)
  - 9 comparison charts saved to results/comparisons/

Can be imported (run_comparison) or run standalone.

Usage:
    python compare_benchmark.py
    python compare_benchmark.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import RESULTS_DIR

_COLORS = {"zeroshot": "#4C72B0", "fewshot": "#55A868", "regex": "#DD8452"}
_LABELS = {"zeroshot": "Zero-Shot", "fewshot": "Few-Shot", "regex": "Regex"}
_METHOD_ORDER = ["zeroshot", "fewshot", "regex"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ─── Data loading ────────────────────────────────────────────────────────────

def _load_master_csv(results_dir: Path) -> List[Dict[str, Any]]:
    path = results_dir / "master_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"master_results.csv not found in {results_dir}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Cast numeric fields
            for k in ("trial", "correct_count", "incorrect_count", "total_symptoms",
                       "prompt_tokens", "completion_tokens", "total_tokens",
                       "novel_count", "novel_matched", "novel_false_positives"):
                r[k] = int(r[k]) if r[k] else 0
            for k in ("accuracy", "extraction_time_sec",
                       "f1_negated", "f1_not_present", "f1_present"):
                r[k] = float(r[k]) if r[k] else 0.0
            r["novel_recall"] = float(r["novel_recall"]) if r["novel_recall"] else None
            rows.append(r)
    return rows


def _group_by(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    return groups


# ─── Summary statistics ─────────────────────────────────────────────────────

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _stdev(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def compute_summary(rows: List[Dict]) -> Dict[str, Any]:
    """Compute comprehensive summary statistics from master CSV rows."""
    by_method = _group_by(rows, "method")
    summary: Dict[str, Any] = {"methods": {}, "case_types": {}, "trials": {}}

    for method in _METHOD_ORDER:
        m_rows = by_method.get(method, [])
        if not m_rows:
            continue

        # Per-trial accuracy
        by_trial = _group_by(m_rows, "trial")
        trial_accs = []
        for t_rows in by_trial.values():
            trial_accs.append(_mean([r["accuracy"] for r in t_rows]))

        accs = [r["accuracy"] for r in m_rows]
        times = [r["extraction_time_sec"] for r in m_rows]
        tokens = [r["total_tokens"] for r in m_rows]

        # Novel recall (only for rows that have novel symptoms)
        novel_rows = [r for r in m_rows if r["novel_count"] > 0]
        novel_recalls = [r["novel_recall"] for r in novel_rows
                         if r["novel_recall"] is not None]
        novel_fps = [r["novel_false_positives"] for r in m_rows]

        summary["methods"][method] = {
            "label": _LABELS[method],
            "mean_accuracy": round(_mean(accs), 4),
            "stdev_accuracy": round(_stdev(accs), 4),
            "mean_trial_accuracy": round(_mean(trial_accs), 4),
            "stdev_trial_accuracy": round(_stdev(trial_accs), 4),
            "trial_accuracies": [round(a, 4) for a in trial_accs],
            "mean_time_sec": round(_mean(times), 2),
            "mean_tokens": round(_mean(tokens), 0),
            "total_tokens": sum(r["total_tokens"] for r in m_rows),
            "mean_f1_negated": round(_mean([r["f1_negated"] for r in m_rows]), 4),
            "mean_f1_not_present": round(_mean([r["f1_not_present"] for r in m_rows]), 4),
            "mean_f1_present": round(_mean([r["f1_present"] for r in m_rows]), 4),
            "novel_recall": round(_mean(novel_recalls), 4) if novel_recalls else None,
            "novel_false_positives": sum(novel_fps),
            "n_samples": len(m_rows),
        }

    # Per case-type per method
    case_types = sorted(set(r["case_type"] for r in rows))
    for ct in case_types:
        ct_rows = [r for r in rows if r["case_type"] == ct]
        ct_by_method = _group_by(ct_rows, "method")
        ct_entry: Dict[str, Any] = {}
        for method in _METHOD_ORDER:
            mm = ct_by_method.get(method, [])
            if mm:
                ct_entry[method] = {
                    "mean_accuracy": round(_mean([r["accuracy"] for r in mm]), 4),
                    "stdev_accuracy": round(_stdev([r["accuracy"] for r in mm]), 4),
                    "count": len(mm),
                }
        summary["case_types"][ct] = ct_entry

    # Per trial per method
    trials = sorted(set(r["trial"] for r in rows))
    for t in trials:
        t_rows = [r for r in rows if r["trial"] == t]
        t_by_method = _group_by(t_rows, "method")
        t_entry: Dict[str, Any] = {}
        for method in _METHOD_ORDER:
            mm = t_by_method.get(method, [])
            if mm:
                t_entry[method] = round(_mean([r["accuracy"] for r in mm]), 4)
        summary["trials"][t] = t_entry

    return summary


# ─── Print summary ───────────────────────────────────────────────────────────

def print_summary(summary: Dict[str, Any]) -> None:
    print(f"\n{'═'*80}")
    print("  COMPARATIVE SUMMARY — v3 Benchmark")
    print(f"{'═'*80}\n")

    header = (f"  {'Method':<12} {'Accuracy':>10} {'±σ':>8} "
              f"{'F1-Neg':>8} {'F1-NP':>8} {'F1-Pres':>8} "
              f"{'Novel':>8} {'FP':>5} "
              f"{'Time':>8} {'Tokens':>10}")
    print(header)
    print("  " + "─" * 92)

    for method in _METHOD_ORDER:
        m = summary["methods"].get(method)
        if not m:
            continue
        nr = f"{m['novel_recall']:.0%}" if m["novel_recall"] is not None else "N/A"
        print(
            f"  {m['label']:<12} {m['mean_accuracy']:>9.1%} "
            f"{m['stdev_accuracy']:>7.3f} "
            f"{m['mean_f1_negated']:>8.3f} {m['mean_f1_not_present']:>8.3f} "
            f"{m['mean_f1_present']:>8.3f} "
            f"{nr:>8} {m['novel_false_positives']:>5} "
            f"{m['mean_time_sec']:>7.1f}s {m['total_tokens']:>10,}"
        )

    # Per case-type
    print(f"\n  {'─'*80}")
    print(f"  ACCURACY BY CASE TYPE\n")
    ct_header = f"  {'Case Type':<30}"
    for method in _METHOD_ORDER:
        ct_header += f" {_LABELS[method]:>12}"
    print(ct_header)
    print("  " + "─" * 70)

    for ct in sorted(summary["case_types"]):
        ct_data = summary["case_types"][ct]
        line = f"  {ct:<30}"
        for method in _METHOD_ORDER:
            md = ct_data.get(method)
            if md:
                line += f" {md['mean_accuracy']:>11.1%}"
            else:
                line += f" {'---':>12}"
        print(line)

    # Per trial
    print(f"\n  {'─'*80}")
    print(f"  ACCURACY BY TRIAL\n")
    t_header = f"  {'Trial':<10}"
    for method in _METHOD_ORDER:
        t_header += f" {_LABELS[method]:>12}"
    print(t_header)
    print("  " + "─" * 50)

    for t in sorted(summary["trials"]):
        t_data = summary["trials"][t]
        line = f"  Trial {t:<3}"
        for method in _METHOD_ORDER:
            acc = t_data.get(method)
            if acc is not None:
                line += f" {acc:>11.1%}"
            else:
                line += f" {'---':>12}"
        print(line)

    print()


# ─── Charts ──────────────────────────────────────────────────────────────────

def _get_methods_present(rows: List[Dict]) -> List[str]:
    return [m for m in _METHOD_ORDER if any(r["method"] == m for r in rows)]


def plot_accuracy_by_method(rows: List[Dict], save_dir: Path) -> None:
    by_method = _group_by(rows, "method")
    methods = _get_methods_present(rows)

    # Compute per-trial means, then mean/stdev of those
    means, stds = [], []
    for m in methods:
        by_trial = _group_by(by_method[m], "trial")
        trial_means = [_mean([r["accuracy"] for r in t]) for t in by_trial.values()]
        means.append(_mean(trial_means))
        stds.append(_stdev(trial_means))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
                  color=[_COLORS[m] for m in methods],
                  edgecolor="black", linewidth=0.5)
    for bar, m_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{m_val:.1%}", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_LABELS[m] for m in methods])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy (mean ± σ across trials)")
    ax.set_title("Aggregate Accuracy by Method")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_by_method.png")
    plt.close(fig)


def plot_accuracy_by_case_type(rows: List[Dict], save_dir: Path) -> None:
    methods = _get_methods_present(rows)
    case_types = sorted(set(r["case_type"] for r in rows))

    fig, ax = plt.subplots(figsize=(max(9, 2 * len(case_types)), 5.5))
    x = np.arange(len(case_types))
    width = 0.22

    for i, m in enumerate(methods):
        m_rows = [r for r in rows if r["method"] == m]
        accs = []
        for ct in case_types:
            ct_rows = [r for r in m_rows if r["case_type"] == ct]
            accs.append(_mean([r["accuracy"] for r in ct_rows]) if ct_rows else 0)
        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=_LABELS[m],
                      color=_COLORS[m], edgecolor="black", linewidth=0.5)
        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{a:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(case_types, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Accuracy by Case Type × Method")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_by_case_type.png")
    plt.close(fig)


def plot_per_class_f1(rows: List[Dict], save_dir: Path) -> None:
    from sklearn.metrics import f1_score as _f1_score
    from symptom_checklist import KNOWN_SYMPTOM_LIST

    results_dir = save_dir.parent  # save_dir is results/comparisons
    meta_dir = (Path(__file__).resolve().parent.parent.parent
                / "Synthetic Transcript Generation" / "Synthetic Transcripts"
                / "Batch 500" / "metadata")

    methods = _get_methods_present(rows)

    # Collect y_true / y_pred per method per trial from raw JSONs
    # Structure: {method: {trial: (yt_list, yp_list)}}
    per_trial_data: Dict[str, Dict[str, tuple]] = {
        m: {} for m in methods
    }

    for trial in ["trial_1", "trial_2", "trial_3"]:
        for method in methods:
            method_dir = results_dir / trial / method
            if not method_dir.exists():
                continue
            yt_list: List[int] = []
            yp_list: List[int] = []
            for jf in sorted(method_dir.glob("*.json")):
                ext = json.loads(jf.read_text(encoding="utf-8"))
                fname = ext.get("filename", jf.stem + ".txt")
                name = fname.replace(".json", "").replace(".txt", "")
                meta_name = name.replace("_SYNTHETIC_", "_METADATA_") + ".json"
                meta_path = meta_dir / meta_name
                if not meta_path.exists():
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                present = set(meta.get("present_symptoms", []))
                negated = set(meta.get("negated_symptoms", []))
                scores = ext.get("checklist_scores", {})
                for symptom in KNOWN_SYMPTOM_LIST:
                    if symptom in present:
                        gt = 1
                    elif symptom in negated:
                        gt = -1
                    else:
                        gt = 0
                    yt_list.append(gt)
                    yp_list.append(max(-1, min(1, int(scores.get(symptom, 0)))))
            per_trial_data[method][trial] = (yt_list, yp_list)

    classes = [("Negated", -1), ("Not Present", 0), ("Present", 1)]
    categories = [c[0] for c in classes] + ["Macro F1"]
    blue_shades = ["#1565C0", "#42A5F5", "#90CAF9"]

    x = np.arange(len(categories))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, m in enumerate(methods):
        # Compute F1 per trial, then mean ± stdev
        trial_f1s = []  # list of [neg_f1, notp_f1, pres_f1, macro]
        for trial in ["trial_1", "trial_2", "trial_3"]:
            if trial not in per_trial_data[m]:
                continue
            yt, yp = per_trial_data[m][trial]
            yt_arr = np.array(yt)
            yp_arr = np.array(yp)
            f1_row = []
            for _, cls_val in classes:
                yt_bin = (yt_arr == cls_val).astype(int)
                yp_bin = (yp_arr == cls_val).astype(int)
                f1_row.append(_f1_score(yt_bin, yp_bin, zero_division=0))
            f1_row.append(sum(f1_row) / len(f1_row))  # macro
            trial_f1s.append(f1_row)

        trial_f1s = np.array(trial_f1s)  # shape (3, 4)
        means = trial_f1s.mean(axis=0)
        stds = trial_f1s.std(axis=0, ddof=1) if len(trial_f1s) > 1 else np.zeros(len(means))

        offset = (i - (len(methods) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=_LABELS[m],
                      color=blue_shades[i], edgecolor="black", linewidth=0.5,
                      error_kw={"linewidth": 1.0})
        for j, (bar, f1_val) in enumerate(zip(bars, means)):
            fmt = f"{f1_val:.3f}" if j == len(means) - 1 else f"{f1_val:.2f}"
            y_pos = bar.get_height() + stds[j] + 0.01
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Bold the Macro F1 tick label
    ax.set_xticks(x)
    tick_labels = []
    for cat in categories:
        if cat == "Macro F1":
            tick_labels.append(cat)
        else:
            tick_labels.append(cat)
    ax.set_xticklabels(tick_labels)
    for label in ax.get_xticklabels():
        if label.get_text() == "Macro F1":
            label.set_fontweight("bold")

    # Vertical separator before Macro F1
    ax.axvline(x=len(classes) - 0.5, color="#999999", linewidth=0.8, linestyle="--")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class & Macro F1 by Extraction Method (Pooled)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9,
              bbox_to_anchor=(0.98, 1.03))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "per_class_f1.png")
    plt.close(fig)


def plot_novel_detection(rows: List[Dict], save_dir: Path) -> None:
    methods = _get_methods_present(rows)
    novel_rows = [r for r in rows if r["novel_count"] > 0]

    recalls, fp_counts = [], []
    method_names = []
    for m in methods:
        m_rows = [r for r in novel_rows if r["method"] == m]
        nr = [r["novel_recall"] for r in m_rows if r["novel_recall"] is not None]
        if nr:
            recalls.append(_mean(nr))
        else:
            recalls.append(0.0)
        fp_counts.append(sum(r["novel_false_positives"] for r in
                             [r for r in rows if r["method"] == m]))
        method_names.append(_LABELS[m])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(method_names))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, recalls, width, label="Recall",
                    color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Recall")
    ax1.set_ylim(0, 1.15)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, fp_counts, width, label="False Positives",
                    color="#C44E52", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("False Positive Count")

    for bar, rec in zip(bars1, recalls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{rec:.0%}", ha="center", va="bottom", fontweight="bold")
    for bar, fp in zip(bars2, fp_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(fp), ha="center", va="bottom", fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names)
    ax1.set_title("Novel Symptom Detection")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.spines["top"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "novel_detection.png")
    plt.close(fig)


def plot_timing(rows: List[Dict], save_dir: Path) -> None:
    methods = _get_methods_present(rows)
    by_method = _group_by(rows, "method")

    means = [_mean([r["extraction_time_sec"] for r in by_method[m]]) for m in methods]
    stds = [_stdev([r["extraction_time_sec"] for r in by_method[m]]) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.5,
                  color=[_COLORS[m] for m in methods],
                  edgecolor="black", linewidth=0.5)
    for bar, m_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{m_val:.1f}s", ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_LABELS[m] for m in methods])
    ax.set_ylabel("Extraction Time (seconds)")
    ax.set_title("Mean Extraction Time by Method")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "timing_comparison.png")
    plt.close(fig)


def plot_token_usage(rows: List[Dict], save_dir: Path) -> None:
    methods = _get_methods_present(rows)
    by_method = _group_by(rows, "method")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.35

    prompt_means = [_mean([r["prompt_tokens"] for r in by_method[m]]) for m in methods]
    comp_means = [_mean([r["completion_tokens"] for r in by_method[m]]) for m in methods]

    ax.bar(x - width / 2, prompt_means, width, label="Prompt Tokens",
           color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, comp_means, width, label="Completion Tokens",
           color="#55A868", edgecolor="black", linewidth=0.5)

    for i, (p, c) in enumerate(zip(prompt_means, comp_means)):
        ax.text(i - width / 2, p + 20, f"{p:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, c + 20, f"{c:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([_LABELS[m] for m in methods])
    ax.set_ylabel("Mean Tokens per Sample")
    ax.set_title("Token Usage by Method")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "token_usage.png")
    plt.close(fig)


def plot_confusion_matrices(rows: List[Dict], save_dir: Path) -> None:
    """Build aggregate confusion matrices from per-sample correct/incorrect counts.

    This re-evaluates from the per-sample JSONs for exact confusion matrices.
    Falls back to a simplified version using accuracy data if JSONs unavailable.
    """
    methods = _get_methods_present(rows)

    # Try loading per-sample JSONs from trial 1 for confusion matrix data
    from evaluate import (
        build_ground_truth, find_metadata, KNOWN_SYMPTOM_LIST
    )
    from config import METADATA_DIR

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    label_strs = ["-1", "0", "1"]

    for i, m in enumerate(methods):
        # Aggregate y_true/y_pred across all trials for this method
        all_y_true = []
        all_y_pred = []
        m_rows = [r for r in rows if r["method"] == m]

        results_dir = save_dir.parent
        for r in m_rows:
            trial_dir = results_dir / f"trial_{r['trial']}" / m
            json_path = trial_dir / r["filename"].replace(".txt", ".json")
            if not json_path.exists():
                continue
            llm = json.loads(json_path.read_text(encoding="utf-8"))
            meta_path = find_metadata(r["filename"], METADATA_DIR)
            if not meta_path:
                continue
            truth = build_ground_truth(meta_path, KNOWN_SYMPTOM_LIST)
            scores = llm.get("checklist_scores", {})
            for sym in KNOWN_SYMPTOM_LIST:
                all_y_true.append(truth[sym])
                all_y_pred.append(max(-1, min(1, int(scores.get(sym, 0)))))

        if not all_y_true:
            continue

        from sklearn.metrics import confusion_matrix as sk_cm
        cm_arr = np.array(sk_cm(all_y_true, all_y_pred, labels=[-1, 0, 1]))

        ax = axes[i]
        im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
        for r_idx in range(cm_arr.shape[0]):
            for c_idx in range(cm_arr.shape[1]):
                val = cm_arr[r_idx, c_idx]
                color = "white" if val > cm_arr.max() * 0.6 else "black"
                ax.text(c_idx, r_idx, str(val), ha="center", va="center",
                        color=color, fontweight="bold")

        ax.set_xticks(range(3))
        ax.set_xticklabels(label_strs)
        ax.set_yticks(range(3))
        ax.set_yticklabels(label_strs)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(_LABELS[m])

    fig.suptitle("Confusion Matrices by Method (all trials)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)


def plot_per_transcript_accuracy(rows: List[Dict], save_dir: Path) -> None:
    """Heatmap: 100 transcripts (y) × 3 methods (x), color = mean accuracy across trials."""
    methods = _get_methods_present(rows)
    filenames = sorted(set(r["filename"] for r in rows))

    # Build matrix: filename × method → mean accuracy across trials
    acc_matrix = np.zeros((len(filenames), len(methods)))
    for j, m in enumerate(methods):
        m_rows = _group_by([r for r in rows if r["method"] == m], "filename")
        for i, fn in enumerate(filenames):
            fn_rows = m_rows.get(fn, [])
            acc_matrix[i, j] = _mean([r["accuracy"] for r in fn_rows]) if fn_rows else 0

    fig, ax = plt.subplots(figsize=(6, max(12, len(filenames) * 0.14)))
    im = ax.imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=0.7, vmax=1.0)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_LABELS[m] for m in methods], fontsize=10)

    # Abbreviated filenames for y-axis
    short_names = []
    for fn in filenames:
        parts = fn.replace(".txt", "").split("_")
        prefix = parts[0] if parts else fn
        patient = "_".join(parts[3:5]) if len(parts) >= 5 else fn[:15]
        short_names.append(f"{prefix} {patient}")

    ax.set_yticks(range(len(filenames)))
    ax.set_yticklabels(short_names, fontsize=5)
    ax.set_title("Per-Transcript Accuracy (mean across trials)")
    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.6)
    fig.tight_layout()
    fig.savefig(save_dir / "per_transcript_accuracy.png")
    plt.close(fig)


def plot_trial_consistency(rows: List[Dict], save_dir: Path) -> None:
    """Line plot: accuracy per trial per method."""
    methods = _get_methods_present(rows)
    trials = sorted(set(r["trial"] for r in rows))

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        by_trial = _group_by(m_rows, "trial")
        accs = [_mean([r["accuracy"] for r in by_trial.get(t, [])]) for t in trials]
        ax.plot(trials, accs, "o-", color=_COLORS[m], label=_LABELS[m],
                linewidth=2, markersize=8)
        for t, a in zip(trials, accs):
            ax.text(t, a + 0.003, f"{a:.1%}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(trials)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean Accuracy")
    ax.set_title("Trial Consistency by Method")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Set y range to show variation
    all_accs = [r["accuracy"] for r in rows]
    if all_accs:
        y_min = max(0, min(all_accs) - 0.05)
        ax.set_ylim(y_min, 1.02)

    fig.tight_layout()
    fig.savefig(save_dir / "trial_consistency.png")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────────────────

def run_comparison(results_dir: Path | None = None) -> None:
    results_dir = results_dir or RESULTS_DIR
    comp_dir = results_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_master_csv(results_dir)
    print(f"  Loaded {len(rows)} rows from master_results.csv")

    # Summary
    summary = compute_summary(rows)
    print_summary(summary)

    # Charts
    print("  Generating charts...")
    plot_accuracy_by_method(rows, comp_dir)
    print("    ✓ accuracy_by_method.png")
    plot_accuracy_by_case_type(rows, comp_dir)
    print("    ✓ accuracy_by_case_type.png")
    plot_per_class_f1(rows, comp_dir)
    print("    ✓ per_class_f1.png")
    plot_novel_detection(rows, comp_dir)
    print("    ✓ novel_detection.png")
    plot_timing(rows, comp_dir)
    print("    ✓ timing_comparison.png")
    plot_token_usage(rows, comp_dir)
    print("    ✓ token_usage.png")
    plot_confusion_matrices(rows, comp_dir)
    print("    ✓ confusion_matrices.png")
    plot_per_transcript_accuracy(rows, comp_dir)
    print("    ✓ per_transcript_accuracy.png")
    plot_trial_consistency(rows, comp_dir)
    print("    ✓ trial_consistency.png")

    # Save summary JSON
    summary_path = comp_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Summary saved to comparisons/comparison_summary.json")
    print(f"  Charts saved to comparisons/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare benchmark results.")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    run_comparison(args.results_dir)
