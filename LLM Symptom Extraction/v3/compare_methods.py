"""
Compare extraction methods: Zero-Shot vs Few-Shot vs Regex (v3).

Loads evaluation_report.json from each method's output directory,
builds side-by-side comparison tables, and generates matplotlib charts.

Usage:
    python compare_methods.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

_V3_DIR = Path(__file__).resolve().parent
_COMPARISON_DIR = _V3_DIR / "output_comparison"

# Method definitions: (label, output_dir)
METHODS = [
    ("Zero-Shot", _V3_DIR / "output"),
    ("Few-Shot",  _V3_DIR / "output_fewshot"),
    ("Regex",     _V3_DIR / "output_regex"),
]

# ─── Styling ─────────────────────────────────────────────────────────────────

_COLORS = ["#4C72B0", "#55A868", "#DD8452"]  # blue, green, orange
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


# ─── Load reports ────────────────────────────────────────────────────────────

def _load_report(output_dir: Path) -> Optional[Dict[str, Any]]:
    report_path = output_dir / "evaluation_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def load_all_reports() -> List[tuple]:
    loaded = []
    for name, out_dir in METHODS:
        report = _load_report(out_dir)
        if report is not None:
            loaded.append((name, report))
        else:
            print(f"  Warning: No evaluation_report.json found for {name} in {out_dir}")
    return loaded


# ─── Summary table ───────────────────────────────────────────────────────────

def print_summary(reports: List[tuple]) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print("COMPARATIVE SUMMARY — v3 Extraction Methods")
    print(f"{'='*80}\n")

    header = (f"{'Method':<15} {'Accuracy':>10} {'F1-Neg':>8} {'F1-NotP':>8} "
              f"{'F1-Pres':>8} {'Novel':>8} {'FP-Unm':>8}")
    print(header)
    print("-" * 80)

    summary: Dict[str, Any] = {"methods": []}

    for name, report in reports:
        agg = report.get("aggregate", {})
        acc = agg.get("accuracy", 0)
        cr = agg.get("classification_report", {})

        f1_neg = cr.get("negated (-1)", {}).get("f1-score", 0)
        f1_notp = cr.get("not_present (0)", {}).get("f1-score", 0)
        f1_pres = cr.get("present (1)", {}).get("f1-score", 0)

        novel_info = agg.get("novel_symptom_recall", {})
        novel_recall = novel_info.get("recall")
        novel_str = f"{novel_recall:.0%}" if novel_recall is not None else "N/A"
        fp_unm = novel_info.get("total_false_positives", 0)

        print(f"{name:<15} {acc:>9.1%} {f1_neg:>8.3f} {f1_notp:>8.3f} "
              f"{f1_pres:>8.3f} {novel_str:>8} {fp_unm:>8}")

        summary["methods"].append({
            "name": name,
            "accuracy": acc,
            "f1_negated": round(f1_neg, 4),
            "f1_not_present": round(f1_notp, 4),
            "f1_present": round(f1_pres, 4),
            "novel_recall": novel_recall,
            "false_positive_unmapped": fp_unm,
        })

    # Per-transcript breakdown
    print(f"\n{'='*80}")
    print("PER-TRANSCRIPT ACCURACY")
    print(f"{'='*80}\n")

    all_filenames: List[str] = []
    for _, report in reports:
        for tr in report.get("per_transcript", []):
            fn = tr.get("filename", "")
            if fn and fn not in all_filenames:
                all_filenames.append(fn)

    method_names = [name for name, _ in reports]
    header = f"{'Transcript':<55} " + " ".join(f"{n:>12}" for n in method_names)
    print(header)
    print("-" * (55 + 13 * len(method_names)))

    per_transcript_data = []
    for fn in all_filenames:
        short_name = fn[:52] + "..." if len(fn) > 55 else fn
        row = {"filename": fn, "accuracies": {}}
        vals = []
        for name, report in reports:
            tr_results = {tr["filename"]: tr for tr in report.get("per_transcript", [])}
            acc = tr_results.get(fn, {}).get("accuracy")
            if acc is not None:
                vals.append(f"{acc:>11.1%}")
                row["accuracies"][name] = acc
            else:
                vals.append(f"{'---':>12}")
        print(f"{short_name:<55} " + " ".join(vals))
        per_transcript_data.append(row)

    summary["per_transcript"] = per_transcript_data
    return summary


# ─── Charts ──────────────────────────────────────────────────────────────────

def plot_aggregate_accuracy(reports: List[tuple], save_dir: Path) -> None:
    names = [n for n, _ in reports]
    accuracies = [r.get("aggregate", {}).get("accuracy", 0) for _, r in reports]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, accuracies, color=_COLORS[:len(names)], width=0.5,
                  edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.1%}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Aggregate Accuracy by Extraction Method (v3)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    fig.savefig(save_dir / "accuracy_comparison.png")
    plt.close(fig)
    print("  Saved accuracy_comparison.png")


def plot_per_class_f1(reports: List[tuple], save_dir: Path) -> None:
    class_labels = ["Negated (-1)", "Not Present (0)", "Present (1)"]
    class_keys = ["negated (-1)", "not_present (0)", "present (1)"]

    method_names = [n for n, _ in reports]
    n_methods = len(method_names)
    x = np.arange(len(class_labels))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, report) in enumerate(reports):
        cr = report.get("aggregate", {}).get("classification_report", {})
        f1s = [cr.get(k, {}).get("f1-score", 0) for k in class_keys]
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, f1s, width, label=name,
                      color=_COLORS[i], edgecolor="black", linewidth=0.5)
        for bar, f1 in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{f1:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(class_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score by Extraction Method (v3)")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "per_class_f1.png")
    plt.close(fig)
    print("  Saved per_class_f1.png")


def plot_novel_recall(reports: List[tuple], save_dir: Path) -> None:
    names = []
    recalls = []
    fp_counts = []
    for name, report in reports:
        nr = report.get("aggregate", {}).get("novel_symptom_recall", {})
        recall = nr.get("recall")
        if recall is not None:
            names.append(name)
            recalls.append(recall)
            fp_counts.append(nr.get("total_false_positives", 0))

    if not names:
        print("  Warning: No novel symptom recall data available.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    x = np.arange(len(names))
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
    ax1.set_xticklabels(names)
    ax1.set_title("Novel Symptom Detection (v3)")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.spines["top"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "novel_detection.png")
    plt.close(fig)
    print("  Saved novel_detection.png")


def plot_confusion_matrices(reports: List[tuple], save_dir: Path) -> None:
    n = len(reports)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    label_strs = ["-1", "0", "1"]

    for i, (name, report) in enumerate(reports):
        cm = report.get("aggregate", {}).get("confusion_matrix", {}).get("matrix", [])
        if not cm:
            continue
        cm_arr = np.array(cm)
        ax = axes[i]
        im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")

        for r in range(cm_arr.shape[0]):
            for c in range(cm_arr.shape[1]):
                val = cm_arr[r, c]
                color = "white" if val > cm_arr.max() * 0.6 else "black"
                ax.text(c, r, str(val), ha="center", va="center",
                        color=color, fontweight="bold")

        ax.set_xticks(range(len(label_strs)))
        ax.set_xticklabels(label_strs)
        ax.set_yticks(range(len(label_strs)))
        ax.set_yticklabels(label_strs)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(name)

    fig.suptitle("Confusion Matrices by Method (v3)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved confusion_matrices.png")


def plot_per_transcript_accuracy(reports: List[tuple], save_dir: Path) -> None:
    all_filenames: List[str] = []
    for _, report in reports:
        for tr in report.get("per_transcript", []):
            fn = tr.get("filename", "")
            if fn and fn not in all_filenames:
                all_filenames.append(fn)

    if not all_filenames:
        return

    short_names = []
    for fn in all_filenames:
        parts = fn.replace(".txt", "").split("_")
        case_type = parts[0] if parts else fn
        patient = "_".join(parts[3:5]) if len(parts) >= 5 else fn[:20]
        short_names.append(f"{case_type}\n{patient}")

    method_names = [n for n, _ in reports]
    n_methods = len(method_names)
    x = np.arange(len(all_filenames))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(all_filenames)), 5.5))
    for i, (name, report) in enumerate(reports):
        tr_map = {tr["filename"]: tr for tr in report.get("per_transcript", [])}
        accs = [tr_map.get(fn, {}).get("accuracy", 0) for fn in all_filenames]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, accs, width, label=name,
               color=_COLORS[i], edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8, ha="center")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Transcript Accuracy by Extraction Method (v3)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    fig.savefig(save_dir / "per_transcript_accuracy.png")
    plt.close(fig)
    print("  Saved per_transcript_accuracy.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    _COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation reports...")
    reports = load_all_reports()

    if not reports:
        print("No evaluation reports found. Run the extraction pipelines first.")
        return

    summary = print_summary(reports)

    print(f"\n{'='*80}")
    print("GENERATING CHARTS")
    print(f"{'='*80}\n")

    plot_aggregate_accuracy(reports, _COMPARISON_DIR)
    plot_per_class_f1(reports, _COMPARISON_DIR)
    plot_novel_recall(reports, _COMPARISON_DIR)
    plot_per_transcript_accuracy(reports, _COMPARISON_DIR)
    plot_confusion_matrices(reports, _COMPARISON_DIR)

    # Save summary JSON
    summary_path = _COMPARISON_DIR / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Summary saved to {summary_path.name}")
    print(f"  Charts saved to {_COMPARISON_DIR}/")


if __name__ == "__main__":
    main()
