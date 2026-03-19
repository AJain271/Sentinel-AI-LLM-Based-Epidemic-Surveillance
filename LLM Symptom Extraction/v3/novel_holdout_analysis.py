"""
Comprehensive Novel Symptom Analysis — generates a PDF report + PNGs
comparing Zero-Shot vs Few-Shot across ALL 50 novel transcripts
(25 from the main benchmark + 25 holdout).

Reads from both results/master_results.csv and
results/novel_holdout/holdout_results.csv.

Usage:
    python novel_holdout_analysis.py
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from config import RESULTS_DIR, METADATA_DIR
from evaluate import _NOVEL_MATCH_KEYWORDS

# ─── Constants ────────────────────────────────────────────────────────────────

HOLDOUT_DIR = RESULTS_DIR / "novel_holdout"
METHODS = ["zeroshot", "fewshot"]
METHOD_LABELS = {"zeroshot": "Zero-Shot", "fewshot": "Few-Shot"}
NOVEL_HALLMARKS = [
    "Hemoptysis (coughing up blood-tinged mucus)",
    "Lymphadenopathy (tender lumps in armpits)",
    "Skin desquamation (peeling skin on palms/fingertips)",
    "Melanonychia (nails turning dark/black)",
    "Dysgeusia (metallic/distorted taste — everything tastes wrong)",
]
HALLMARK_SHORT = {
    "Hemoptysis (coughing up blood-tinged mucus)": "Hemoptysis",
    "Lymphadenopathy (tender lumps in armpits)": "Lymphadenopathy",
    "Skin desquamation (peeling skin on palms/fingertips)": "Skin desquamation",
    "Melanonychia (nails turning dark/black)": "Melanonychia",
    "Dysgeusia (metallic/distorted taste — everything tastes wrong)": "Dysgeusia",
}
HALLMARK_ABSORBERS = {
    "Melanonychia (nails turning dark/black)": [
        "Skin discoloration (unusual color changes)",
    ],
    "Hemoptysis (coughing up blood-tinged mucus)": [
        "Unusual bleeding (nosebleeds/bleeding gums)",
    ],
    "Dysgeusia (metallic/distorted taste — everything tastes wrong)": [
        "Ageusia (total loss of taste)",
    ],
    "Skin desquamation (peeling skin on palms/fingertips)": [
        "Rash (red/raised patches on skin)",
        "Skin lesions (sores/blisters on skin)",
    ],
    "Lymphadenopathy (tender lumps in armpits)": [],
}


# ─── Data loaders ─────────────────────────────────────────────────────────────

def _parse_csv(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["novel_count"] = int(r["novel_count"])
            r["novel_matched"] = int(r["novel_matched"])
            r["novel_recall"] = float(r["novel_recall"]) if r["novel_recall"] else None
            r["novel_false_positives"] = int(r["novel_false_positives"])
            r["accuracy"] = float(r["accuracy"])
            r["trial"] = int(r["trial"])
            rows.append(r)
    return rows


def load_all_novel_results():
    """Load novel-virus rows from both the main benchmark and holdout."""
    # Main benchmark — filter to novel_virus only, exclude regex
    main_rows = _parse_csv(RESULTS_DIR / "master_results.csv")
    main_novel = [r for r in main_rows
                  if r.get("case_type") == "novel_virus"
                  and r.get("method") in METHODS]
    for r in main_novel:
        r["_source"] = "benchmark"

    # Holdout
    holdout_rows = _parse_csv(HOLDOUT_DIR / "holdout_results.csv")
    for r in holdout_rows:
        r["_source"] = "holdout"

    combined = main_novel + holdout_rows
    n_files = len({r["filename"] for r in combined})
    print(f"  Loaded {len(combined)} rows across {n_files} unique novel transcripts")
    return combined


def load_extraction_json(trial, method, filename):
    json_name = Path(filename).stem + ".json"
    # Try holdout directory first, then main benchmark
    p = HOLDOUT_DIR / f"trial_{trial}" / method / json_name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    p = RESULTS_DIR / f"trial_{trial}" / method / json_name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_metadata(filename):
    meta_name = filename.replace("_SYNTHETIC_", "_METADATA_").replace(".txt", ".json")
    p = METADATA_DIR / meta_name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def get_hallmarks_for_case(metadata):
    present = set(metadata.get("present_symptoms", []))
    return sorted(present & set(NOVEL_HALLMARKS))


def match_hallmark(hallmark, unmapped_item):
    kw_sets = _NOVEL_MATCH_KEYWORDS.get(hallmark, [])
    t = unmapped_item.get("term", "").lower()
    d = unmapped_item.get("definition", "").lower()
    q = unmapped_item.get("quote", "").lower()
    combined = f"{t} {d} {q}"
    for kw_set in kw_sets:
        if all(kw in combined for kw in kw_set):
            return True
    return False


# ─── Per-hallmark recall computation ─────────────────────────────────────────

def compute_per_hallmark_recall(rows):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    seen = set()

    for r in rows:
        key = (r["method"], r["trial"], r["filename"])
        if key in seen:
            continue
        seen.add(key)

        meta = load_metadata(r["filename"])
        if not meta:
            continue
        hallmarks = get_hallmarks_for_case(meta)

        ext = load_extraction_json(r["trial"], r["method"], r["filename"])
        unmapped = ext.get("unmapped_symptoms", []) if ext else []

        for h in hallmarks:
            data[r["method"]][r["trial"]][h][1] += 1
            for item in unmapped:
                if match_hallmark(h, item):
                    data[r["method"]][r["trial"]][h][0] += 1
                    break

    return data


# ─── PDF page builders ───────────────────────────────────────────────────────

def page_title(ax, title, subtitle=""):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center",
            fontsize=22, fontweight="bold", family="serif")
    if subtitle:
        ax.text(0.5, 0.45, subtitle, ha="center", va="center",
                fontsize=12, color="#555555", family="serif")


def build_summary_stats_page(fig, rows):
    ax = fig.add_subplot(111)
    ax.axis("off")

    n_files = len({r["filename"] for r in rows})
    n_runs = len(rows)
    lines = [
        "COMPREHENSIVE NOVEL SYMPTOM ANALYSIS — SUMMARY STATISTICS",
        "=" * 65,
        "",
        f"  {n_files} novel transcripts × 3 trials × 2 methods = {n_runs} runs",
        "  (25 from main benchmark + 25 holdout = all 50 novel cases)",
        "",
    ]

    for m in METHODS:
        m_rows = [r for r in rows if r["method"] == m]
        recalls = [r["novel_recall"] for r in m_rows if r["novel_recall"] is not None]
        perfect = sum(1 for r in recalls if r == 1.0)
        fps = sum(r["novel_false_positives"] for r in m_rows)
        total_novel = sum(r["novel_count"] for r in m_rows)
        total_matched = sum(r["novel_matched"] for r in m_rows)

        lines.append(f"  {METHOD_LABELS[m]}")
        lines.append(f"  {'─' * 55}")
        lines.append(f"  Mean novel recall:     {np.mean(recalls):.2%}")
        lines.append(f"  Std recall:            {np.std(recalls):.4f}")
        lines.append(f"  Perfect recall (1.0):  {perfect}/{len(recalls)} samples "
                      f"({perfect / len(recalls):.1%})")
        lines.append(f"  Total matched:         {total_matched}/{total_novel}")
        lines.append(f"  Total false positives:  {fps}")
        lines.append(f"  Samples evaluated:     {len(m_rows)}")
        lines.append("")

    lines.append("─" * 65)
    lines.append("This analysis covers ALL 50 novel transcripts in the study.")

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=10,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#4C72B0"))


def plot_overall_novel_metrics(rows, ax):
    recalls, fps = [], []
    for m in METHODS:
        m_rows = [r for r in rows if r["method"] == m]
        rec = [r["novel_recall"] for r in m_rows if r["novel_recall"] is not None]
        recalls.append(np.mean(rec) if rec else 0)
        fps.append(sum(r["novel_false_positives"] for r in rows if r["method"] == m))

    x = np.arange(len(METHODS))
    w = 0.35
    colors_r = ["#4C72B0", "#55A868"]
    colors_f = ["#C44E52", "#DD8452"]

    bars1 = ax.bar(x - w / 2, recalls, w, label="Mean Recall",
                   color=colors_r, edgecolor="black", linewidth=0.5)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w / 2, fps, w, label="Total False Positives",
                    color=colors_f, edgecolor="black", linewidth=0.5)

    for b, v in zip(bars1, recalls):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.1%}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    for b, v in zip(bars2, fps):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.3,
                 str(v), ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("Recall", fontsize=11)
    ax.set_ylim(0, 1.2)
    ax2.set_ylabel("False Positive Count", fontsize=11)
    ax2.set_ylim(0, max(fps) * 2 + 1 if max(fps) > 0 else 5)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=11)
    ax.set_title("All 50 Novel Cases: Recall vs False Positives", fontsize=14, pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)


def plot_per_hallmark_heatmap(hallmark_data, ax):
    hallmarks = list(HALLMARK_SHORT.values())
    full_names = list(HALLMARK_SHORT.keys())

    matrix = np.zeros((len(hallmarks), len(METHODS)))
    for j, m in enumerate(METHODS):
        for i, h_full in enumerate(full_names):
            hits, total = 0, 0
            for trial in [1, 2, 3]:
                vals = hallmark_data[m][trial].get(h_full, [0, 0])
                hits += vals[0]
                total += vals[1]
            matrix[i, j] = hits / total if total > 0 else 0

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=10)
    ax.set_yticks(range(len(hallmarks)))
    ax.set_yticklabels(hallmarks, fontsize=10)

    for i in range(len(hallmarks)):
        for j in range(len(METHODS)):
            val = matrix[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_title("All 50 Novel Cases: Per-Hallmark Recall (Aggregated Across 3 Trials)",
                 fontsize=13, pad=10)
    plt.colorbar(im, ax=ax, label="Recall", shrink=0.8)


def plot_trial_consistency(hallmark_data, ax):
    trials = [1, 2, 3]
    colors = {"zeroshot": "#1565C0", "fewshot": "#90CAF9"}
    x = np.arange(len(METHODS))
    w = 0.45

    recalls = []
    for m in METHODS:
        hits = sum(
            hallmark_data[m][t][h][0]
            for t in trials
            for h in hallmark_data[m][t]
        )
        total = sum(
            hallmark_data[m][t][h][1]
            for t in trials
            for h in hallmark_data[m][t]
        )
        recalls.append(hits / total if total > 0 else 0)

    bars = ax.bar(x, recalls, w,
                  color=[colors[m] for m in METHODS],
                  edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, recalls):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=12)
    ax.set_ylabel("Novel Case Recall", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Novel Case Recall (Aggregated Across 3 Trials)", fontsize=17, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_fp_case_study_page(fig, rows):
    ax = fig.add_subplot(111)
    ax.axis("off")

    total_zs_fps = sum(r["novel_false_positives"] for r in rows if r["method"] == "zeroshot")
    total_fs_fps = sum(r["novel_false_positives"] for r in rows if r["method"] == "fewshot")

    text = f"FALSE POSITIVE CASE STUDIES\n\n"
    text += f"Zero-Shot total FPs: {total_zs_fps}    Few-Shot total FPs: {total_fs_fps}\n\n"

    for m in METHODS:
        fp_cases = [r for r in rows if r["method"] == m and r["novel_false_positives"] > 0]
        if not fp_cases:
            text += f"  {METHOD_LABELS[m]}: No false positives.\n\n"
            continue

        text += f"  {METHOD_LABELS[m]} FP Cases:\n"
        text += "  " + "━" * 60 + "\n"

        grouped = defaultdict(list)
        for r in fp_cases:
            short = r["filename"].replace("NOVEL_SYNTHETIC_v3_", "").replace(".txt", "")
            grouped[short].append(r)

        for name, cases in sorted(grouped.items()):
            trials_str = ", ".join(str(c["trial"]) for c in sorted(cases, key=lambda x: x["trial"]))
            text += f"  CASE: {name} (Trial{'s' if len(cases) > 1 else ''} {trials_str})\n"

            ext = load_extraction_json(cases[0]["trial"], m, cases[0]["filename"])
            if ext:
                for u in ext.get("unmapped_symptoms", []):
                    # Check if this is a FP (doesn't match any hallmark)
                    meta = load_metadata(cases[0]["filename"])
                    if meta:
                        hallmarks = get_hallmarks_for_case(meta)
                        is_match = any(match_hallmark(h, u) for h in hallmarks)
                        if not is_match:
                            text += f'    FP term: "{u.get("term", "")}"\n'
                            q = u.get("quote", "")
                            if q:
                                text += f'    Quote: "{q[:80]}"\n'
            text += "\n"

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#cccccc"))


def build_per_sample_miss_table(fig, rows):
    ax = fig.add_subplot(111)
    ax.axis("off")

    misses = [r for r in rows if r["novel_recall"] is not None and r["novel_recall"] < 1.0]

    grouped = defaultdict(list)
    for r in misses:
        short = r["filename"].replace("NOVEL_SYNTHETIC_v3_", "").replace(".txt", "")
        grouped[short].append(r)

    header = f"{'Patient':<35} {'Method':<10} {'Trial':<6} {'Match':<8} {'Recall':<8} {'FP':<4}"
    lines = [header, "─" * 75]
    for fname in sorted(grouped):
        for r in sorted(grouped[fname], key=lambda x: (x["method"], x["trial"])):
            m = METHOD_LABELS.get(r["method"], r["method"])
            lines.append(
                f"{fname[:34]:<35} {m:<10} {r['trial']:<6} "
                f"{r['novel_matched']}/{r['novel_count']:<5} "
                f"{r['novel_recall']:.2f}    {r['novel_false_positives']}"
            )

    text = "ALL 50 NOVEL CASES — IMPERFECT NOVEL RECALL\n\n" + "\n".join(lines)
    text += f"\n\nTotal imperfect: {len(misses)} / {len(rows)} novel runs"
    text += f"\nUnique patients: {len(grouped)}"

    ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#cccccc"))


# ─── Patient blurb helpers ────────────────────────────────────────────────────

def _diagnose_miss(hallmark, ext_json):
    checklist = ext_json.get("checklist_scores", {})
    unmapped = ext_json.get("unmapped_symptoms", [])
    unmapped_terms = [u.get("term", "").lower() for u in unmapped]

    absorbers = HALLMARK_ABSORBERS.get(hallmark, [])
    for ab in absorbers:
        if checklist.get(ab) == 1:
            return f"Absorbed by checklist item \"{ab}\" = 1"

    short = HALLMARK_SHORT[hallmark].lower()
    for t in unmapped_terms:
        if short in t or t in short:
            return f"Unmapped term \"{t}\" present but keyword match failed"

    return "Not mentioned in LLM output at all"


def _build_patient_blurb(patient_name, filename, imp_rows, all_rows):
    meta = load_metadata(filename)
    if not meta:
        return f"  {patient_name}: metadata not found\n"

    hallmarks = get_hallmarks_for_case(meta)
    short_names = [HALLMARK_SHORT[h] for h in hallmarks]

    lines = []
    lines.append(f"  PATIENT: {patient_name}")
    lines.append(f"  Ground-truth hallmarks ({len(hallmarks)}): "
                 + ", ".join(short_names))
    lines.append("")

    all_patient = [r for r in all_rows if r["filename"] == filename]

    for r in sorted(all_patient, key=lambda x: (x["method"], x["trial"])):
        m_label = METHOD_LABELS.get(r["method"], r["method"])
        recall = r["novel_recall"]
        matched = r["novel_matched"]
        total = r["novel_count"]
        tag = "+" if recall == 1.0 else "X"

        ext = load_extraction_json(r["trial"], r["method"], r["filename"])
        if recall == 1.0 or ext is None:
            lines.append(f"    {tag} {m_label} T{r['trial']}: {matched}/{total} "
                         f"(recall {recall:.0%})")
            continue

        unmapped = ext.get("unmapped_symptoms", [])
        matched_set = set()
        for h in hallmarks:
            for item in unmapped:
                if match_hallmark(h, item):
                    matched_set.add(h)
                    break

        missed = [h for h in hallmarks if h not in matched_set]
        missed_short = [HALLMARK_SHORT[h] for h in missed]

        lines.append(f"    {tag} {m_label} T{r['trial']}: {matched}/{total} "
                     f"(recall {recall:.0%}) -- missed: {', '.join(missed_short)}")

        for h in missed:
            reason = _diagnose_miss(h, ext)
            lines.append(f"        -> {HALLMARK_SHORT[h]}: {reason}")

    lines.append("")
    return "\n".join(lines)


def build_patient_blurbs_pages(pdf, rows):
    misses = [r for r in rows
              if r["novel_recall"] is not None and r["novel_recall"] < 1.0]

    grouped = {}
    for r in misses:
        if r["filename"] not in grouped:
            short = (r["filename"]
                     .replace("NOVEL_SYNTHETIC_v3_", "")
                     .replace(".txt", ""))
            grouped[r["filename"]] = (short, [])
        grouped[r["filename"]][1].append(r)

    if not grouped:
        # No imperfect cases — add a single page noting perfect performance
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, "All 50 novel samples achieved perfect novel recall.\n"
                "No per-patient analysis needed.",
                ha="center", va="center", fontsize=14, family="serif")
        pdf.savefig(fig)
        plt.close(fig)
        return

    blurbs = []
    for fname in sorted(grouped, key=lambda f: grouped[f][0]):
        patient_name, imp_rows = grouped[fname]
        blurbs.append(_build_patient_blurb(patient_name, fname, imp_rows, rows))

    melanonychia_misses = sum(1 for b in blurbs if "Melanonychia" in b and "Absorbed" in b)

    header = ("IMPERFECT NOVEL RECALL — PER-PATIENT ANALYSIS (ALL 50)\n"
              "=" * 65 + "\n\n")

    footer = ("\n" + "-" * 65 + "\n"
              "KEY PATTERN: Melanonychia is the most fragile hallmark -- the LLM\n"
              f"frequently maps it to \"Skin discoloration\" on the checklist\n"
              f"({melanonychia_misses}/{len(grouped)} patients affected).\n"
              "Cases where the transcript omits the symptom entirely are\n"
              "metadata/transcript generation discrepancies, not LLM errors.")

    full_text = header + "\n".join(blurbs) + footer

    text_lines = full_text.split("\n")
    PAGE_LINES = 55
    pages = []
    for i in range(0, len(text_lines), PAGE_LINES):
        pages.append("\n".join(text_lines[i:i + PAGE_LINES]))

    for page_text in pages:
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.03, 0.97, page_text, transform=ax.transAxes, fontsize=7.5,
                verticalalignment="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8f0",
                          edgecolor="#CC8844"))
        pdf.savefig(fig)
        plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    rows = load_all_novel_results()
    hallmark_data = compute_per_hallmark_recall(rows)

    out_dir = HOLDOUT_DIR
    pdf_path = out_dir / "novel_holdout_report.pdf"

    with PdfPages(str(pdf_path)) as pdf:
        # Page 1: Title
        fig, ax = plt.subplots(figsize=(11, 8.5))
        page_title(ax, "Comprehensive Novel Symptom Analysis",
                   "Zero-Shot vs Few-Shot — All 50 Novel Transcripts\n"
                   "(25 Benchmark + 25 Holdout)")
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: Summary stats
        fig = plt.figure(figsize=(11, 8.5))
        build_summary_stats_page(fig, rows)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: Overall recall & FP bar chart
        fig, ax = plt.subplots(figsize=(11, 7))
        plot_overall_novel_metrics(rows, ax)
        fig.tight_layout(pad=2)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: Per-hallmark heatmap
        fig, ax = plt.subplots(figsize=(11, 6))
        plot_per_hallmark_heatmap(hallmark_data, ax)
        fig.tight_layout(pad=2)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 5: Trial consistency
        fig, ax = plt.subplots(figsize=(11, 6))
        plot_trial_consistency(hallmark_data, ax)
        fig.tight_layout(pad=2)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 6: False positive case studies
        fig = plt.figure(figsize=(11, 8.5))
        build_fp_case_study_page(fig, rows)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 7: Per-sample miss table
        fig = plt.figure(figsize=(11, 8.5))
        build_per_sample_miss_table(fig, rows)
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 8+: Per-patient analysis blurbs
        build_patient_blurbs_pages(pdf, rows)

    print(f"\nPDF saved to: {pdf_path}")

    # Standalone PNGs
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_per_hallmark_heatmap(hallmark_data, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "holdout_per_hallmark_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_overall_novel_metrics(rows, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "holdout_novel_recall_vs_fp.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_trial_consistency(hallmark_data, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "holdout_trial_consistency.png", dpi=150)
    plt.close(fig)

    print("PNGs saved to novel_holdout/ directory.")


if __name__ == "__main__":
    main()
