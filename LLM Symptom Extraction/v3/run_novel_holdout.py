"""
Novel Holdout Runner — runs the 25 untested novel transcripts through
Zero-Shot and Few-Shot extraction (3 trials each = 150 LLM calls).

Results are stored in results/novel_holdout/ and do NOT touch the
main benchmark results.

Usage:
    python run_novel_holdout.py              # full run (3 trials)
    python run_novel_holdout.py --trials 1   # quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from config import (
    METADATA_DIR,
    NUM_TRIALS,
    RESULTS_DIR,
    TRANSCRIPT_DIR,
)
from benchmark_sample import select_benchmark_sample
from evaluate import evaluate_single, find_metadata
from symptom_checklist import KNOWN_SYMPTOM_LIST


# ─── Identify the 25 holdout novel transcripts ──────────────────────────────

def get_holdout_novel_paths() -> List[Path]:
    """Return the 25 novel transcripts NOT selected by the main benchmark."""
    # Get the 25 used in the benchmark
    benchmark_paths, _ = select_benchmark_sample()
    benchmark_names = {p.name for p in benchmark_paths}

    # Get all 50 novel transcripts from the source directory
    all_novel = sorted(TRANSCRIPT_DIR.glob("NOVEL_SYNTHETIC_*.txt"))

    # The holdout = all novel minus the ones in the benchmark
    holdout = [p for p in all_novel if p.name not in benchmark_names]

    if len(holdout) != 25:
        raise ValueError(
            f"Expected 25 holdout novel transcripts but found {len(holdout)}. "
            f"Total novel: {len(all_novel)}, in benchmark: "
            f"{sum(1 for p in all_novel if p.name in benchmark_names)}"
        )
    return holdout


# ─── Method registry (ZS + FS only) ─────────────────────────────────────────

def _load_methods():
    from extract_zeroshot import extract_single as zs_extract, _get_client as zs_client
    from extract_fewshot import extract_single as fs_extract, _get_client as fs_client

    zs_c = zs_client()
    fs_c = fs_client()

    return [
        ("zeroshot", "Zero-Shot", lambda tp: zs_extract(tp, zs_c)),
        ("fewshot", "Few-Shot", lambda tp: fs_extract(tp, fs_c)),
    ]


# ─── CSV columns ─────────────────────────────────────────────────────────────

_COLUMNS = [
    "trial", "method", "filename", "case_type",
    "accuracy", "correct_count", "incorrect_count", "total_symptoms",
    "extraction_time_sec", "prompt_tokens", "completion_tokens", "total_tokens",
    "novel_count", "novel_matched", "novel_recall", "novel_false_positives",
    "f1_negated", "f1_not_present", "f1_present",
]


# ─── Row builder ─────────────────────────────────────────────────────────────

def _build_row(
    trial: int,
    method_key: str,
    filename: str,
    extraction_result: Dict[str, Any],
    eval_result: Dict[str, Any],
) -> Dict[str, Any]:
    cr = eval_result.get("classification_report", {})
    ne = eval_result.get("novel_symptom_evaluation", {})

    return {
        "trial": trial,
        "method": method_key,
        "filename": filename,
        "case_type": "novel_virus",
        "accuracy": round(eval_result.get("accuracy", 0.0), 4),
        "correct_count": eval_result.get("correct_count", 0),
        "incorrect_count": eval_result.get("incorrect_count", 0),
        "total_symptoms": eval_result.get("total_symptoms", len(KNOWN_SYMPTOM_LIST)),
        "extraction_time_sec": extraction_result.get("extraction_time_sec", 0.0),
        "prompt_tokens": extraction_result.get("prompt_tokens", 0),
        "completion_tokens": extraction_result.get("completion_tokens", 0),
        "total_tokens": (extraction_result.get("prompt_tokens", 0)
                         + extraction_result.get("completion_tokens", 0)),
        "novel_count": ne.get("novel_count", 0),
        "novel_matched": ne.get("matched_count", 0),
        "novel_recall": ne.get("novel_recall", ""),
        "novel_false_positives": len(ne.get("false_positives", [])),
        "f1_negated": round(cr.get("negated (-1)", {}).get("f1-score", 0.0), 4),
        "f1_not_present": round(cr.get("not_present (0)", {}).get("f1-score", 0.0), 4),
        "f1_present": round(cr.get("present (1)", {}).get("f1-score", 0.0), 4),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def run_holdout(num_trials: int | None = None) -> None:
    num_trials = num_trials or NUM_TRIALS
    holdout_dir = RESULTS_DIR / "novel_holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    holdout_paths = get_holdout_novel_paths()
    n = len(holdout_paths)

    print(f"\n{'═' * 70}")
    print(f"  NOVEL HOLDOUT: {n} transcripts × {num_trials} trials × 2 methods (ZS + FS)")
    print(f"  Total extraction runs: {n * num_trials * 2}")
    print(f"{'═' * 70}\n")

    print("  Holdout transcripts:")
    for p in holdout_paths:
        short = p.name.replace("NOVEL_SYNTHETIC_v3_", "").replace(".txt", "")
        print(f"    • {short}")
    print()

    methods = _load_methods()
    all_rows: List[Dict[str, Any]] = []

    for trial in range(1, num_trials + 1):
        print(f"\n{'═' * 70}")
        print(f"  TRIAL {trial}/{num_trials}")
        print(f"{'═' * 70}")

        for method_key, method_label, extract_fn in methods:
            trial_dir = holdout_dir / f"trial_{trial}" / method_key
            trial_dir.mkdir(parents=True, exist_ok=True)

            method_start = time.time()
            total_tokens = 0

            print(f"\n  ── {method_label} ──")

            for i, tp in enumerate(holdout_paths, 1):
                out_json = trial_dir / tp.with_suffix(".json").name

                if out_json.exists():
                    # Resume: reload existing extraction
                    extraction_result = json.loads(
                        out_json.read_text(encoding="utf-8")
                    )
                    meta_path = find_metadata(tp.name, METADATA_DIR)
                    if meta_path is None:
                        print(f"  ⤳ [{method_label}] [{i:>3}/{n}] {tp.name} — SKIPPED (no meta)")
                        continue
                    eval_result = evaluate_single(
                        extraction_result, meta_path, KNOWN_SYMPTOM_LIST
                    )
                    row = _build_row(trial, method_key, tp.name,
                                     extraction_result, eval_result)
                    all_rows.append(row)
                    total_tokens += row["total_tokens"]
                    print(f"  ⤳ [{method_label}] [{i:>3}/{n}] "
                          f"{tp.name[:52]:52s} → SKIPPED (exists)")
                    continue

                # Rate limit
                if i > 1:
                    time.sleep(1)

                # Extract
                extraction_result = extract_fn(tp)

                # Save JSON
                out_json.write_text(
                    json.dumps(extraction_result, indent=2), encoding="utf-8"
                )

                # Evaluate
                meta_path = find_metadata(tp.name, METADATA_DIR)
                if meta_path is None:
                    print(f"  ✗ [{method_label}] [{i:>3}/{n}] {tp.name} — NO METADATA")
                    continue

                eval_result = evaluate_single(
                    extraction_result, meta_path, KNOWN_SYMPTOM_LIST
                )
                row = _build_row(trial, method_key, tp.name,
                                 extraction_result, eval_result)
                all_rows.append(row)
                total_tokens += row["total_tokens"]

                # Live status
                nr = row["novel_recall"]
                nr_str = f"{nr:.0%}" if nr != "" else "N/A"
                print(
                    f"  [{method_label:>9}] [{i:>3}/{n}] "
                    f"{tp.name[:52]:52s} → acc={row['accuracy']:5.1%} "
                    f"novel={row['novel_matched']}/{row['novel_count']} ({nr_str}) | "
                    f"{row['extraction_time_sec']:5.1f}s | {row['total_tokens']:>5} tok"
                )

            elapsed = time.time() - method_start
            trial_rows = [r for r in all_rows
                          if r["trial"] == trial and r["method"] == method_key]
            if trial_rows:
                mean_recall = sum(
                    r["novel_recall"] for r in trial_rows
                    if r["novel_recall"] != ""
                ) / len(trial_rows)
                print(f"\n  {method_label} summary: "
                      f"mean_novel_recall={mean_recall:.1%}  "
                      f"time={elapsed:.0f}s  tokens={total_tokens:,}")

    # ── Write CSVs ────────────────────────────────────────────────────────

    print(f"\n{'═' * 70}")
    print("  WRITING CSVs")
    print(f"{'═' * 70}\n")

    # Master holdout CSV
    master_path = holdout_dir / "holdout_results.csv"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    with open(master_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  holdout_results.csv       ({len(all_rows)} rows)")

    # Per-method CSVs
    method_cols = [c for c in _COLUMNS if c != "method"]
    for method_key, _, _ in methods:
        m_rows = [r for r in all_rows if r["method"] == method_key]
        m_path = holdout_dir / f"{method_key}_holdout.csv"
        with open(m_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=method_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(m_rows)
        print(f"  {method_key}_holdout.csv{' ' * (13 - len(method_key))}({len(m_rows)} rows)")

    print(f"\n  Results saved to: {holdout_dir}")

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'═' * 70}")
    print("  HOLDOUT COMPLETE — NOVEL RECALL SUMMARY")
    print(f"{'═' * 70}\n")

    for method_key, method_label, _ in methods:
        m_rows = [r for r in all_rows if r["method"] == method_key]
        recalls = [r["novel_recall"] for r in m_rows if r["novel_recall"] != ""]
        perfect = sum(1 for r in recalls if r == 1.0)
        fps = sum(r["novel_false_positives"] for r in m_rows)
        total_hallmarks = sum(r["novel_count"] for r in m_rows)
        total_matched = sum(r["novel_matched"] for r in m_rows)

        import numpy as np
        mean_r = np.mean(recalls) if recalls else 0
        std_r = np.std(recalls) if recalls else 0

        print(f"  {method_label}:")
        print(f"    Mean novel recall:  {mean_r:.2%}  (σ = {std_r:.4f})")
        print(f"    Perfect recall:     {perfect}/{len(recalls)} "
              f"({perfect / len(recalls):.0%})")
        print(f"    Total matched:      {total_matched}/{total_hallmarks}")
        print(f"    False positives:    {fps}")
        print()

    print("  Next step:")
    print("    python novel_holdout_analysis.py")
    print("  → Generates novel_holdout_report.pdf\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run novel holdout analysis.")
    parser.add_argument(
        "--trials", type=int, default=None,
        help=f"Number of trials (default: {NUM_TRIALS})."
    )
    args = parser.parse_args()
    run_holdout(num_trials=args.trials)
