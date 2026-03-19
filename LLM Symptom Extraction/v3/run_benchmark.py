"""
Benchmark runner for v3 symptom extraction.

Runs 3 trials × 3 methods (Zero-Shot, Few-Shot, Regex) on the same
100 stratified transcripts from Batch 500.  Saves per-sample JSONs,
master CSV, per-method CSVs, and prints live progress.

Usage:
    python run_benchmark.py              # full benchmark (3 trials)
    python run_benchmark.py --trials 1   # quick test (1 trial)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from config import (
    METADATA_DIR,
    NUM_TRIALS,
    RESULTS_DIR,
)
from benchmark_sample import select_benchmark_sample
from evaluate import (
    evaluate_single,
    find_metadata,
)
from symptom_checklist import KNOWN_SYMPTOM_LIST


# ─── Method registry ─────────────────────────────────────────────────────────

def _load_methods():
    """Lazy-import extractors so run_benchmark.py can be read-checked."""
    from extract_zeroshot import extract_single as zs_extract, _get_client as zs_client
    from extract_fewshot import extract_single as fs_extract, _get_client as fs_client
    from extract_regex import extract_single as rx_extract

    zs_c = zs_client()
    fs_c = fs_client()

    def zeroshot(tp):
        return zs_extract(tp, zs_c)

    def fewshot(tp):
        return fs_extract(tp, fs_c)

    def regex(tp):
        return rx_extract(tp)

    return [
        ("zeroshot", "Zero-Shot", zeroshot),
        ("fewshot", "Few-Shot", fewshot),
        ("regex", "Regex", regex),
    ]


# ─── CSV column definitions ─────────────────────────────────────────────────

_MASTER_COLUMNS = [
    "trial", "method", "filename", "case_type", "differential_system",
    "accuracy", "correct_count", "incorrect_count", "total_symptoms",
    "extraction_time_sec", "prompt_tokens", "completion_tokens", "total_tokens",
    "novel_count", "novel_matched", "novel_recall", "novel_false_positives",
    "f1_negated", "f1_not_present", "f1_present",
]

_METHOD_COLUMNS = [c for c in _MASTER_COLUMNS if c != "method"]


# ─── Row building ────────────────────────────────────────────────────────────

def _build_row(
    trial: int,
    method_key: str,
    filename: str,
    case_type_label: str,
    extraction_result: Dict[str, Any],
    eval_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single CSV row dict from extraction + evaluation results."""

    # Parse case_type / differential_system from label
    if case_type_label.startswith("differential_"):
        case_type = "differential"
        diff_system = case_type_label.replace("differential_", "")
    else:
        case_type = case_type_label
        diff_system = ""

    # F1 scores from classification report
    cr = eval_result.get("classification_report", {})
    f1_neg = cr.get("negated (-1)", {}).get("f1-score", 0.0)
    f1_notp = cr.get("not_present (0)", {}).get("f1-score", 0.0)
    f1_pres = cr.get("present (1)", {}).get("f1-score", 0.0)

    # Novel symptom metrics
    ne = eval_result.get("novel_symptom_evaluation", {})

    return {
        "trial": trial,
        "method": method_key,
        "filename": filename,
        "case_type": case_type,
        "differential_system": diff_system,
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
        "f1_negated": round(f1_neg, 4),
        "f1_not_present": round(f1_notp, 4),
        "f1_present": round(f1_pres, 4),
    }


# ─── CSV writing ─────────────────────────────────────────────────────────────

def _write_csv(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ─── Main benchmark loop ────────────────────────────────────────────────────

def run_benchmark(num_trials: int | None = None) -> None:
    num_trials = num_trials or NUM_TRIALS
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Select the 100 transcripts (deterministic)
    sample_paths, case_type_map = select_benchmark_sample()
    n = len(sample_paths)

    print(f"\n{'═'*70}")
    print(f"  BENCHMARK: {n} transcripts × {num_trials} trials × 3 methods")
    print(f"  Total extraction runs: {n * num_trials * 3}")
    print(f"{'═'*70}\n")

    # Show sample breakdown
    ct_counts: Dict[str, int] = {}
    for label in case_type_map.values():
        ct_counts[label] = ct_counts.get(label, 0) + 1
    for label in sorted(ct_counts):
        print(f"  {label:40s} {ct_counts[label]:>3}")
    print()

    methods = _load_methods()
    all_rows: List[Dict[str, Any]] = []

    for trial in range(1, num_trials + 1):
        print(f"\n{'═'*70}")
        print(f"  TRIAL {trial}/{num_trials}")
        print(f"{'═'*70}")

        for method_key, method_label, extract_fn in methods:
            trial_dir = results_dir / f"trial_{trial}" / method_key
            trial_dir.mkdir(parents=True, exist_ok=True)

            method_rows: List[Dict[str, Any]] = []
            method_start = time.time()
            total_tokens_method = 0

            print(f"\n  ── {method_label} ──")

            for i, tp in enumerate(sample_paths, 1):
                # Skip if already extracted (resume support)
                out_json = trial_dir / tp.with_suffix(".json").name
                if out_json.exists():
                    # Reload existing result
                    extraction_result = json.loads(
                        out_json.read_text(encoding="utf-8")
                    )
                    meta_path = find_metadata(tp.name, METADATA_DIR)
                    if meta_path is None:
                        print(f"  ⤳ [{method_label}] [{i:>3}/{n}] {tp.name} — SKIPPED (exists, no meta)")
                        continue
                    eval_result = evaluate_single(
                        extraction_result, meta_path, KNOWN_SYMPTOM_LIST
                    )
                    case_label = case_type_map.get(tp.name, "unknown")
                    row = _build_row(
                        trial, method_key, tp.name, case_label,
                        extraction_result, eval_result,
                    )
                    all_rows.append(row)
                    method_rows.append(row)
                    total_tokens_method += row["total_tokens"]
                    print(f"  ⤳ [{method_label}] [{i:>3}/{n}] {tp.name[:52]:52s} → SKIPPED (exists)")
                    continue

                # Small delay between LLM calls to respect rate limits
                if method_key in ("zeroshot", "fewshot") and i > 1:
                    time.sleep(1)

                # Extract (only transcript text goes to LLM)
                extraction_result = extract_fn(tp)

                # Save raw extraction JSON
                out_json = trial_dir / tp.with_suffix(".json").name
                out_json.write_text(
                    json.dumps(extraction_result, indent=2), encoding="utf-8"
                )

                # Evaluate against metadata ground truth
                meta_path = find_metadata(tp.name, METADATA_DIR)
                if meta_path is None:
                    print(f"  ✗ [{method_label}] [{i:>3}/{n}] {tp.name} — NO METADATA")
                    continue

                eval_result = evaluate_single(
                    extraction_result, meta_path, KNOWN_SYMPTOM_LIST
                )

                # Build CSV row
                case_label = case_type_map.get(tp.name, "unknown")
                row = _build_row(
                    trial, method_key, tp.name, case_label,
                    extraction_result, eval_result,
                )
                all_rows.append(row)
                method_rows.append(row)

                total_tokens_method += row["total_tokens"]

                # Live status
                acc = row["accuracy"]
                t_sec = row["extraction_time_sec"]
                tok = row["total_tokens"]
                print(
                    f"  [{method_label:>9}] [{i:>3}/{n}] "
                    f"{tp.name[:52]:52s} → {acc:5.1%} | "
                    f"{t_sec:5.1f}s | {tok:>5} tok"
                )

            # Method summary
            method_elapsed = time.time() - method_start
            if method_rows:
                mean_acc = sum(r["accuracy"] for r in method_rows) / len(method_rows)
                print(
                    f"\n  {method_label} summary: "
                    f"mean_acc={mean_acc:.1%}  "
                    f"total_time={method_elapsed:.0f}s  "
                    f"total_tokens={total_tokens_method:,}"
                )

    # ── Write CSVs ────────────────────────────────────────────────────────

    print(f"\n{'═'*70}")
    print("  WRITING CSVs")
    print(f"{'═'*70}\n")

    # Master CSV
    master_path = results_dir / "master_results.csv"
    _write_csv(master_path, _MASTER_COLUMNS, all_rows)
    print(f"  master_results.csv         ({len(all_rows)} rows)")

    # Per-method CSVs
    for method_key, _, _ in methods:
        method_rows = [r for r in all_rows if r["method"] == method_key]
        method_path = results_dir / f"{method_key}_results.csv"
        _write_csv(method_path, _METHOD_COLUMNS, method_rows)
        print(f"  {method_key}_results.csv{' ' * (13 - len(method_key))}({len(method_rows)} rows)")

    print(f"\n  Results saved to: {results_dir}")

    # ── Auto-run comparisons ─────────────────────────────────────────────

    print(f"\n{'═'*70}")
    print("  GENERATING COMPARISONS")
    print(f"{'═'*70}\n")

    try:
        from compare_benchmark import run_comparison
        run_comparison(results_dir)
    except Exception as e:
        print(f"  Warning: comparison generation failed: {e}")
        print("  You can run it manually: python compare_benchmark.py")

    print(f"\n{'═'*70}")
    print("  BENCHMARK COMPLETE")
    print(f"{'═'*70}")
    print(f"\n  Next step (optional / post-review):")
    print(f"    python run_checker_audit.py")
    print(f"  → Generates checker_flags.csv for manual novel-symptom audit\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run v3 benchmark.")
    parser.add_argument(
        "--trials", type=int, default=None,
        help=f"Number of trials (default: {NUM_TRIALS})."
    )
    args = parser.parse_args()
    run_benchmark(num_trials=args.trials)
