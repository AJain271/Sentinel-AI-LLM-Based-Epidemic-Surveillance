"""
Full 500-sample few-shot symptom extraction.

Runs the proven v3 few-shot extractor on ALL 500 synthetic transcripts
from Batch 500 (single pass, temperature=0).  Saves per-patient JSONs
and a master_results.csv with the same columns as the v3 benchmark.

Usage:
    python run_full_extraction.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ─── Path setup: import v3 modules ──────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent                         # snhs symposium results
_V3_DIR = _ROOT / "LLM Symptom Extraction" / "v3"
sys.path.insert(0, str(_V3_DIR))

from config import TRANSCRIPT_DIR, METADATA_DIR  # noqa: E402
from extract_fewshot import extract_single, _get_client  # noqa: E402
from evaluate import evaluate_single, find_metadata  # noqa: E402
from symptom_checklist import KNOWN_SYMPTOM_LIST  # noqa: E402

# ─── Output paths ────────────────────────────────────────────────────────────
RESULTS_DIR = _THIS_DIR / "results"
FEWSHOT_DIR = RESULTS_DIR / "fewshot"
MASTER_CSV  = RESULTS_DIR / "master_results.csv"

# ─── Transcript index (for case_type / differential_subtype) ─────────────────
_INDEX_CSV = (
    _ROOT
    / "Synthetic Transcript Generation"
    / "Synthetic Transcripts"
    / "Batch 500"
    / "transcript_index.csv"
)


def _load_case_type_map() -> Dict[str, Dict[str, str]]:
    """Return {filename: {"case_type": ..., "differential_subtype": ...}}."""
    mapping: Dict[str, Dict[str, str]] = {}
    with open(_INDEX_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["filename"]] = {
                "case_type": row["case_type"],
                "differential_subtype": row.get("differential_subtype", ""),
            }
    return mapping


# ─── CSV column definitions ─────────────────────────────────────────────────

MASTER_COLUMNS = [
    "filename", "case_type", "differential_system",
    "accuracy", "correct_count", "incorrect_count", "total_symptoms",
    "extraction_time_sec", "prompt_tokens", "completion_tokens", "total_tokens",
    "novel_count", "novel_matched", "novel_recall", "novel_false_positives",
    "f1_negated", "f1_not_present", "f1_present",
]


def _build_row(
    filename: str,
    case_info: Dict[str, str],
    extraction_result: Dict[str, Any],
    eval_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single CSV row from extraction + evaluation results."""

    cr = eval_result.get("classification_report", {})
    f1_neg  = cr.get("negated (-1)", {}).get("f1-score", 0.0)
    f1_notp = cr.get("not_present (0)", {}).get("f1-score", 0.0)
    f1_pres = cr.get("present (1)", {}).get("f1-score", 0.0)

    ne = eval_result.get("novel_symptom_evaluation", {})

    return {
        "filename": filename,
        "case_type": case_info.get("case_type", ""),
        "differential_system": case_info.get("differential_subtype", ""),
        "accuracy": round(eval_result.get("accuracy", 0.0), 4),
        "correct_count": eval_result.get("correct_count", 0),
        "incorrect_count": eval_result.get("incorrect_count", 0),
        "total_symptoms": eval_result.get("total_symptoms", len(KNOWN_SYMPTOM_LIST)),
        "extraction_time_sec": extraction_result.get("extraction_time_sec", 0.0),
        "prompt_tokens": extraction_result.get("prompt_tokens", 0),
        "completion_tokens": extraction_result.get("completion_tokens", 0),
        "total_tokens": (
            extraction_result.get("prompt_tokens", 0)
            + extraction_result.get("completion_tokens", 0)
        ),
        "novel_count": ne.get("novel_count", 0),
        "novel_matched": ne.get("matched_count", 0),
        "novel_recall": ne.get("novel_recall", ""),
        "novel_false_positives": len(ne.get("false_positives", [])),
        "f1_negated": round(f1_neg, 4),
        "f1_not_present": round(f1_notp, 4),
        "f1_present": round(f1_pres, 4),
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def run_full_extraction() -> None:
    FEWSHOT_DIR.mkdir(parents=True, exist_ok=True)

    case_type_map = _load_case_type_map()
    transcript_paths = sorted(TRANSCRIPT_DIR.glob("*SYNTHETIC*.txt"))
    n = len(transcript_paths)

    print(f"\n{'=' * 70}")
    print(f"  FULL FEW-SHOT EXTRACTION: {n} transcripts")
    print(f"{'=' * 70}\n")

    # Show case-type breakdown
    ct_counts: Dict[str, int] = {}
    for tp in transcript_paths:
        info = case_type_map.get(tp.name, {})
        ct = info.get("case_type", "unknown")
        sub = info.get("differential_subtype", "")
        label = f"{ct}_{sub}" if sub else ct
        ct_counts[label] = ct_counts.get(label, 0) + 1
    for label in sorted(ct_counts):
        print(f"  {label:40s} {ct_counts[label]:>3}")
    print()

    client = _get_client()
    all_rows: List[Dict[str, Any]] = []
    total_tokens_all = 0
    run_start = time.time()

    for i, tp in enumerate(transcript_paths, 1):
        out_json = FEWSHOT_DIR / tp.with_suffix(".json").name
        case_info = case_type_map.get(tp.name, {"case_type": "unknown", "differential_subtype": ""})

        # ── Resume support: skip if already extracted ──
        if out_json.exists():
            extraction_result = json.loads(out_json.read_text(encoding="utf-8"))
            meta_path = find_metadata(tp.name, METADATA_DIR)
            if meta_path is None:
                print(f"  [{i:>3}/{n}] {tp.name[:55]:55s} → SKIPPED (no metadata)")
                continue
            eval_result = evaluate_single(extraction_result, meta_path, KNOWN_SYMPTOM_LIST)
            row = _build_row(tp.name, case_info, extraction_result, eval_result)
            all_rows.append(row)
            total_tokens_all += row["total_tokens"]
            print(f"  [{i:>3}/{n}] {tp.name[:55]:55s} → SKIPPED ({row['accuracy']:5.1%}, already extracted)")
            continue

        # ── Rate-limit delay ──
        if i > 1:
            time.sleep(1)

        # ── Extract ──
        extraction_result = extract_single(tp, client)

        # ── Save JSON ──
        out_json.write_text(json.dumps(extraction_result, indent=2), encoding="utf-8")

        # ── Evaluate ──
        meta_path = find_metadata(tp.name, METADATA_DIR)
        if meta_path is None:
            print(f"  [{i:>3}/{n}] {tp.name[:55]:55s} → NO METADATA")
            continue

        eval_result = evaluate_single(extraction_result, meta_path, KNOWN_SYMPTOM_LIST)
        row = _build_row(tp.name, case_info, extraction_result, eval_result)
        all_rows.append(row)
        total_tokens_all += row["total_tokens"]

        # ── Live status ──
        acc = row["accuracy"]
        t_sec = row["extraction_time_sec"]
        tok = row["total_tokens"]
        print(
            f"  [{i:>3}/{n}] {tp.name[:55]:55s} → "
            f"{acc:5.1%} | {t_sec:5.1f}s | {tok:>5} tok"
        )

    # ─── Write master CSV ─────────────────────────────────────────────────
    with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MASTER_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.time() - run_start

    # ─── Summary report ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Transcripts processed : {len(all_rows)}")
    print(f"  Total time            : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Total tokens          : {total_tokens_all:,}")

    if all_rows:
        mean_acc = sum(r["accuracy"] for r in all_rows) / len(all_rows)
        print(f"  Mean accuracy         : {mean_acc:.2%}")

        # Per-case-type summary
        ct_accs: Dict[str, List[float]] = {}
        for r in all_rows:
            ct = r["case_type"]
            ct_accs.setdefault(ct, []).append(r["accuracy"])
        print(f"\n  {'Case Type':25s} {'Count':>5}  {'Mean Acc':>8}")
        print(f"  {'─' * 42}")
        for ct in sorted(ct_accs):
            vals = ct_accs[ct]
            print(f"  {ct:25s} {len(vals):>5}  {sum(vals)/len(vals):>8.2%}")

        # Novel symptom recall summary
        novel_rows = [r for r in all_rows if r["novel_count"] > 0]
        if novel_rows:
            total_novel = sum(r["novel_count"] for r in novel_rows)
            total_matched = sum(r["novel_matched"] for r in novel_rows)
            total_fp = sum(r["novel_false_positives"] for r in novel_rows)
            print(f"\n  Novel symptom recall  : {total_matched}/{total_novel} "
                  f"({total_matched/total_novel:.1%})")
            print(f"  Novel false positives : {total_fp}")

    print(f"\n  Results saved to:")
    print(f"    JSON files : {FEWSHOT_DIR}")
    print(f"    Master CSV : {MASTER_CSV}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_full_extraction()
