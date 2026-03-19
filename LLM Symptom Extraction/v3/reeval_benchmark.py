"""
Re-evaluate existing benchmark extraction JSONs with the current evaluate.py.

This avoids re-running the LLM — it reads the saved extraction JSONs,
re-runs evaluate_single (with the fixed novel matcher), and regenerates
master_results.csv and per-method CSVs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from config import METADATA_DIR, NUM_TRIALS, RESULTS_DIR
from benchmark_sample import select_benchmark_sample
from evaluate import evaluate_single, find_metadata
from symptom_checklist import KNOWN_SYMPTOM_LIST

_MASTER_COLUMNS = [
    "trial", "method", "filename", "case_type", "differential_system",
    "accuracy", "correct_count", "incorrect_count", "total_symptoms",
    "extraction_time_sec", "prompt_tokens", "completion_tokens", "total_tokens",
    "novel_count", "novel_matched", "novel_recall", "novel_false_positives",
    "f1_negated", "f1_not_present", "f1_present",
]

METHODS = ["zeroshot", "fewshot", "regex"]


def _build_row(trial, method_key, filename, case_label, ext, ev):
    if case_label.startswith("differential_"):
        case_type = "differential"
        diff_system = case_label.replace("differential_", "")
    else:
        case_type = case_label
        diff_system = ""

    cr = ev.get("classification_report", {})
    f1_neg = cr.get("negated (-1)", {}).get("f1-score", 0.0)
    f1_notp = cr.get("not_present (0)", {}).get("f1-score", 0.0)
    f1_pres = cr.get("present (1)", {}).get("f1-score", 0.0)
    ne = ev.get("novel_symptom_evaluation", {})

    return {
        "trial": trial,
        "method": method_key,
        "filename": filename,
        "case_type": case_type,
        "differential_system": diff_system,
        "accuracy": round(ev.get("accuracy", 0.0), 4),
        "correct_count": ev.get("correct_count", 0),
        "incorrect_count": ev.get("incorrect_count", 0),
        "total_symptoms": ev.get("total_symptoms", len(KNOWN_SYMPTOM_LIST)),
        "extraction_time_sec": ext.get("extraction_time_sec", 0.0),
        "prompt_tokens": ext.get("prompt_tokens", 0),
        "completion_tokens": ext.get("completion_tokens", 0),
        "total_tokens": ext.get("prompt_tokens", 0) + ext.get("completion_tokens", 0),
        "novel_count": ne.get("novel_count", 0),
        "novel_matched": ne.get("matched_count", 0),
        "novel_recall": ne.get("novel_recall", ""),
        "novel_false_positives": len(ne.get("false_positives", [])),
        "f1_negated": round(f1_neg, 4),
        "f1_not_present": round(f1_notp, 4),
        "f1_present": round(f1_pres, 4),
    }


def main():
    sample_paths, case_type_map = select_benchmark_sample()
    all_rows: List[Dict[str, Any]] = []

    for trial in range(1, NUM_TRIALS + 1):
        for method in METHODS:
            trial_dir = RESULTS_DIR / f"trial_{trial}" / method
            for tp in sample_paths:
                json_name = tp.with_suffix(".json").name
                json_path = trial_dir / json_name
                if not json_path.exists():
                    print(f"  MISSING: {json_path}")
                    continue

                ext = json.loads(json_path.read_text(encoding="utf-8"))
                meta_path = find_metadata(tp.name, METADATA_DIR)
                if meta_path is None:
                    print(f"  NO META: {tp.name}")
                    continue

                ev = evaluate_single(ext, meta_path, KNOWN_SYMPTOM_LIST)
                case_label = case_type_map.get(tp.name, "unknown")
                row = _build_row(trial, method, tp.name, case_label, ext, ev)
                all_rows.append(row)

        print(f"  Trial {trial} re-evaluated ({len(METHODS)} methods × {len(sample_paths)} samples)")

    # Write master CSV
    master = RESULTS_DIR / "master_results.csv"
    with open(master, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_MASTER_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nmaster_results.csv rewritten ({len(all_rows)} rows)")

    # Per-method CSVs
    cols = [c for c in _MASTER_COLUMNS if c != "method"]
    for m in METHODS:
        m_rows = [r for r in all_rows if r["method"] == m]
        p = RESULTS_DIR / f"{m}_results.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_MASTER_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(m_rows)
        print(f"  {m}_results.csv ({len(m_rows)} rows)")

    # Quick novel stats
    novel = [r for r in all_rows if r["novel_count"] > 0 and r["method"] != "regex"]
    for m in ["zeroshot", "fewshot"]:
        mr = [r for r in novel if r["method"] == m]
        recalls = [r["novel_recall"] for r in mr if r["novel_recall"] != ""]
        fps = sum(r["novel_false_positives"] for r in all_rows if r["method"] == m)
        perfect = sum(1 for r in recalls if float(r) == 1.0)
        print(f"\n  {m}: mean_recall={sum(float(x) for x in recalls)/len(recalls):.2%}  "
              f"perfect={perfect}/{len(recalls)}  FPs={fps}")


if __name__ == "__main__":
    main()
