"""
Post-hoc checker-miss audit for novel symptom matching (v3).

Run this AFTER the main benchmark and initial comparisons are done.
It scans all per-sample JSONs for NOVEL cases and flags instances where
the LLM output an unmapped symptom that *partially* matches a ground-truth
hallmark keyword set but wasn't caught by the full keyword matcher.

These flags let you manually review whether the checker (fuzzy matcher)
incorrectly missed a valid LLM detection.

Usage:
    python run_checker_audit.py
    python run_checker_audit.py --results-dir path/to/results
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Force UTF-8 stdout on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import METADATA_DIR, RESULTS_DIR, SYMPTOM_LIB_DIR

# Import novel hallmarks + matching keywords from evaluate.py
sys.path.insert(0, str(SYMPTOM_LIB_DIR))
from symptom_library_v3 import NOVEL_VIRUS_HALLMARKS  # noqa: E402

from evaluate import (
    _NOVEL_MATCH_KEYWORDS,
    find_metadata,
    get_novel_symptoms_from_metadata,
    match_novel_symptoms,
)


# ─── Checker-flag logic ─────────────────────────────────────────────────────

def find_checker_flags(
    novel_in_metadata: List[str],
    unmapped_from_llm: List[Dict[str, Any]],
    match_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find cases where an unmatched LLM unmapped symptom *partially* matches
    a missed hallmark — suggesting the checker's keyword matcher may have
    failed rather than the LLM.

    A "partial match" means ≥1 keyword from any keyword set for the
    hallmark appears in the LLM symptom's combined term+definition+quote,
    but the full "all keywords in set" rule wasn't satisfied.
    """
    missed_hallmarks = match_result.get("missed", [])
    if not missed_hallmarks:
        return []

    # Indices of LLM unmapped symptoms already matched to a hallmark
    matched_indices = set()
    for m in match_result.get("matches", []):
        for i, item in enumerate(unmapped_from_llm):
            if item.get("term") == m.get("llm_term"):
                matched_indices.add(i)
                break

    flags: List[Dict[str, Any]] = []

    for hallmark in missed_hallmarks:
        keyword_sets = _NOVEL_MATCH_KEYWORDS.get(hallmark, [])
        all_keywords = set()
        for kw_set in keyword_sets:
            all_keywords.update(kw_set)

        for idx, llm_item in enumerate(unmapped_from_llm):
            if idx in matched_indices:
                continue

            llm_term = llm_item.get("term", "").lower()
            llm_definition = llm_item.get("definition", "").lower()
            llm_quote = llm_item.get("quote", "").lower()
            combined = f"{llm_term} {llm_definition} {llm_quote}"

            # Check for partial keyword match
            matching_keywords = [kw for kw in all_keywords if kw in combined]
            if matching_keywords:
                flags.append({
                    "hallmark": hallmark,
                    "llm_term": llm_item.get("term", ""),
                    "llm_definition": llm_item.get("definition", ""),
                    "llm_quote": llm_item.get("quote", ""),
                    "flag_reason": f"partial_keyword_match({', '.join(matching_keywords)})",
                })

    return flags


# ─── Main audit ──────────────────────────────────────────────────────────────

def run_audit(results_dir: Path | None = None) -> None:
    results_dir = results_dir or RESULTS_DIR

    print(f"\n{'═'*70}")
    print("  CHECKER-MISS AUDIT")
    print(f"{'═'*70}\n")

    # Find all trial/method directories
    trial_dirs = sorted(results_dir.glob("trial_*"))
    if not trial_dirs:
        print("  No trial directories found. Run the benchmark first.")
        return

    csv_columns = [
        "trial", "method", "filename", "case_type",
        "hallmark", "llm_term", "llm_definition", "llm_quote", "flag_reason",
    ]
    all_flags: List[Dict[str, Any]] = []
    total_novel_checked = 0

    for trial_dir in trial_dirs:
        trial_num = trial_dir.name.replace("trial_", "")
        method_dirs = sorted(d for d in trial_dir.iterdir() if d.is_dir())

        for method_dir in method_dirs:
            method_key = method_dir.name
            json_files = sorted(method_dir.glob("*.json"))

            for jf in json_files:
                # Only process NOVEL transcripts
                if not jf.name.startswith("NOVEL"):
                    continue

                llm_output = json.loads(jf.read_text(encoding="utf-8"))
                filename = llm_output.get("filename", jf.stem + ".txt")

                # Find metadata
                meta_path = find_metadata(filename, METADATA_DIR)
                if not meta_path:
                    continue

                # Get ground truth novel hallmarks
                novel_in_metadata = get_novel_symptoms_from_metadata(meta_path)
                if not novel_in_metadata:
                    continue

                total_novel_checked += 1
                unmapped = llm_output.get("unmapped_symptoms", [])

                # Run standard matching first
                match_result = match_novel_symptoms(novel_in_metadata, unmapped)

                # Run checker-flag detection
                flags = find_checker_flags(
                    novel_in_metadata, unmapped, match_result
                )

                for flag in flags:
                    all_flags.append({
                        "trial": trial_num,
                        "method": method_key,
                        "filename": filename,
                        "case_type": "novel_virus",
                        **flag,
                    })

    # Write checker_flags.csv
    flags_path = results_dir / "checker_flags.csv"
    with open(flags_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_flags)

    # Summary
    print(f"  NOVEL samples audited: {total_novel_checked}")
    print(f"  Checker flags found:   {len(all_flags)}")
    print(f"  Saved to: {flags_path.name}\n")

    if all_flags:
        # Group by method
        by_method: Dict[str, int] = {}
        by_hallmark: Dict[str, int] = {}
        for f in all_flags:
            by_method[f["method"]] = by_method.get(f["method"], 0) + 1
            by_hallmark[f["hallmark"]] = by_hallmark.get(f["hallmark"], 0) + 1

        print("  Flags by method:")
        for m, c in sorted(by_method.items()):
            print(f"    {m:20s} {c}")

        print("\n  Flags by hallmark:")
        for h, c in sorted(by_hallmark.items(), key=lambda x: -x[1]):
            print(f"    {h:55s} {c}")

        print(f"\n  Review checker_flags.csv to determine which flags are")
        print(f"  true checker errors vs. genuine LLM misses.")
    else:
        print("  No checker flags — all novel matches look clean.")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-hoc checker-miss audit.")
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    run_audit(args.results_dir)
