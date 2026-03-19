"""
Evaluate LLM extraction output against METADATA ground truth (v3).

Compares the checklist_scores from the LLM against the present_symptoms
and negated_symptoms in the METADATA JSON files. Computes accuracy metrics
for the known symptom checklist and evaluates novel symptom detection via
quote-binding with fuzzy matching.

Changes from v2:
- Supports unmapped_symptoms (quote-binding: term + definition + quote + status)
- Enhanced novel matching using definition field
- False-positive tracking for unmapped symptoms
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import METADATA_DIR, OUTPUT_DIR
from symptom_checklist import KNOWN_SYMPTOM_LIST

# Import novel hallmarks for evaluation only — these are NEVER shown to the LLM
import sys
from config import SYMPTOM_LIB_DIR
sys.path.insert(0, str(SYMPTOM_LIB_DIR))
from symptom_library_v3 import NOVEL_VIRUS_HALLMARKS  # noqa: E402


# ─── Filename mapping ────────────────────────────────────────────────────────

def _transcript_to_metadata_name(transcript_filename: str) -> Optional[str]:
    """Map a transcript filename to its METADATA filename.

    E.g. FLU_SYNTHETIC_v3_Berta524_...s303.txt
      → FLU_METADATA_v3_Berta524_...s303.json
    """
    name = transcript_filename.replace(".json", "").replace(".txt", "")
    meta_name = name.replace("_SYNTHETIC_", "_METADATA_") + ".json"
    return meta_name


def find_metadata(transcript_filename: str, metadata_dir: Path) -> Optional[Path]:
    """Find the METADATA file matching a transcript filename."""
    meta_name = _transcript_to_metadata_name(transcript_filename)
    meta_path = metadata_dir / meta_name
    if meta_path.exists():
        return meta_path

    # Fallback: search for a METADATA file with matching patient name suffix
    match = re.search(r"_v3_(.+?)\.(?:txt|json)$", transcript_filename)
    if not match:
        return None
    suffix = match.group(1)
    for p in metadata_dir.glob("*_METADATA_*"):
        if suffix in p.name:
            return p
    return None


# ─── Ground truth construction ───────────────────────────────────────────────

def build_ground_truth(
    metadata_path: Path,
    known_symptoms: List[str],
) -> Dict[str, int]:
    """Build a ground truth dict from a METADATA JSON file.

    Returns a dict mapping each known symptom → -1, 0, or 1.
    """
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    present = set(meta.get("present_symptoms", []))
    negated = set(meta.get("negated_symptoms", []))

    truth: Dict[str, int] = {}
    for symptom in known_symptoms:
        if symptom in present:
            truth[symptom] = 1
        elif symptom in negated:
            truth[symptom] = -1
        else:
            truth[symptom] = 0
    return truth


def get_novel_symptoms_from_metadata(metadata_path: Path) -> List[str]:
    """Return any novel hallmark symptoms present in the metadata."""
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    present = set(meta.get("present_symptoms", []))
    novel_set = set(NOVEL_VIRUS_HALLMARKS)
    return sorted(present & novel_set)


# ─── Novel symptom fuzzy matching ────────────────────────────────────────────

# Keywords that identify each novel hallmark in LLM output names/definitions
_NOVEL_MATCH_KEYWORDS: Dict[str, List[List[str]]] = {
    "Hemoptysis (coughing up blood-tinged mucus)": [
        ["hemoptysis"], ["blood", "cough"], ["blood", "mucus"],
        ["bloody", "cough"], ["bloody", "mucus"],
    ],
    "Lymphadenopathy (tender lumps in armpits)": [
        ["lymphadenopathy"], ["lymph", "node"], ["lump", "armpit"],
        ["lumps", "armpit"], ["axillary"], ["tender", "lump"],
        ["swollen", "node"], ["swollen", "gland"],
    ],
    "Skin desquamation (peeling skin on palms/fingertips)": [
        ["desquamation"], ["peeling"], ["peel", "skin"],
        ["skin", "peel"], ["flaking", "skin"],
    ],
    "Melanonychia (nails turning dark/black)": [
        ["melanonychia"], ["nail", "dark"], ["nail", "black"],
        ["nail", "discolor"], ["dark", "nail"], ["black", "nail"],
    ],
    "Dysgeusia (metallic/distorted taste \u2014 everything tastes wrong)": [
        ["dysgeusia"], ["metallic", "taste"], ["distorted", "taste"],
        ["taste", "wrong"], ["taste", "change"], ["altered", "taste"],
    ],
}


def _kw_match_field(keyword_sets: List[List[str]], text: str) -> bool:
    """Return True if any keyword set is fully contained in *text*."""
    for kw_set in keyword_sets:
        if all(kw in text for kw in kw_set):
            return True
    return False


def match_novel_symptoms(
    novel_in_metadata: List[str],
    unmapped_from_llm: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fuzzy-match LLM unmapped symptoms against ground-truth novel hallmarks.

    Uses a priority-based multi-pass strategy to avoid matching-order bugs:
      Pass 1 — match on term only  (highest confidence)
      Pass 2 — match on term + definition
      Pass 3 — match on term + definition + quote  (broadest)

    Within each pass the best (hallmark, item) pairs are consumed first,
    preventing a quote-level keyword collision from stealing a match that
    belongs to a different hallmark at the term level.
    """
    if not novel_in_metadata:
        return {
            "novel_count": 0,
            "matched_count": 0,
            "novel_recall": None,
            "matches": [],
            "missed": [],
            "false_positives": [item.get("term", "") for item in unmapped_from_llm],
        }

    matched: List[Dict[str, str]] = []
    matched_hallmarks: set = set()
    matched_llm_indices: set = set()

    # Three passes with increasingly broad text scope
    field_scopes = [
        lambda item: item.get("term", "").lower(),                          # term only
        lambda item: f"{item.get('term', '')} {item.get('definition', '')}".lower(),  # term+def
        lambda item: f"{item.get('term', '')} {item.get('definition', '')} {item.get('quote', '')}".lower(),  # all
    ]

    for scope_fn in field_scopes:
        for hallmark in novel_in_metadata:
            if hallmark in matched_hallmarks:
                continue
            keyword_sets = _NOVEL_MATCH_KEYWORDS.get(hallmark, [])

            for idx, llm_item in enumerate(unmapped_from_llm):
                if idx in matched_llm_indices:
                    continue
                text = scope_fn(llm_item)
                if _kw_match_field(keyword_sets, text):
                    matched.append({
                        "ground_truth": hallmark,
                        "llm_term": llm_item.get("term", ""),
                        "llm_definition": llm_item.get("definition", ""),
                    })
                    matched_hallmarks.add(hallmark)
                    matched_llm_indices.add(idx)
                    break

    missed = [h for h in novel_in_metadata if h not in matched_hallmarks]

    # False positives: unmapped symptoms that didn't match any ground truth hallmark
    false_positives = [
        unmapped_from_llm[i].get("term", "")
        for i in range(len(unmapped_from_llm))
        if i not in matched_llm_indices
    ]

    recall = len(matched) / len(novel_in_metadata) if novel_in_metadata else 0.0

    return {
        "novel_count": len(novel_in_metadata),
        "matched_count": len(matched),
        "novel_recall": round(recall, 4),
        "matches": matched,
        "missed": missed,
        "false_positives": false_positives,
    }


# ─── Per-transcript evaluation ───────────────────────────────────────────────

def evaluate_single(
    llm_output: Dict[str, Any],
    metadata_path: Path,
    known_symptoms: List[str],
) -> Dict[str, Any]:
    """Compare one LLM output against its METADATA ground truth."""

    truth = build_ground_truth(metadata_path, known_symptoms)
    scores = llm_output.get("checklist_scores", {})

    y_true: List[int] = []
    y_pred: List[int] = []
    per_symptom: List[Dict[str, Any]] = []

    for symptom in known_symptoms:
        gt = truth[symptom]
        pred = scores.get(symptom, 0)
        pred = max(-1, min(1, int(pred)))
        y_true.append(gt)
        y_pred.append(pred)
        per_symptom.append({
            "symptom": symptom,
            "ground_truth": gt,
            "predicted": pred,
            "correct": gt == pred,
        })

    accuracy = accuracy_score(y_true, y_pred)

    labels = [-1, 0, 1]
    target_names = ["negated (-1)", "not_present (0)", "present (1)"]
    present_labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=[target_names[labels.index(l)] for l in present_labels],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Novel symptom review
    novel_in_metadata = get_novel_symptoms_from_metadata(metadata_path)
    unmapped_from_llm = llm_output.get("unmapped_symptoms", [])

    incorrect = [s for s in per_symptom if not s["correct"]]

    # Novel symptom fuzzy matching
    novel_matching = match_novel_symptoms(novel_in_metadata, unmapped_from_llm)

    return {
        "filename": llm_output.get("filename", ""),
        "accuracy": round(accuracy, 4),
        "total_symptoms": len(known_symptoms),
        "correct_count": sum(1 for s in per_symptom if s["correct"]),
        "incorrect_count": len(incorrect),
        "classification_report": report,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm,
        },
        "incorrect_symptoms": incorrect,
        "novel_symptom_review": {
            "novel_in_metadata": novel_in_metadata,
            "unmapped_from_llm": unmapped_from_llm,
        },
        "novel_symptom_evaluation": novel_matching,
    }


# ─── Aggregate evaluation ───────────────────────────────────────────────────

def evaluate_all(
    output_dir: Path | None = None,
    metadata_dir: Path | None = None,
) -> Dict[str, Any]:
    """Evaluate all LLM output files against metadata ground truth."""

    output_dir = output_dir or OUTPUT_DIR
    metadata_dir = metadata_dir or METADATA_DIR

    llm_files = sorted(output_dir.glob("*.json"))
    llm_files = [f for f in llm_files if "evaluation" not in f.name.lower()]

    all_y_true: List[int] = []
    all_y_pred: List[int] = []
    per_transcript: List[Dict[str, Any]] = []

    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}\n")

    for llm_file in llm_files:
        llm_output = json.loads(llm_file.read_text(encoding="utf-8"))
        filename = llm_output.get("filename", llm_file.stem + ".txt")

        meta_path = find_metadata(filename, metadata_dir)
        if meta_path is None:
            print(f"  No metadata found for {filename}, skipping.")
            continue

        result = evaluate_single(llm_output, meta_path, KNOWN_SYMPTOM_LIST)
        per_transcript.append(result)

        # Collect for aggregate
        truth = build_ground_truth(meta_path, KNOWN_SYMPTOM_LIST)
        scores = llm_output.get("checklist_scores", {})
        for symptom in KNOWN_SYMPTOM_LIST:
            all_y_true.append(truth[symptom])
            all_y_pred.append(max(-1, min(1, int(scores.get(symptom, 0)))))

        # Print per-transcript summary
        print(f"--- {filename} ---")
        print(f"  Accuracy: {result['accuracy']:.1%}  "
              f"({result['correct_count']}/{result['total_symptoms']} correct)")

        if result["incorrect_symptoms"]:
            print(f"  Mismatches ({result['incorrect_count']}):")
            for s in result["incorrect_symptoms"]:
                print(f"    - {s['symptom']}: "
                      f"truth={s['ground_truth']}, pred={s['predicted']}")

        novel_review = result["novel_symptom_review"]
        if novel_review["novel_in_metadata"]:
            print(f"  Novel symptoms in ground truth:")
            for s in novel_review["novel_in_metadata"]:
                print(f"    * {s}")
        if novel_review["unmapped_from_llm"]:
            print(f"  Unmapped symptoms from LLM:")
            for s in novel_review["unmapped_from_llm"]:
                status_str = "PRESENT" if s.get("status") == 1 else "NEGATED"
                print(f"    > {s.get('term', '?')} [{status_str}]: \"{s.get('quote', '')}\"")
                if s.get("definition"):
                    print(f"      Def: {s['definition']}")

        novel_eval = result.get("novel_symptom_evaluation", {})
        if novel_eval.get("novel_count", 0) > 0:
            recall = novel_eval.get("novel_recall", 0)
            print(f"  Novel symptom recall: {recall:.0%} "
                  f"({novel_eval['matched_count']}/{novel_eval['novel_count']})")
            if novel_eval.get("missed"):
                for m in novel_eval["missed"]:
                    print(f"    MISSED: {m}")
        if novel_eval.get("false_positives"):
            print(f"  False positive unmapped: {novel_eval['false_positives']}")
        print()

    # ─── Aggregate metrics ───────────────────────────────────────────
    if not all_y_true:
        print("No transcripts evaluated.")
        return {"per_transcript": [], "aggregate": {}}

    agg_accuracy = accuracy_score(all_y_true, all_y_pred)
    labels = [-1, 0, 1]
    target_names = ["negated (-1)", "not_present (0)", "present (1)"]
    present_labels = sorted(set(all_y_true) | set(all_y_pred))

    agg_report = classification_report(
        all_y_true, all_y_pred,
        labels=present_labels,
        target_names=[target_names[labels.index(l)] for l in present_labels],
        output_dict=True,
        zero_division=0,
    )

    agg_report_str = classification_report(
        all_y_true, all_y_pred,
        labels=present_labels,
        target_names=[target_names[labels.index(l)] for l in present_labels],
        zero_division=0,
    )

    agg_cm = confusion_matrix(all_y_true, all_y_pred, labels=labels).tolist()

    print(f"{'='*70}")
    print("AGGREGATE METRICS")
    print(f"{'='*70}")
    print(f"  Total symptom judgments: {len(all_y_true)}")
    print(f"  Overall accuracy: {agg_accuracy:.1%}\n")
    print("  Classification Report:")
    for line in agg_report_str.split("\n"):
        print(f"    {line}")
    print(f"\n  Confusion Matrix (rows=truth, cols=pred):")
    print(f"    Labels: {labels}")
    for row in agg_cm:
        print(f"    {row}")

    # ─── Aggregate novel symptom recall ───────────────────────────────
    total_novel = 0
    total_matched = 0
    total_false_positives = 0
    for tr in per_transcript:
        ne = tr.get("novel_symptom_evaluation", {})
        total_novel += ne.get("novel_count", 0)
        total_matched += ne.get("matched_count", 0)
        total_false_positives += len(ne.get("false_positives", []))

    agg_novel_recall = round(total_matched / total_novel, 4) if total_novel > 0 else None
    if total_novel > 0:
        print(f"\n  Aggregate novel symptom recall: {agg_novel_recall:.0%} "
              f"({total_matched}/{total_novel})")
        print(f"  Total false positive unmapped: {total_false_positives}")

    # ─── Save report ─────────────────────────────────────────────────
    report = {
        "per_transcript": per_transcript,
        "aggregate": {
            "total_judgments": len(all_y_true),
            "accuracy": round(agg_accuracy, 4),
            "classification_report": agg_report,
            "confusion_matrix": {
                "labels": labels,
                "matrix": agg_cm,
            },
            "novel_symptom_recall": {
                "total_novel": total_novel,
                "total_matched": total_matched,
                "recall": agg_novel_recall,
                "total_false_positives": total_false_positives,
            },
        },
    }

    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved to {report_path.name}")

    return report


if __name__ == "__main__":
    evaluate_all()
