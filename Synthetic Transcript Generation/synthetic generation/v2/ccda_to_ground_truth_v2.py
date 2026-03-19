"""
v2 CCDA → ground truth JSON.

Design goals:
- Do not modify the existing v1 pipeline.
- Reuse the v1 CCDA parser where possible.
- Normalize outputs and attach `case_metadata` useful for scenario generation.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class CaseMetadata:
    source_file: str
    parsed_at: str
    is_covid_case: bool
    clinical_suspicion: str


def _import_v1_ccda_parser():
    """
    Import v1 `ccda_to_ground_truth.py` without changing it.
    This file lives at: ../ccda_to_ground_truth.py relative to this v2 module.
    """
    v2_dir = os.path.dirname(os.path.abspath(__file__))
    v1_dir = os.path.abspath(os.path.join(v2_dir, ".."))
    if v1_dir not in sys.path:
        sys.path.insert(0, v1_dir)

    import ccda_to_ground_truth as v1  # type: ignore

    if not hasattr(v1, "parse_ccda"):
        raise ImportError("v1 ccda_to_ground_truth.parse_ccda not found")
    return v1


def _infer_is_covid_case(gt: Dict[str, Any]) -> bool:
    """
    Conservative rule-based flag based on CCDA ground truth.
    This is metadata for scenario generation; the conversation generator should
    still avoid explicitly diagnosing COVID.
    """
    hay = []
    for c in gt.get("conditions", []) or []:
        if isinstance(c, dict):
            desc = str(c.get("description", "")).lower()
            code = str(c.get("code", "")).lower()
            hay.append(desc)
            hay.append(code)
        else:
            hay.append(str(c).lower())
    joined = " ".join(hay)
    return ("covid" in joined) or ("sars-cov" in joined) or ("sarscov" in joined) or ("cov-2" in joined)


def _infer_clinical_suspicion(gt: Dict[str, Any], is_covid_case: bool) -> str:
    """
    Minimal, extendable syndrome label to guide questioning.
    """
    if is_covid_case:
        return "viral_respiratory"
    return "general_primary_care"


def parse_ccda_to_ground_truth_v2(ccda_path: str) -> Dict[str, Any]:
    v1 = _import_v1_ccda_parser()
    gt = v1.parse_ccda(ccda_path)

    is_covid_case = _infer_is_covid_case(gt)
    clinical_suspicion = _infer_clinical_suspicion(gt, is_covid_case)

    meta = CaseMetadata(
        source_file=os.path.basename(ccda_path),
        parsed_at=datetime.now().isoformat(),
        is_covid_case=is_covid_case,
        clinical_suspicion=clinical_suspicion,
    )

    # Normalize wrapper: keep original keys but attach v2 metadata block.
    gt_v2: Dict[str, Any] = dict(gt)
    gt_v2["case_metadata"] = {
        "source_file": meta.source_file,
        "parsed_at": meta.parsed_at,
        "is_covid_case": meta.is_covid_case,
        "clinical_suspicion": meta.clinical_suspicion,
    }
    gt_v2["ground_truth_version"] = "v2"

    return gt_v2


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Parse CCDA to v2 ground truth JSON.")
    parser.add_argument("--ccda", required=True, help="Path to CCDA XML file")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    ccda_path = os.path.abspath(args.ccda)
    if not os.path.exists(ccda_path):
        raise SystemExit(f"File not found: {ccda_path}")

    gt_v2 = parse_ccda_to_ground_truth_v2(ccda_path)

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        v2_dir = os.path.dirname(os.path.abspath(__file__))
        base = os.path.basename(ccda_path).replace(".xml", "_ground_truth_v2.json")
        out_path = os.path.join(v2_dir, base)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt_v2, f, indent=2, ensure_ascii=False)

    print(out_path)


if __name__ == "__main__":
    main()

