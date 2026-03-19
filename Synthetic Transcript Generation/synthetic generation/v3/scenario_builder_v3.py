"""
v3 Scenario Builder — sample a clinically coherent symptom set and merge
with CCDA ground-truth data to produce a self-contained scenario dict.

Usage (standalone dry-run):
    python scenario_builder_v3.py --ground_truth path/to/ground_truth.json [--seed 42]

The scenario dict is also consumed programmatically by generate_transcript_v3.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

# Allow imports when running from the v3/ directory or from parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from symptom_library_v3 import CASE_TYPES, sample_symptoms  # noqa: E402
from sc_hospital_hubs import assign_location  # noqa: E402

# ─── CCDA noise caps ────────────────────────────────────────────────────────
MAX_CONDITIONS = 5
MAX_MEDICATIONS = 5


def _load_ground_truth(json_path: str) -> Dict[str, Any]:
    """Load a v1/ ground-truth JSON file (produced by ccda_to_ground_truth.py)."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_demographics(gt: Dict[str, Any]) -> Dict[str, Any]:
    demo = gt.get("demographics", {})
    return {
        "name": demo.get("name", "Unknown"),
        "age": demo.get("age", ""),
        "gender": demo.get("gender", ""),
        "race": demo.get("race", ""),
        "ethnicity": demo.get("ethnicity", ""),
        "city": demo.get("city", ""),
        "state": demo.get("state", ""),
    }


def _extract_ccda_noise(gt: Dict[str, Any]) -> Dict[str, Any]:
    """Pull conditions, medications, allergies, social history, immunizations
    from the ground-truth JSON.  Caps conditions and medications to keep the
    background noise subtle."""

    # Conditions — deduplicate, cap
    seen_cond: set = set()
    conditions: List[str] = []
    for c in gt.get("conditions", []):
        desc = c["description"] if isinstance(c, dict) else str(c)
        if desc not in seen_cond:
            seen_cond.add(desc)
            conditions.append(desc)
    conditions = conditions[:MAX_CONDITIONS]

    # Medications — deduplicate, cap
    seen_med: set = set()
    medications: List[str] = []
    for m in gt.get("medications", []):
        desc = m["description"] if isinstance(m, dict) else str(m)
        if desc not in seen_med:
            seen_med.add(desc)
            medications.append(desc)
    medications = medications[:MAX_MEDICATIONS]

    # Allergies (usually short, no cap needed)
    allergies = [
        a["description"] if isinstance(a, dict) else str(a)
        for a in gt.get("allergies", [])
    ]

    # Social history
    social_history: List[str] = []
    for s in gt.get("social_history", []):
        desc = s.get("description", "") if isinstance(s, dict) else str(s)
        val = s.get("value", "") if isinstance(s, dict) else ""
        entry = f"{desc}: {val}".strip(": ") if val else desc
        if entry:
            social_history.append(entry)

    # Immunizations (brief mention)
    immunizations = [
        i["description"] if isinstance(i, dict) else str(i)
        for i in gt.get("immunizations", [])
    ][:5]

    # Injected symptoms (from inject_symptoms.py)
    injected: List[str] = []
    for s in gt.get("injected_symptoms", []):
        if isinstance(s, dict) and s.get("description"):
            injected.append(s["description"])
        elif isinstance(s, str):
            injected.append(s)

    return {
        "conditions": conditions,
        "medications": medications,
        "allergies": allergies,
        "social_history": social_history,
        "immunizations": immunizations,
        "injected_symptoms": injected,
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def build_scenario(
    ground_truth_path: str,
    seed: Optional[int] = None,
    case_type: str = "novel_virus",
    hub_id: Optional[str] = None,
    differential_system: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a complete v3 scenario dict from a ground-truth JSON file.

    Parameters
    ----------
    ground_truth_path : str
        Path to the ground-truth JSON produced by ccda_to_ground_truth.py.
    seed : int | None
        RNG seed for reproducibility.  None → random.
    case_type : str
        One of ``novel_virus``, ``flu_like``, ``differential``, ``healthy``.
    hub_id : str | None
        SC hospital hub id (e.g. "H3").  None → random hub.
    differential_system : str | None
        For ``differential`` cases, force a specific body system.
        None → random selection.

    Returns
    -------
    dict with keys:
        demographics, chief_complaint, present_symptoms, negated_symptoms,
        ccda_noise, seed, case_type, location, differential_system (if applicable)
    """
    if case_type not in CASE_TYPES:
        raise ValueError(f"Unknown case_type {case_type!r}. Choose from {CASE_TYPES}")

    gt = _load_ground_truth(ground_truth_path)
    rng = random.Random(seed)

    # Sample symptoms based on case type
    present, chief_complaint, negated, differential_system = sample_symptoms(
        rng, case_type=case_type, differential_system=differential_system
    )

    # If there are injected symptoms, add them to present list
    # (most relevant for novel_virus cases, but allowed for any type)
    injected = []
    for s in gt.get("injected_symptoms", []):
        desc = s["description"] if isinstance(s, dict) else str(s)
        if desc not in present:
            present.append(desc)
            injected.append(desc)

    demographics = _extract_demographics(gt)
    ccda_noise = _extract_ccda_noise(gt)

    # Assign SC hospital hub location and override CCDA city/state
    location = assign_location(rng, hub_id=hub_id)
    demographics["city"] = location["city"]
    demographics["state"] = location["state"]

    scenario = {
        "demographics": demographics,
        "chief_complaint": chief_complaint,
        "present_symptoms": present,
        "negated_symptoms": negated,
        "ccda_noise": ccda_noise,
        "location": location,
        "seed": seed,
        "case_type": case_type,
    }
    if differential_system:
        scenario["differential_system"] = differential_system
    return scenario


# ─── CLI for dry-run testing ─────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a v3 scenario from ground-truth JSON.")
    parser.add_argument("--ground_truth", required=True, help="Path to ground_truth.json")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    parser.add_argument("--case_type", default="novel_virus",
                        choices=list(CASE_TYPES),
                        help="Case type (default: novel_virus)")
    parser.add_argument("--hub", default=None,
                        help="SC hospital hub id (e.g. H3). Default: random")
    parser.add_argument("--out", default="", help="Optional path to write scenario JSON")
    args = parser.parse_args()

    scenario = build_scenario(args.ground_truth, seed=args.seed,
                               case_type=args.case_type, hub_id=args.hub)

    print("\n" + "=" * 60)
    print("  v3 Scenario (dry-run)")
    print("=" * 60)
    print(f"\n  Case type: {scenario['case_type']}")
    if scenario.get('differential_system'):
        print(f"  System focus: {scenario['differential_system']}")
    loc = scenario.get("location", {})
    if loc:
        print(f"  Location: {loc.get('hospital', '?')} ({loc.get('city')}, {loc.get('state')})")
        print(f"  Coords: {loc.get('lat')}, {loc.get('lng')}")
    print(f"  Patient: {scenario['demographics']['name']}")
    print(f"  Chief complaint: {scenario['chief_complaint']}")
    print(f"\n  Present symptoms ({len(scenario['present_symptoms'])}):")
    for s in scenario["present_symptoms"]:
        print(f"    🟢 {s}")
    print(f"\n  Negated symptoms ({len(scenario['negated_symptoms'])}):")
    for s in scenario["negated_symptoms"]:
        print(f"    🔴 {s}")
    print(f"\n  CCDA noise:")
    noise = scenario["ccda_noise"]
    print(f"    Conditions ({len(noise['conditions'])}): {noise['conditions']}")
    print(f"    Medications ({len(noise['medications'])}): {noise['medications']}")
    print(f"    Allergies: {noise['allergies']}")
    print(f"    Social history: {noise['social_history'][:3]}")
    if noise["injected_symptoms"]:
        print(f"    Injected: {noise['injected_symptoms']}")
    print(f"\n  Seed: {scenario['seed']}")

    if args.out:
        out_path = os.path.abspath(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scenario, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved to: {out_path}")

    print()


if __name__ == "__main__":
    main()
