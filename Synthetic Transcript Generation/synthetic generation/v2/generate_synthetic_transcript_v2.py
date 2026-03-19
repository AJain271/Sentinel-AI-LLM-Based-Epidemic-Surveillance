"""
v2 entrypoint: CCDA → ground_truth_v2 → scenario_v2 → transcript.

This script does not modify or depend on v1 outputs beyond reusing the v1 CCDA parser.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

if __package__:
    from .ccda_to_ground_truth_v2 import parse_ccda_to_ground_truth_v2
    from .scenario_generator_v2 import build_scenario
    from .conversation_orchestrator_v2 import OrchestrationConfig, generate_transcript_from_scenario
else:
    # Allow running as a script: `python generate_synthetic_transcript_v2.py ...`
    V2_DIR = os.path.dirname(os.path.abspath(__file__))
    if V2_DIR not in sys.path:
        sys.path.insert(0, V2_DIR)
    from ccda_to_ground_truth_v2 import parse_ccda_to_ground_truth_v2  # type: ignore
    from scenario_generator_v2 import build_scenario  # type: ignore
    from conversation_orchestrator_v2 import OrchestrationConfig, generate_transcript_from_scenario  # type: ignore


def _default_paths(ccda_path: str) -> Dict[str, str]:
    v2_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.basename(ccda_path).replace(".xml", "")

    gt_path = os.path.join(v2_dir, f"{base}_ground_truth_v2.json")
    scenario_path = os.path.join(v2_dir, f"{base}_scenario_v2.json")

    out_dir = os.path.abspath(
        os.path.join(v2_dir, "..", "..", "Synthetic Transcripts", "Generated Transcripts v2")
    )
    out_path = os.path.join(out_dir, f"SYNTHETIC_{base}_v2.txt")

    catalog_path = os.path.join(v2_dir, "symptom_catalog_v2.json")

    return {
        "ground_truth": gt_path,
        "scenario": scenario_path,
        "out_dir": out_dir,
        "out": out_path,
        "catalog": catalog_path,
    }


def run_v2(
    ccda_path: str,
    catalog_path: str,
    ground_truth_out: str,
    scenario_out: str,
    transcript_out: str,
    min_turns: int,
    max_turns: int,
    model: str,
    temperature: float,
    seed: Optional[int],
) -> str:
    # If a ground-truth file already exists (possibly with injected symptoms),
    # reuse it; otherwise, parse CCDA to create it.
    if os.path.exists(ground_truth_out):
        with open(ground_truth_out, "r", encoding="utf-8") as f:
            gt_v2 = json.load(f)
    else:
        gt_v2 = parse_ccda_to_ground_truth_v2(ccda_path)
        os.makedirs(os.path.dirname(os.path.abspath(ground_truth_out)), exist_ok=True)
        with open(ground_truth_out, "w", encoding="utf-8") as f:
            json.dump(gt_v2, f, indent=2, ensure_ascii=False)

    scenario = build_scenario(
        ground_truth_path=ground_truth_out,
        catalog_path=catalog_path,
        seed=seed,
    )
    with open(scenario_out, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)

    config = OrchestrationConfig(
        min_turns=min_turns,
        max_turns=max_turns,
        model=model,
        temperature=temperature,
        seed=seed,
    )
    transcript = generate_transcript_from_scenario(scenario, config)

    os.makedirs(os.path.dirname(os.path.abspath(transcript_out)), exist_ok=True)
    with open(transcript_out, "w", encoding="utf-8") as f:
        f.write(transcript)

    return transcript_out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the synthetic generation v2 pipeline.")
    parser.add_argument("--ccda", required=True, help="Path to Synthea CCDA XML")
    parser.add_argument("--catalog", default="", help="Path to symptom_catalog_v2.json (defaults to v2 folder)")
    parser.add_argument("--ground_truth_out", default="", help="Optional path to write ground truth v2 JSON")
    parser.add_argument("--scenario_out", default="", help="Optional path to write scenario v2 JSON")
    parser.add_argument("--out", default="", help="Optional path to write transcript")
    parser.add_argument("--min_turns", type=int, default=30)
    parser.add_argument("--max_turns", type=int, default=45)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ccda_path = os.path.abspath(args.ccda)
    defaults = _default_paths(ccda_path)

    catalog_path = os.path.abspath(args.catalog) if args.catalog else defaults["catalog"]
    ground_truth_out = os.path.abspath(args.ground_truth_out) if args.ground_truth_out else defaults["ground_truth"]
    scenario_out = os.path.abspath(args.scenario_out) if args.scenario_out else defaults["scenario"]
    transcript_out = os.path.abspath(args.out) if args.out else defaults["out"]

    out_path = run_v2(
        ccda_path=ccda_path,
        catalog_path=catalog_path,
        ground_truth_out=ground_truth_out,
        scenario_out=scenario_out,
        transcript_out=transcript_out,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
    )

    print(out_path)


if __name__ == "__main__":
    main()

