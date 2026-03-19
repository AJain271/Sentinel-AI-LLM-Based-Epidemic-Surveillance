"""
v3 Pipeline Runner — orchestrates CCDA parsing, optional symptom injection,
scenario sampling, and one-shot transcript generation.

Usage:
    python run_pipeline_v3.py --ccda path/to/ccda.xml --case_type novel_virus [--seed 42] [--symptoms "Purple toes"]

Case types:
    novel_virus   — COVID-like viral respiratory illness (default)
    flu_like      — classic influenza presentation
    differential  — single body-system distractor (GI, cardiac, MSK, etc.)
    healthy       — routine wellness checkup (no symptoms)

Or run without arguments for interactive mode.

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from typing import List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # synthetic generation/
CCDA_DIR = os.path.join(PARENT_DIR, "..", "Synthea CCDAs")
OUTPUT_DIR = os.path.join(PARENT_DIR, "..", "Synthetic Transcripts", "Generated Transcripts v3")


def _run_step(command: List[str], description: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        print(f"\nERROR: Pipeline failed at step: {description}")
        print(f"Command: {' '.join(command)}")
        sys.exit(1)


# Filename prefixes per case type
_CASE_TYPE_PREFIX = {
    "novel_virus":  "NOVEL",
    "flu_like":     "FLU",
    "differential": "DIFF",
    "healthy":      "HEALTHY",
}


def execute_pipeline(
    ccda_path: str,
    seed: Optional[int] = None,
    symptoms_list: Optional[List[str]] = None,
    case_type: str = "novel_virus",
    hub_id: Optional[str] = None,
    differential_system: Optional[str] = None,
) -> str:
    """Run the full v3 pipeline and return the output transcript path.

    Steps:
        1. Parse CCDA → ground_truth.json  (reuses v1 ccda_to_ground_truth.py)
        2. (Optional) Inject symptoms       (reuses inject_symptoms.py)
        3. Build v3 scenario                (scenario_builder_v3.py)
        4. Generate transcript              (generate_transcript_v3.py)
        5. Write metadata JSON
        6. (Optional) Launch comparison viewer
    """
    if symptoms_list is None:
        symptoms_list = []

    ccda_path = os.path.abspath(ccda_path)
    ccda_filename = os.path.basename(ccda_path)
    prefix = _CASE_TYPE_PREFIX.get(case_type, "SYNTHETIC")
    print(f"\nStarting v3 pipeline for: {ccda_filename}  [case_type={case_type}, hub={hub_id or 'random'}]")

    # ── Step 1: Parse CCDA ───────────────────────────────────────────
    ccda_parser = os.path.join(PARENT_DIR, "ccda_to_ground_truth.py")
    _run_step(
        [sys.executable, ccda_parser, ccda_path],
        "Step 1: Parse C-CDA → Ground Truth JSON",
    )

    # Resolve ground-truth path (ccda_to_ground_truth.py writes to PARENT_DIR)
    base_json_name = ccda_filename.replace(".xml", "_ground_truth.json")
    modified_json_name = ccda_filename.replace(".xml", "_modified_ground_truth.json")
    base_json = os.path.join(PARENT_DIR, base_json_name)
    modified_json = os.path.join(PARENT_DIR, modified_json_name)

    # ── Step 2: Inject symptoms (optional) ───────────────────────────
    gt_path = base_json
    if symptoms_list:
        inject_cmd = [
            sys.executable,
            os.path.join(PARENT_DIR, "inject_symptoms.py"),
            "--input", base_json,
            "--output", modified_json,
        ]
        for symp in symptoms_list:
            inject_cmd.extend(["--symptom", symp])
        _run_step(inject_cmd, "Step 2: Inject Novel Symptoms")
        gt_path = modified_json
    else:
        print(f"\n{'=' * 60}")
        print("  Step 2: Skipped (no symptoms to inject)")
        print(f"{'=' * 60}")

    # ── Step 3: Build scenario ───────────────────────────────────────
    # Import scenario builder directly (faster than subprocess)
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    from scenario_builder_v3 import build_scenario

    print(f"\n{'=' * 60}")
    print("  Step 3: Build v3 Scenario")
    print(f"{'=' * 60}")

    scenario = build_scenario(gt_path, seed=seed, case_type=case_type, hub_id=hub_id, differential_system=differential_system)

    # Print scenario summary
    print(f"\n  Case type: {scenario['case_type']}")
    if scenario.get('differential_system'):
        print(f"  System focus: {scenario['differential_system']}")
    loc = scenario.get('location', {})
    if loc:
        print(f"  Hospital hub: {loc.get('hospital', '?')} ({loc.get('city')}, {loc.get('state')})")
        print(f"  Patient coords: {loc.get('lat')}, {loc.get('lng')}")
    print(f"  Patient: {scenario['demographics']['name']}")
    print(f"  Chief complaint: {scenario['chief_complaint']}")
    print(f"  Present symptoms ({len(scenario['present_symptoms'])}):")
    for s in scenario["present_symptoms"]:
        print(f"    🟢 {s}")
    print(f"  Negated symptoms ({len(scenario['negated_symptoms'])}):")
    for s in scenario["negated_symptoms"]:
        print(f"    🔴 {s}")
    noise = scenario["ccda_noise"]
    print(f"  CCDA noise: {len(noise['conditions'])} conditions, "
          f"{len(noise['medications'])} meds, "
          f"{len(noise['allergies'])} allergies")

    # Save scenario JSON for reference
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patient_name = scenario["demographics"]["name"].replace(" ", "_")
    seed_tag = f"_s{seed}" if seed is not None else ""
    scenario_path = os.path.join(OUTPUT_DIR, f"scenario_v3_{prefix}_{patient_name}{seed_tag}.json")
    with open(scenario_path, "w", encoding="utf-8") as f:
        import json
        json.dump(scenario, f, indent=2, ensure_ascii=False)
    print(f"\n  Scenario saved: {os.path.basename(scenario_path)}")

    # ── Step 4: Generate transcript ──────────────────────────────────
    from generate_transcript_v3 import build_prompt, generate_transcript

    print(f"\n{'=' * 60}")
    print("  Step 4: Generate Synthetic Transcript (one-shot GPT-4o)")
    print(f"{'=' * 60}")

    system_prompt, user_prompt = build_prompt(scenario)
    transcript, _token_usage = generate_transcript(system_prompt, user_prompt)

    # Save transcript
    transcript_path = os.path.join(OUTPUT_DIR, f"{prefix}_SYNTHETIC_v3_{patient_name}{seed_tag}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # Stats
    lines = transcript.strip().split("\n")
    d_lines = sum(1 for l in lines if l.startswith("D:"))
    p_lines = sum(1 for l in lines if l.startswith("P:"))
    print(f"\n  Saved to: {os.path.basename(transcript_path)}")
    print(f"  Stats: {d_lines} doctor lines, {p_lines} patient lines, {len(lines)} total lines")

    # ── Step 5: Write metadata JSON ────────────────────────────────
    import json as _json
    from datetime import datetime, timezone

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_type": case_type,
        "seed": seed,
        "ccda_source": ccda_filename,
        "demographics": scenario["demographics"],
        "location": scenario.get("location"),
        "chief_complaint": scenario["chief_complaint"],
        "present_symptoms": scenario["present_symptoms"],
        "negated_symptoms": scenario["negated_symptoms"],
        "differential_system": scenario.get("differential_system"),
        "ccda_noise": {
            "conditions": scenario["ccda_noise"]["conditions"],
            "medications": scenario["ccda_noise"]["medications"],
            "allergies": scenario["ccda_noise"]["allergies"],
        },
        "transcript_stats": {
            "total_lines": len(lines),
            "doctor_lines": d_lines,
            "patient_lines": p_lines,
        },
        "transcript_file": os.path.basename(transcript_path),
        "scenario_file": os.path.basename(scenario_path),
    }

    metadata_path = os.path.join(
        OUTPUT_DIR, f"{prefix}_METADATA_v3_{patient_name}{seed_tag}.json"
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        _json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved: {os.path.basename(metadata_path)}")

    # Preview
    print(f"\n{'=' * 60}")
    print("  PREVIEW (first 20 lines)")
    print(f"{'=' * 60}")
    for line in lines[:20]:
        print(line)
    print("...")

    # ── Step 6: Launch v3 comparison viewer ────────────────────────
    viewer_path = os.path.join(PARENT_DIR, "..", "..", "Dashboards + Annotater", "v3_comparison_viewer.py")
    if os.path.exists(viewer_path):
        print(f"\n{'=' * 60}")
        print("  Step 6: Launch v3 Comparison Viewer")
        print(f"{'=' * 60}")
        try:
            subprocess.run([
                sys.executable, viewer_path,
                "--scenario", scenario_path,
                "--transcript", transcript_path,
            ], check=True)
        except subprocess.CalledProcessError:
            print("  (viewer failed — non-critical)")

    print("\n✓ v3 pipeline completed successfully!")
    return transcript_path


# ─── Interactive mode ────────────────────────────────────────────────────────

def main_interactive() -> None:
    print("=" * 60)
    print("  Synthetic Transcript Generation Pipeline (v3)")
    print("=" * 60)

    if not os.path.exists(CCDA_DIR):
        print(f"ERROR: CCDA directory not found: {CCDA_DIR}")
        return

    xml_files = sorted(glob.glob(os.path.join(CCDA_DIR, "*.xml")))
    if not xml_files:
        print(f"No XML files found in {CCDA_DIR}")
        return

    print("\nAvailable Synthea CCDA files:")
    for i, fp in enumerate(xml_files, 1):
        print(f"  {i}. {os.path.basename(fp)}")

    while True:
        choice = input("\nSelect a CCDA file by number (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(xml_files):
                selected_ccda = xml_files[idx]
                break
            print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    print(f"\nSelected: {os.path.basename(selected_ccda)}")

    # Case type
    case_types = ["novel_virus", "flu_like", "differential", "healthy"]
    print("\nCase types:")
    for i, ct in enumerate(case_types, 1):
        print(f"  {i}. {ct}")
    while True:
        ct_choice = input("Select case type by number (default: 1 = novel_virus): ").strip()
        if not ct_choice:
            case_type = "novel_virus"
            break
        try:
            ct_idx = int(ct_choice) - 1
            if 0 <= ct_idx < len(case_types):
                case_type = case_types[ct_idx]
                break
            print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")
    print(f"  Case type: {case_type}")

    # Hospital hub
    from sc_hospital_hubs import SC_HOSPITAL_HUBS
    print("\nSC Hospital Hubs:")
    for i, hub in enumerate(SC_HOSPITAL_HUBS, 1):
        print(f"  {i}. [{hub['hub_id']}] {hub['name']} ({hub['city']})")
    print(f"  {len(SC_HOSPITAL_HUBS) + 1}. Random (pick for me)")
    while True:
        hub_choice = input(f"Select hub by number (default: {len(SC_HOSPITAL_HUBS) + 1} = random): ").strip()
        if not hub_choice:
            hub_id = None
            break
        try:
            h_idx = int(hub_choice) - 1
            if h_idx == len(SC_HOSPITAL_HUBS):
                hub_id = None
                break
            if 0 <= h_idx < len(SC_HOSPITAL_HUBS):
                hub_id = SC_HOSPITAL_HUBS[h_idx]["hub_id"]
                break
            print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")
    print(f"  Hub: {hub_id or 'random'}")

    # Seed
    seed_input = input("Enter RNG seed (or press Enter for random): ").strip()
    seed = int(seed_input) if seed_input else None

    # Symptoms to inject
    symptoms_input = input("Enter novel symptoms to inject (comma-separated), or press Enter to skip: ").strip()
    symptoms_list = [s.strip() for s in symptoms_input.split(",") if s.strip()] if symptoms_input else []

    execute_pipeline(selected_ccda, seed=seed, symptoms_list=symptoms_list,
                      case_type=case_type, hub_id=hub_id)


# ─── CLI with args ───────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="v3 Synthetic Transcript Generation Pipeline",
        epilog="Run without arguments for interactive mode.",
    )
    parser.add_argument("--ccda", default="", help="Path to Synthea CCDA XML file")
    parser.add_argument("--case_type", default="novel_virus",
                        choices=["novel_virus", "flu_like", "differential", "healthy"],
                        help="Case type (default: novel_virus)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    parser.add_argument("--hub", default=None,
                        help="SC hospital hub id (e.g. H3). Default: random")
    parser.add_argument("--symptoms", nargs="*", default=[], help="Novel symptoms to inject")
    args = parser.parse_args()

    if not args.ccda:
        main_interactive()
    else:
        execute_pipeline(args.ccda, seed=args.seed, symptoms_list=args.symptoms,
                          case_type=args.case_type, hub_id=args.hub)


if __name__ == "__main__":
    main()
