"""
Interactive v2 pipeline runner.

Flow:
1. Let user pick a Synthea CCDA XML.
2. Optionally specify injected symptom descriptions.
3. Optionally choose the OpenAI model.
4. Run v2 pipeline:
   - CCDA → ground_truth_v2.json
   - Inject symptoms into ground_truth_v2 (if provided)
   - ground_truth_v2(+injected) → scenario_v2 → transcript_v2
5. Launch the v2 CCDA comparison viewer in the browser.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from typing import List


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CCDA_DIR = os.path.join(SCRIPT_DIR, "..", "Synthea CCDAs")
V2_DIR = os.path.join(SCRIPT_DIR, "v2")
VIEWER_V2 = os.path.join(SCRIPT_DIR, "..", "..", "Dashboards + Annotater", "ccda_comparison_viewer_v2.py")


def run_step(command: List[str], description: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Pipeline failed at step: {description}")
        print(f"Command run: {' '.join(command)}")
        sys.exit(1)


def execute_pipeline_v2(ccda_path: str, symptoms_list: List[str], model: str) -> None:
    base_xml = os.path.basename(ccda_path)
    base_name = base_xml.replace(".xml", "")

    print(f"\nStarting v2 Pipeline for: {base_xml}")

    # 1) Parse CCDA → v2 ground truth
    gt_v2_path = os.path.join(V2_DIR, f"{base_name}_ground_truth_v2.json")
    run_step(
        [
            sys.executable,
            os.path.join(V2_DIR, "ccda_to_ground_truth_v2.py"),
            "--ccda",
            ccda_path,
        ],
        "Step 1 (v2): Parse C-CDA to Ground Truth v2 JSON",
    )

    # 2) Inject symptoms into v2 ground truth (optional)
    if symptoms_list:
        modified_gt_v2 = os.path.join(V2_DIR, f"{base_name}_ground_truth_v2_modified.json")
        inject_cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "inject_symptoms.py"),
            "--input",
            gt_v2_path,
            "--output",
            modified_gt_v2,
        ]
        for symp in symptoms_list:
            inject_cmd.extend(["--symptom", symp])
        run_step(
            inject_cmd,
            "Step 2 (v2): Injecting Novel Symptoms into v2 Ground Truth",
        )
        gt_for_generation = modified_gt_v2
    else:
        print(f"\n{'='*60}")
        print("  Step 2 (v2): Skipped (no symptoms provided to inject)")
        print(f"{'='*60}")
        gt_for_generation = gt_v2_path

    # 3) Generate synthetic transcript via v2 pipeline
    out_dir = os.path.join(SCRIPT_DIR, "..", "Synthetic Transcripts", "Generated Transcripts v2")
    os.makedirs(out_dir, exist_ok=True)
    transcript_path = os.path.join(out_dir, f"SYNTHETIC_{base_name}_v2.txt")

    run_step(
        [
            sys.executable,
            os.path.join(V2_DIR, "generate_synthetic_transcript_v2.py"),
            "--ccda",
            ccda_path,
            "--ground_truth_out",
            gt_for_generation,
            "--out",
            transcript_path,
            "--model",
            model,
            "--min_turns",
            "30",
            "--max_turns",
            "45",
        ],
        "Step 3 (v2): Generate Synthetic Transcript via LLM",
    )

    # 4) Launch v2 comparison viewer
    if not os.path.exists(VIEWER_V2):
        print(f"\nWARNING: v2 viewer script not found: {VIEWER_V2}")
    else:
        run_step(
            [
                sys.executable,
                VIEWER_V2,
                gt_for_generation,
                transcript_path,
            ],
            "Step 4 (v2): Launching Comparison Viewer v2",
        )

    print("\n✓ v2 Pipeline completed successfully!")


def main_interactive() -> None:
    print("======================================================")
    print("      Synthetic Transcript Generation Pipeline v2")
    print("======================================================")

    if not os.path.exists(CCDA_DIR):
        print(f"ERROR: CCDA directory not found: {CCDA_DIR}")
        return

    xml_files = glob.glob(os.path.join(CCDA_DIR, "*.xml"))
    if not xml_files:
        print(f"No XML files found in {CCDA_DIR}")
        return

    print("\nAvailable Synthea CCDA files:")
    for i, file_path in enumerate(xml_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")

    # Select CCDA
    while True:
        try:
            choice = input("\nSelect a CCDA file by number (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                return
            idx = int(choice) - 1
            if 0 <= idx < len(xml_files):
                selected_ccda = xml_files[idx]
                break
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Prompt for injected symptoms
    print(f"\nSelected: {os.path.basename(selected_ccda)}")
    symptoms_input = input(
        "Enter novel symptoms to inject (comma-separated, optional), or press Enter to skip: "
    ).strip()
    symptoms_list: List[str] = []
    if symptoms_input:
        symptoms_list = [s.strip() for s in symptoms_input.split(",") if s.strip()]

    # Prompt for model (with default)
    default_model = "gpt-4o-mini"
    model_input = input(f"Enter OpenAI chat model to use [{default_model}]: ").strip()
    model = model_input or default_model

    execute_pipeline_v2(selected_ccda, symptoms_list, model)


if __name__ == "__main__":
    main_interactive()

