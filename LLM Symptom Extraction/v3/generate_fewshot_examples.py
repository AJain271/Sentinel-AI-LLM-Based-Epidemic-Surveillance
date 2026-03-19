"""
Generate 3 few-shot example transcripts for the v3 extraction pipeline.

Creates one novel_virus, one flu_like, and one differential transcript using
the existing v3 generation pipeline. For the novel_virus case, monkey-patches
NOVEL_VIRUS_HALLMARKS with 3 completely fake symptoms (not the study's 5
hallmarks) so the few-shot examples teach the LLM the quote-binding pattern
without introducing bias about the real novel symptoms.

Usage:
    python generate_fewshot_examples.py

Outputs:
    LLM Symptom Extraction/v3/few_shot_examples/
        example_novel_transcript.txt
        example_novel_metadata.json
        example_flu_transcript.txt
        example_flu_metadata.json
        example_diff_transcript.txt
        example_diff_metadata.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

# ─── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # snhs symposium results
V3_PIPELINE_DIR = os.path.join(ROOT_DIR, "Synthetic Transcript Generation", "synthetic generation", "v3")
CCDA_DIR = os.path.join(ROOT_DIR, "Synthetic Transcript Generation", "Synthea CCDAs", "to use")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "few_shot_examples")

# ─── Fake novel symptoms (NOT the study's 5 hallmarks) ──────────────────────
# These are medically plausible symptoms completely distinct from:
#   Lymphadenopathy, Dysgeusia, Hemoptysis, Skin desquamation, Melanonychia
# They exist only to teach the LLM the pattern of quote-binding unmapped symptoms.

FAKE_NOVEL_SYMPTOMS = [
    "Periorbital cyanosis (blue-purple discoloration around the eyes)",
    "Odynophagia with vesicles (painful swallowing with small blisters in throat)",
    "Paronychia (swollen, red, tender nail beds)",
]

# ─── Case plan ───────────────────────────────────────────────────────────────
# Each tuple: (case_type, seed, output_prefix)
CASE_PLAN = [
    ("novel_virus",  9001, "example_novel"),
    ("flu_like",     9002, "example_flu"),
    ("differential", 9003, "example_diff"),
]


def main() -> None:
    # Add v3 pipeline dir to path so we can import its modules
    if V3_PIPELINE_DIR not in sys.path:
        sys.path.insert(0, V3_PIPELINE_DIR)

    # Monkey-patch NOVEL_VIRUS_HALLMARKS BEFORE importing modules that use it
    import symptom_library_v3
    original_hallmarks = symptom_library_v3.NOVEL_VIRUS_HALLMARKS[:]
    symptom_library_v3.NOVEL_VIRUS_HALLMARKS = FAKE_NOVEL_SYMPTOMS

    # Also patch the feasibility cluster for novel_hallmark
    original_cluster = symptom_library_v3.FEASIBILITY_CLUSTERS.get("novel_hallmark", [])[:]
    symptom_library_v3.FEASIBILITY_CLUSTERS["novel_hallmark"] = FAKE_NOVEL_SYMPTOMS

    import run_pipeline_v3 as pipeline

    # Gather 3 unused CCDAs
    xml_files = sorted(glob.glob(os.path.join(CCDA_DIR, "*.xml")))
    if len(xml_files) < 3:
        print(f"ERROR: Need at least 3 CCDAs in {CCDA_DIR}, found {len(xml_files)}")
        sys.exit(1)

    # Use the last 3 CCDAs in the list (least likely to overlap with batch 500
    # which used the first ~500)
    selected_ccdas = xml_files[-3:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Redirect pipeline output
    pipeline.OUTPUT_DIR = OUTPUT_DIR

    print("=" * 60)
    print("  Generating Few-Shot Example Transcripts")
    print("  (using FAKE novel symptoms — NOT study hallmarks)")
    print("=" * 60)

    for i, (case_type, seed, prefix) in enumerate(CASE_PLAN):
        ccda = selected_ccdas[i]
        print(f"\n>>> Example {i+1}/{len(CASE_PLAN)}: {case_type} | seed={seed}")
        print(f"    CCDA: {os.path.basename(ccda)}")

        if case_type == "novel_virus":
            print(f"    Fake novel symptoms: {FAKE_NOVEL_SYMPTOMS}")

        transcript_path = pipeline.execute_pipeline(
            ccda, seed=seed, case_type=case_type,
        )

        # Rename outputs to our standard names
        transcript_dir = os.path.dirname(transcript_path)

        # Find and rename the generated files
        for f in os.listdir(transcript_dir):
            full_path = os.path.join(transcript_dir, f)
            if f.endswith(".txt") and "SYNTHETIC" in f and str(seed) in f:
                new_name = f"{prefix}_transcript.txt"
                os.replace(full_path, os.path.join(OUTPUT_DIR, new_name))
                print(f"    Renamed: {f} -> {new_name}")
            elif f.endswith(".json") and "METADATA" in f and str(seed) in f:
                new_name = f"{prefix}_metadata.json"
                os.replace(full_path, os.path.join(OUTPUT_DIR, new_name))
                print(f"    Renamed: {f} -> {new_name}")
            elif f.endswith(".json") and "scenario" in f and str(seed) in f:
                new_name = f"{prefix}_scenario.json"
                os.replace(full_path, os.path.join(OUTPUT_DIR, new_name))
                print(f"    Renamed: {f} -> {new_name}")

    # Restore original hallmarks (good practice even though script exits)
    symptom_library_v3.NOVEL_VIRUS_HALLMARKS = original_hallmarks
    symptom_library_v3.FEASIBILITY_CLUSTERS["novel_hallmark"] = original_cluster

    print("\n" + "=" * 60)
    print("  Few-shot example generation complete!")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Review transcripts to verify fake novel symptoms appear naturally")
    print("  2. Create expected extraction JSONs (example_*_expected.json)")
    print("  3. Run the extraction pipeline with these examples")


if __name__ == "__main__":
    main()
