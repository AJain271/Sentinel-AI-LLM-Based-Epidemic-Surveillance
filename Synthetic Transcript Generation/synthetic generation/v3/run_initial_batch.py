"""
Generate the Initial Batch Test — 5 cases (2 novel_virus, 1 flu_like,
1 differential, 1 healthy) using the 5 unique Synthea CCDAs.

Outputs are saved under:
    Synthetic Transcript Generation/Initial Batch Test/transcripts/
    Synthetic Transcript Generation/Initial Batch Test/metadata/
"""

from __future__ import annotations

import glob
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # synthetic generation/
CCDA_DIR = os.path.join(PARENT_DIR, "..", "Synthea CCDAs")
BATCH_DIR = os.path.join(PARENT_DIR, "..", "Initial Batch Test")
TRANSCRIPT_DIR = os.path.join(BATCH_DIR, "transcripts")
METADATA_DIR = os.path.join(BATCH_DIR, "metadata")

# Assignments: (case_type, seed)
CASE_PLAN = [
    ("novel_virus",  101),
    ("novel_virus",  202),
    ("flu_like",     303),
    ("differential", 404),
    ("healthy",      505),
]

def main() -> None:
    # Gather CCDAs
    xml_files = sorted(glob.glob(os.path.join(CCDA_DIR, "*.xml")))
    if len(xml_files) < len(CASE_PLAN):
        print(f"ERROR: Expected at least {len(CASE_PLAN)} CCDAs in {CCDA_DIR}, found {len(xml_files)}")
        sys.exit(1)

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    # Temporarily redirect OUTPUT_DIR used by run_pipeline_v3
    import run_pipeline_v3 as pipeline
    pipeline.OUTPUT_DIR = TRANSCRIPT_DIR  # transcripts + scenario JSONs go here

    print("=" * 60)
    print("  Initial Batch Test — 5 cases")
    print("=" * 60)
    for i, (case_type, seed) in enumerate(CASE_PLAN):
        ccda = xml_files[i]
        print(f"\n>>> Case {i+1}/{len(CASE_PLAN)}: {case_type} | seed={seed} | {os.path.basename(ccda)}")
        pipeline.execute_pipeline(ccda, seed=seed, case_type=case_type)

    # Move metadata files from transcript dir to metadata dir
    for f in os.listdir(TRANSCRIPT_DIR):
        if "METADATA" in f and f.endswith(".json"):
            src = os.path.join(TRANSCRIPT_DIR, f)
            dst = os.path.join(METADATA_DIR, f)
            os.replace(src, dst)
            print(f"  Moved metadata: {f}")

    # Also move scenario JSONs to metadata dir
    for f in os.listdir(TRANSCRIPT_DIR):
        if f.startswith("scenario_v3_") and f.endswith(".json"):
            src = os.path.join(TRANSCRIPT_DIR, f)
            dst = os.path.join(METADATA_DIR, f)
            os.replace(src, dst)
            print(f"  Moved scenario: {f}")

    print("\n" + "=" * 60)
    print("  Batch complete!")
    print(f"  Transcripts: {TRANSCRIPT_DIR}")
    print(f"  Metadata:    {METADATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
