"""
Runner for the v3 regex baseline symptom extraction pipeline.

Usage:
    python run_regex.py                # extract + evaluate on dev sample
    python run_regex.py --extract      # extract only
    python run_regex.py --evaluate     # evaluate only
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import OUTPUT_DIR, METADATA_DIR
from dev_sample import select_dev_sample
from extract_regex import extract_all
from evaluate import evaluate_all


_OUTPUT = OUTPUT_DIR.parent / "output_regex"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regex baseline symptom extraction (v3)."
    )
    parser.add_argument("--extract", action="store_true",
                        help="Run extraction only.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation only.")
    args = parser.parse_args()

    run_extract = True
    run_evaluate = True
    if args.extract and not args.evaluate:
        run_evaluate = False
    elif args.evaluate and not args.extract:
        run_extract = False

    if run_extract:
        sample = select_dev_sample()
        print(f"Dev sample: {len(sample)} transcripts")
        for p in sample:
            print(f"  {p.name}")
        print()
        extract_all(transcript_paths=sample, output_dir=_OUTPUT)

    if run_evaluate:
        evaluate_all(output_dir=_OUTPUT, metadata_dir=METADATA_DIR)


if __name__ == "__main__":
    main()
