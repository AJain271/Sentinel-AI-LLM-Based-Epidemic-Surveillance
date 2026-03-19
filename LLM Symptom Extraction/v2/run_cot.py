"""
Runner for the Chain-of-Thought symptom extraction pipeline.

Usage:
    python run_cot.py                # extract + evaluate
    python run_cot.py --extract      # extract only
    python run_cot.py --evaluate     # evaluate only (requires prior extraction)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from extract_cot import extract_all
from evaluate import evaluate_all

OUTPUT_DIR = Path(__file__).resolve().parent / "output_cot"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chain-of-Thought symptom extraction pipeline."
    )
    parser.add_argument("--extract", action="store_true",
                        help="Run extraction only.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation only (uses existing output_cot/).")
    args = parser.parse_args()

    run_extract = True
    run_evaluate = True

    if args.extract and not args.evaluate:
        run_evaluate = False
    elif args.evaluate and not args.extract:
        run_extract = False

    if run_extract:
        extract_all(output_dir=OUTPUT_DIR)

    if run_evaluate:
        evaluate_all(output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
