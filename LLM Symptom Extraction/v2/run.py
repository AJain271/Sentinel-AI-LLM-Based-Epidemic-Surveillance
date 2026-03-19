"""
Orchestrator for the v2 zero-shot LLM symptom extraction pipeline.

Usage:
    python run.py                # extract + evaluate
    python run.py --extract      # extract only
    python run.py --evaluate     # evaluate only (requires prior extraction)
"""

from __future__ import annotations

import argparse

from extract import extract_all
from evaluate import evaluate_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM symptom extraction pipeline (v2)."
    )
    parser.add_argument("--extract", action="store_true",
                        help="Run extraction only.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation only (uses existing output/).")
    args = parser.parse_args()

    run_extract = True
    run_evaluate = True

    if args.extract and not args.evaluate:
        run_evaluate = False
    elif args.evaluate and not args.extract:
        run_extract = False

    if run_extract:
        extract_all()

    if run_evaluate:
        evaluate_all()


if __name__ == "__main__":
    main()
