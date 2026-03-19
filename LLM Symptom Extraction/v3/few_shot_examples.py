"""
Load few-shot examples for the v3 extraction prompt.

Reads the three (transcript, expected JSON) pairs from the
few_shot_examples/ directory and formats them into a prompt block
that can be prepended to the user message in extract_fewshot.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from config import FEW_SHOT_DIR

# (transcript_file, expected_file, label)
# Only 1 example to keep token cost comparable to zero-shot.
_EXAMPLE_SPECS = [
    ("example_novel_transcript.txt", "example_novel_expected.json", "Example (novel virus case)"),
]


def load_examples(
    examples_dir: Path | None = None,
) -> List[Tuple[str, str, str]]:
    """Return a list of (label, transcript_text, expected_json_text) tuples."""
    examples_dir = examples_dir or FEW_SHOT_DIR
    examples = []
    for t_file, e_file, label in _EXAMPLE_SPECS:
        transcript = (examples_dir / t_file).read_text(encoding="utf-8")
        expected = (examples_dir / e_file).read_text(encoding="utf-8")
        examples.append((label, transcript, expected))
    return examples


def _compress_expected_json(expected_json_text: str) -> str:
    """Keep only non-zero checklist scores to cut token usage."""
    data = json.loads(expected_json_text)
    scores = data.get("checklist_scores", {})
    nonzero = {k: v for k, v in scores.items() if v != 0}
    compressed = {
        "checklist_scores": {
            "__NOTE__": "All 70 symptoms are scored. Only non-zero shown here; every other symptom is 0.",
            **nonzero,
        },
        "unmapped_symptoms": data.get("unmapped_symptoms", []),
    }
    return json.dumps(compressed, indent=2)


def format_examples_block(
    examples: List[Tuple[str, str, str]] | None = None,
) -> str:
    """Format loaded examples into a single string block for the prompt."""
    if examples is None:
        examples = load_examples()

    parts = []
    for label, transcript, expected_json in examples:
        compressed = _compress_expected_json(expected_json)
        parts.append(
            f"──── {label} ────\n"
            f"TRANSCRIPT:\n{transcript}\n\n"
            f"CORRECT OUTPUT:\n{compressed}\n"
        )
    return "\n\n".join(parts)


if __name__ == "__main__":
    examples = load_examples()
    print(f"Loaded {len(examples)} few-shot examples:\n")
    for label, transcript, expected_json in examples:
        expected = json.loads(expected_json)
        present = sum(1 for v in expected["checklist_scores"].values() if v == 1)
        negated = sum(1 for v in expected["checklist_scores"].values() if v == -1)
        unmapped = len(expected.get("unmapped_symptoms", []))
        print(f"  {label}")
        print(f"    Transcript: {len(transcript.splitlines())} lines")
        print(f"    Expected: {present} present, {negated} negated, {unmapped} unmapped\n")
