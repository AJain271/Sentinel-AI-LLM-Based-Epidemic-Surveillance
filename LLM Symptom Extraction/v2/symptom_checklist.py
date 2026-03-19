"""
Builds the known symptom checklist for zero-shot extraction.

Takes the master symptom list (~71 symptoms) and explicitly removes the
5 novel virus hallmark symptoms. The LLM will only be asked to score
symptoms on this known list.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from config import SYMPTOM_LIB_DIR

# Add the v3 symptom library directory to sys.path so we can import from it
sys.path.insert(0, str(SYMPTOM_LIB_DIR))

from master_symptom_list import MASTER_SYMPTOM_LIST  # noqa: E402
from symptom_library_v3 import NOVEL_VIRUS_HALLMARKS  # noqa: E402


def build_known_checklist() -> List[str]:
    """Return the master symptom list with novel hallmarks removed."""
    novel_set = set(NOVEL_VIRUS_HALLMARKS)
    return [s for s in MASTER_SYMPTOM_LIST if s not in novel_set]


KNOWN_SYMPTOM_LIST: List[str] = build_known_checklist()


if __name__ == "__main__":
    print(f"Known symptom checklist ({len(KNOWN_SYMPTOM_LIST)} symptoms):\n")
    for i, s in enumerate(KNOWN_SYMPTOM_LIST, 1):
        print(f"  {i:3d}. {s}")

    print(f"\nExcluded novel hallmarks ({len(NOVEL_VIRUS_HALLMARKS)}):")
    for s in NOVEL_VIRUS_HALLMARKS:
        print(f"    - {s}")
