"""
Master symptom list — single source of truth for downstream LLM extraction.

Collects every unique symptom string from SYMPTOM_LIBRARY,
FLU_LIKE_SYMPTOMS, FLU_RELEVANT_NEGATIONS, and NON_COVID_DIFFERENTIALS,
deduplicates, and exposes a sorted list.

Usage (standalone):
    python master_symptom_list.py            # pretty-print
    python master_symptom_list.py --json     # JSON array to stdout
"""

from __future__ import annotations

from typing import List

from symptom_library_v3 import (
    FLU_LIKE_SYMPTOMS,
    FLU_RELEVANT_NEGATIONS,
    NON_COVID_DIFFERENTIALS,
    SYMPTOM_LIBRARY,
)


def build_master_list() -> List[str]:
    """Return a sorted, deduplicated list of every symptom string."""
    pool: set = set()

    for symptoms in SYMPTOM_LIBRARY.values():
        pool.update(symptoms)

    for symptoms in FLU_LIKE_SYMPTOMS.values():
        pool.update(symptoms)

    pool.update(FLU_RELEVANT_NEGATIONS)

    for symptoms in NON_COVID_DIFFERENTIALS.values():
        pool.update(symptoms)

    return sorted(pool)


MASTER_SYMPTOM_LIST: List[str] = build_master_list()


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Print the master symptom list.")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Output as JSON array")
    args = parser.parse_args()

    if args.as_json:
        print(json.dumps(MASTER_SYMPTOM_LIST, indent=2))
    else:
        print(f"Master symptom list ({len(MASTER_SYMPTOM_LIST)} unique symptoms):\n")
        for i, s in enumerate(MASTER_SYMPTOM_LIST, 1):
            print(f"  {i:3d}. {s}")
