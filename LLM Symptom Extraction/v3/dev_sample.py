"""
Dev-sample selection for v3 extraction.

Selects a small stratified sample from Batch 500 for rapid iteration:
  2 novel  +  1 flu  +  1 differential  +  1 healthy  =  5 transcripts

All three runners import this to ensure they evaluate the exact same set.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

from config import TRANSCRIPT_DIR, DEV_SAMPLE_SEED


def select_dev_sample(
    transcript_dir: Path | None = None,
    seed: int | None = None,
) -> List[Path]:
    """Return a deterministic stratified sample of transcript paths."""
    transcript_dir = transcript_dir or TRANSCRIPT_DIR
    seed = seed if seed is not None else DEV_SAMPLE_SEED

    # Group transcripts by case-type prefix
    all_txts = sorted(transcript_dir.glob("*_SYNTHETIC_*.txt"))
    by_type: Dict[str, List[Path]] = {}
    for p in all_txts:
        prefix = p.name.split("_")[0]  # NOVEL, FLU, DIFF, HEALTHY
        by_type.setdefault(prefix, []).append(p)

    # Stratified draw
    rng = random.Random(seed)
    sample: List[Path] = []

    quotas = {"NOVEL": 2, "FLU": 1, "DIFF": 1, "HEALTHY": 1}
    for case_type, n in quotas.items():
        pool = by_type.get(case_type, [])
        if len(pool) < n:
            raise ValueError(
                f"Need {n} {case_type} transcripts but only found {len(pool)}"
            )
        sample.extend(rng.sample(pool, n))

    return sorted(sample)


if __name__ == "__main__":
    paths = select_dev_sample()
    print(f"Dev sample ({len(paths)} transcripts):\n")
    for p in paths:
        print(f"  {p.name}")
