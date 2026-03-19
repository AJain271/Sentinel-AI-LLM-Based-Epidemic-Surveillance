"""
Benchmark sample selector for v3 extraction.

Selects a deterministic stratified sample of 100 transcripts from Batch 500:
    25 novel_virus
    25 flu_like
    25 differential  (7 musculoskeletal + 6 dermatological +
                      6 gastrointestinal + 6 neurological)
    25 healthy

All methods and all trials use the SAME 100 transcripts for fair comparison.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from config import TRANSCRIPT_DIR, METADATA_DIR, BENCHMARK_SEED


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _transcript_to_metadata_name(transcript_filename: str) -> str:
    """NOVEL_SYNTHETIC_v3_...s101.txt → NOVEL_METADATA_v3_...s101.json"""
    name = transcript_filename.replace(".txt", "")
    return name.replace("_SYNTHETIC_", "_METADATA_") + ".json"


def _get_differential_system(transcript_path: Path, metadata_dir: Path) -> str | None:
    """Read the metadata JSON for a DIFF transcript to get its body system."""
    meta_name = _transcript_to_metadata_name(transcript_path.name)
    meta_path = metadata_dir / meta_name
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("differential_system")


# ─── Main selector ───────────────────────────────────────────────────────────

def select_benchmark_sample(
    transcript_dir: Path | None = None,
    metadata_dir: Path | None = None,
    seed: int | None = None,
) -> Tuple[List[Path], Dict[str, str]]:
    """Return a deterministic stratified sample of 100 transcript paths.

    Returns
    -------
    sample_paths : list[Path]
        100 transcript file paths, sorted.
    case_type_map : dict[str, str]
        Maps filename → full case label, e.g.:
            "NOVEL_SYNTHETIC_v3_...txt" → "novel_virus"
            "DIFF_SYNTHETIC_v3_...txt"  → "differential_musculoskeletal"
    """
    transcript_dir = transcript_dir or TRANSCRIPT_DIR
    metadata_dir = metadata_dir or METADATA_DIR
    seed = seed if seed is not None else BENCHMARK_SEED
    rng = random.Random(seed)

    # Gather all synthetic transcripts
    all_txts = sorted(transcript_dir.glob("*_SYNTHETIC_*.txt"))

    # Group by prefix
    by_prefix: Dict[str, List[Path]] = {}
    for p in all_txts:
        prefix = p.name.split("_")[0]  # NOVEL, FLU, DIFF, HEALTHY
        by_prefix.setdefault(prefix, []).append(p)

    # ── NOVEL: 25 ────────────────────────────────────────────────────────
    novel_pool = by_prefix.get("NOVEL", [])
    if len(novel_pool) < 25:
        raise ValueError(f"Need 25 NOVEL but only found {len(novel_pool)}")
    novel = rng.sample(novel_pool, 25)

    # ── FLU: 25 ──────────────────────────────────────────────────────────
    flu_pool = by_prefix.get("FLU", [])
    if len(flu_pool) < 25:
        raise ValueError(f"Need 25 FLU but only found {len(flu_pool)}")
    flu = rng.sample(flu_pool, 25)

    # ── DIFF: 25 (7 MSK + 6 derm + 6 gastro + 6 neuro) ──────────────────
    diff_pool = by_prefix.get("DIFF", [])
    diff_by_system: Dict[str, List[Path]] = {}
    for dp in diff_pool:
        sys = _get_differential_system(dp, metadata_dir)
        if sys:
            diff_by_system.setdefault(sys, []).append(dp)

    diff_quotas = {
        "musculoskeletal": 7,
        "dermatological": 6,
        "gastrointestinal": 6,
        "neurological": 6,
    }
    diff_selected: List[Path] = []
    for system, n in diff_quotas.items():
        pool = diff_by_system.get(system, [])
        if len(pool) < n:
            raise ValueError(
                f"Need {n} DIFF-{system} but only found {len(pool)}"
            )
        diff_selected.extend(rng.sample(pool, n))

    # ── HEALTHY: 25 ──────────────────────────────────────────────────────
    healthy_pool = by_prefix.get("HEALTHY", [])
    if len(healthy_pool) < 25:
        raise ValueError(f"Need 25 HEALTHY but only found {len(healthy_pool)}")
    healthy = rng.sample(healthy_pool, 25)

    # ── Combine + sort ───────────────────────────────────────────────────
    sample = sorted(novel + flu + diff_selected + healthy)

    # Build case type map
    case_type_map: Dict[str, str] = {}
    for p in novel:
        case_type_map[p.name] = "novel_virus"
    for p in flu:
        case_type_map[p.name] = "flu_like"
    for p in diff_selected:
        sys = _get_differential_system(p, metadata_dir)
        case_type_map[p.name] = f"differential_{sys}" if sys else "differential"
    for p in healthy:
        case_type_map[p.name] = "healthy"

    return sample, case_type_map


# ─── Standalone verification ─────────────────────────────────────────────────

if __name__ == "__main__":
    sample, ct_map = select_benchmark_sample()

    # Count by case type
    counts: Dict[str, int] = {}
    for label in ct_map.values():
        counts[label] = counts.get(label, 0) + 1

    print(f"Benchmark sample: {len(sample)} transcripts  (seed={BENCHMARK_SEED})\n")
    print("Stratification:")
    for label in sorted(counts):
        print(f"  {label:40s} {counts[label]:>3}")

    print(f"\n{'─'*60}")
    for p in sample:
        print(f"  [{ct_map[p.name]:40s}] {p.name}")
