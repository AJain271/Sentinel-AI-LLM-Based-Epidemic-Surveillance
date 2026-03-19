"""
Assign outbreak-simulation metadata to the 500 extraction results.

Run this AFTER run_full_extraction.py has completed.  It augments each
per-patient JSON with two new keys:

  • hospital_hub  — one of the 7 SC Hospital Hubs (with jittered coords)
  • timestamp     — one of the 7 outbreak days (Dec 26 2025 → Jan 1 2026)

Hub assignment:
  - 50 novel-virus cases  →  H1 (Anderson) + H2 (Greenville) only
  - 450 non-novel cases   →  all 7 hubs, uniform distribution

Timestamp assignment (steeper acceleration for the novel outbreak):
  Day 1 (Dec 26): 1    Day 5 (Dec 30): 8
  Day 2 (Dec 27): 2    Day 6 (Dec 31): 12
  Day 3 (Dec 28): 4    Day 7 (Jan  1): 17
  Day 4 (Dec 29): 6                   ──── total = 50

Non-novel cases are spread uniformly across all 7 days.

Also updates master_results.csv with hub_id, hospital, lat, lng, timestamp.

Usage:
    python assign_outbreak_metadata.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

# ─── Path setup ──────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent

# Import hospital-hub helpers from the synthetic-generation codebase
_HUB_DIR = _ROOT / "Synthetic Transcript Generation" / "synthetic generation" / "v3"
sys.path.insert(0, str(_HUB_DIR))
from sc_hospital_hubs import assign_location, SC_HOSPITAL_HUBS  # noqa: E402

# ─── Paths ───────────────────────────────────────────────────────────────────
RESULTS_DIR  = _THIS_DIR / "results"
FEWSHOT_DIR  = RESULTS_DIR / "fewshot"
MASTER_CSV   = RESULTS_DIR / "master_results.csv"

_INDEX_CSV = (
    _ROOT
    / "Synthetic Transcript Generation"
    / "Synthetic Transcripts"
    / "Batch 500"
    / "transcript_index.csv"
)

# ─── Constants ───────────────────────────────────────────────────────────────
SEED = 2026
OUTBREAK_START = date(2025, 12, 26)       # Day 1
OUTBREAK_DAYS = 7
NOVEL_DAY_COUNTS = [1, 2, 4, 6, 8, 12, 17]   # steeper acceleration, sum=50
UPSTATE_HUBS = ["H1", "H2"]                    # Anderson + Greenville


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_case_type_map() -> Dict[str, str]:
    """Return {transcript_filename: case_type}."""
    mapping: Dict[str, str] = {}
    with open(_INDEX_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["filename"]] = row["case_type"]
    return mapping


def _assign_days(items: List[str], day_counts: List[int], rng: random.Random) -> Dict[str, date]:
    """Assign each item to a day based on the specified count distribution.

    Returns {item_name: date}.
    """
    shuffled = list(items)
    rng.shuffle(shuffled)

    assignment: Dict[str, date] = {}
    idx = 0
    for day_offset, count in enumerate(day_counts):
        d = OUTBREAK_START + timedelta(days=day_offset)
        for _ in range(count):
            if idx < len(shuffled):
                assignment[shuffled[idx]] = d
                idx += 1
    return assignment


def _assign_days_uniform(items: List[str], num_days: int, rng: random.Random) -> Dict[str, date]:
    """Spread items uniformly across num_days starting at OUTBREAK_START."""
    shuffled = list(items)
    rng.shuffle(shuffled)

    assignment: Dict[str, date] = {}
    for i, item in enumerate(shuffled):
        day_offset = i % num_days
        assignment[item] = OUTBREAK_START + timedelta(days=day_offset)
    return assignment


# ─── Main ────────────────────────────────────────────────────────────────────

def assign_metadata() -> None:
    case_type_map = _load_case_type_map()
    rng = random.Random(SEED)

    json_files = sorted(FEWSHOT_DIR.glob("*.json"))
    if not json_files:
        print("No JSON result files found. Run run_full_extraction.py first.")
        return

    # Partition filenames by novel vs non-novel
    # JSON filenames use the transcript name with .json extension
    novel_files: List[Path] = []
    non_novel_files: List[Path] = []

    for jf in json_files:
        # Map JSON filename back to transcript filename (.txt)
        txt_name = jf.stem + ".txt"
        ct = case_type_map.get(txt_name, "")
        if ct == "novel_virus":
            novel_files.append(jf)
        else:
            non_novel_files.append(jf)

    print(f"\n{'═' * 70}")
    print("  OUTBREAK METADATA ASSIGNMENT")
    print(f"{'═' * 70}")
    print(f"  Novel cases  : {len(novel_files)}  → upstate hubs (H1 + H2)")
    print(f"  Non-novel    : {len(non_novel_files)}  → all 7 hubs (uniform)")
    print(f"  Outbreak days: {OUTBREAK_START.isoformat()} → "
          f"{(OUTBREAK_START + timedelta(days=OUTBREAK_DAYS - 1)).isoformat()}")
    print(f"  Novel day dist: {NOVEL_DAY_COUNTS}  (sum={sum(NOVEL_DAY_COUNTS)})")
    print()

    # ── Assign timestamps ─────────────────────────────────────────────────
    novel_names  = [f.name for f in novel_files]
    non_novel_names = [f.name for f in non_novel_files]

    novel_dates   = _assign_days(novel_names, NOVEL_DAY_COUNTS, rng)
    non_novel_dates = _assign_days_uniform(non_novel_names, OUTBREAK_DAYS, rng)

    all_dates = {**novel_dates, **non_novel_dates}

    # ── Assign hospital hubs ──────────────────────────────────────────────
    hub_assignments: Dict[str, Dict[str, Any]] = {}

    # Day 6 & 7 novel spread: 1 at H3 on day 6, 2 at H3 + 1 at H4 on day 7
    DAY6 = OUTBREAK_START + timedelta(days=5)   # Dec 31
    DAY7 = OUTBREAK_START + timedelta(days=6)   # Jan 1
    _spread_day6_h3 = 1
    _spread_day7_h3 = 2
    _spread_day7_h4 = 1

    for jf in novel_files:
        d = all_dates[jf.name]
        if d == DAY6 and _spread_day6_h3 > 0:
            hub_id = "H3"
            _spread_day6_h3 -= 1
        elif d == DAY7 and _spread_day7_h3 > 0:
            hub_id = "H3"
            _spread_day7_h3 -= 1
        elif d == DAY7 and _spread_day7_h4 > 0:
            hub_id = "H4"
            _spread_day7_h4 -= 1
        else:
            hub_id = rng.choice(UPSTATE_HUBS)
        loc = assign_location(rng, hub_id=hub_id, radius_km=25.0)
        hub_assignments[jf.name] = loc

    for jf in non_novel_files:
        loc = assign_location(rng, hub_id=None, radius_km=25.0)
        hub_assignments[jf.name] = loc

    # ── Augment each JSON file ────────────────────────────────────────────
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        data["hospital_hub"] = hub_assignments[jf.name]
        data["timestamp"] = all_dates[jf.name].isoformat()
        jf.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── Update master CSV ─────────────────────────────────────────────────
    if MASTER_CSV.exists():
        rows: List[Dict[str, Any]] = []
        with open(MASTER_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            original_columns = list(reader.fieldnames or [])
            for row in reader:
                rows.append(dict(row))

        # Add new columns to each row
        new_columns = ["hub_id", "hospital", "lat", "lng", "timestamp"]
        all_columns = original_columns + new_columns

        for row in rows:
            fn_json = Path(row["filename"]).with_suffix(".json").name
            hub = hub_assignments.get(fn_json, {})
            row["hub_id"]    = hub.get("hub_id", "")
            row["hospital"]  = hub.get("hospital", "")
            row["lat"]       = hub.get("lat", "")
            row["lng"]       = hub.get("lng", "")
            row["timestamp"] = all_dates.get(fn_json, "")

        with open(MASTER_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        print(f"  master_results.csv updated with {len(new_columns)} new columns")

    # ── Summary ───────────────────────────────────────────────────────────
    # Hub distribution
    hub_counts: Dict[str, int] = {}
    for loc in hub_assignments.values():
        hid = loc["hub_id"]
        hub_counts[hid] = hub_counts.get(hid, 0) + 1

    print(f"\n  Hub distribution:")
    for hub in SC_HOSPITAL_HUBS:
        hid = hub["hub_id"]
        cnt = hub_counts.get(hid, 0)
        print(f"    {hid}  {hub['name']:35s}  {cnt:>4} patients")

    # Day distribution for novel cases
    novel_day_counts: Dict[str, int] = {}
    for fn, d in novel_dates.items():
        ds = d.isoformat()
        novel_day_counts[ds] = novel_day_counts.get(ds, 0) + 1

    print(f"\n  Novel-virus cases per day:")
    for d in sorted(novel_day_counts):
        print(f"    {d}  {novel_day_counts[d]:>3} cases")

    # Day distribution for all cases
    all_day_counts: Dict[str, int] = {}
    for fn, d in all_dates.items():
        ds = d.isoformat()
        all_day_counts[ds] = all_day_counts.get(ds, 0) + 1

    print(f"\n  All cases per day:")
    for d in sorted(all_day_counts):
        print(f"    {d}  {all_day_counts[d]:>3} cases")

    print(f"\n  {len(json_files)} JSON files augmented with hospital_hub + timestamp")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    assign_metadata()
