"""
Batch runner — generate 500 synthetic transcripts with specific
case-type distributions, each grounded in a unique Synthea CCDA.

Usage:
    python run_batch_500.py --test     # 8-9 sample transcripts for review
    python run_batch_500.py            # full 500 (resume-capable)

Distribution:
    Healthy/Preventative  252
    Flu                    35
    Novel Virus            50
    Differential: MSK      63
    Differential: GI       55
    Differential: Derm     29
    Differential: Neuro    16
    ──────────────────────────
    Total                 500
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ─── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)           # synthetic generation/
PROJECT_DIR = os.path.dirname(PARENT_DIR)           # Synthetic Transcript Generation/

CCDA_TO_USE = os.path.join(PROJECT_DIR, "Synthea CCDAs", "to use")
CCDA_USED   = os.path.join(PROJECT_DIR, "Synthea CCDAs", "used")

BATCH_DIR        = os.path.join(PROJECT_DIR, "Synthetic Transcripts", "Batch 500")
TRANSCRIPT_DIR   = os.path.join(BATCH_DIR, "transcripts")
METADATA_DIR     = os.path.join(BATCH_DIR, "metadata")
CSV_PATH         = os.path.join(BATCH_DIR, "transcript_index.csv")

# Ensure v3/ and parent are importable
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# ─── Distribution config ────────────────────────────────────────────────────
# Each entry: (case_type, differential_system, count)
DISTRIBUTION: List[Tuple[str, Optional[str], int]] = [
    ("healthy",      None,               251),
    ("flu_like",     None,                34),
    ("novel_virus",  None,                47),
    ("differential", "musculoskeletal",   62),
    ("differential", "gastrointestinal",  54),
    ("differential", "dermatological",    28),
    ("differential", "neurological",      15),
]

TOTAL = sum(c for _, _, c in DISTRIBUTION)  # 500

# Filename prefixes
_PREFIX = {
    "novel_virus":  "NOVEL",
    "flu_like":     "FLU",
    "differential": "DIFF",
    "healthy":      "HEALTHY",
}

# Rate-limiting defaults
BASE_DELAY_S   = 2.0     # seconds between API calls
MAX_RETRIES    = 5
BACKOFF_FACTOR = 2.0

# CSV column names
CSV_COLUMNS = [
    "index",
    "patient_name",
    "filename",
    "metadata_filename",
    "scenario_filename",
    "ccda_source",
    "case_type",
    "differential_subtype",
    "seed",
    "present_symptom_count",
    "negated_symptom_count",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "generation_time_seconds",
]

# ─── Test-mode distribution ─────────────────────────────────────────────────
TEST_DISTRIBUTION: List[Tuple[str, Optional[str], int]] = [
    ("healthy",      None,              1),
    ("flu_like",     None,              1),
    ("novel_virus",  None,              3),
    ("differential", "musculoskeletal", 1),
    ("differential", "gastrointestinal",1),
    ("differential", "dermatological",  1),
    ("differential", "neurological",    1),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_case_plan(distribution: List[Tuple[str, Optional[str], int]]) -> List[Dict[str, Any]]:
    """Expand distribution into a shuffled list of individual case dicts."""
    plan: List[Dict[str, Any]] = []
    for case_type, diff_system, count in distribution:
        for _ in range(count):
            plan.append({
                "case_type": case_type,
                "differential_system": diff_system,
                "seed": random.randint(1, 999_999),
            })
    random.shuffle(plan)
    return plan


def _label(case_type: str, diff_system: Optional[str]) -> str:
    """Human-readable label for terminal output."""
    if diff_system:
        return f"DIFF:{diff_system}"
    return case_type.upper()


def _load_completed(csv_path: str) -> set:
    """Return set of transcript filenames already in the CSV (for resume)."""
    completed: set = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["filename"])
    return completed


def _init_csv(csv_path: str) -> None:
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def _append_csv(csv_path: str, row: Dict[str, Any]) -> None:
    """Append one row to the CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def _rate_limit_sleep(delay: float) -> None:
    """Sleep for rate limiting."""
    if delay > 0:
        time.sleep(delay)


# ─── Core generation for one transcript ─────────────────────────────────────

def generate_one(
    ccda_path: str,
    case_type: str,
    seed: int,
    differential_system: Optional[str],
    transcript_dir: str,
    metadata_dir: str,
) -> Dict[str, Any]:
    """Generate a single transcript + metadata, return info dict.

    Directly calls the v3 pipeline components (parse CCDA, build scenario,
    generate transcript) without going through execute_pipeline(), giving
    full control over output paths and no viewer launch.
    """
    from ccda_to_ground_truth import parse_ccda
    from scenario_builder_v3 import build_scenario
    from generate_transcript_v3 import build_prompt, generate_transcript

    ccda_filename = os.path.basename(ccda_path)
    prefix = _PREFIX.get(case_type, "SYNTHETIC")

    # 1. Parse CCDA → ground truth JSON (write to temp location)
    gt = parse_ccda(ccda_path)
    gt_json_name = ccda_filename.replace(".xml", "_ground_truth.json")
    gt_path = os.path.join(PARENT_DIR, gt_json_name)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    # 2. Build scenario
    scenario = build_scenario(
        gt_path,
        seed=seed,
        case_type=case_type,
        hub_id=None,
        differential_system=differential_system,
    )

    patient_name = scenario["demographics"]["name"].replace(" ", "_")
    seed_tag = f"_s{seed}"

    # 3. Save scenario JSON
    scenario_filename = f"scenario_v3_{prefix}_{patient_name}{seed_tag}.json"
    scenario_path = os.path.join(metadata_dir, scenario_filename)
    with open(scenario_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)

    # 4. Generate transcript via LLM (this is the API call)
    system_prompt, user_prompt = build_prompt(scenario)
    transcript, token_usage = generate_transcript(system_prompt, user_prompt)

    # 5. Save transcript
    transcript_filename = f"{prefix}_SYNTHETIC_v3_{patient_name}{seed_tag}.txt"
    transcript_path = os.path.join(transcript_dir, transcript_filename)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # 6. Compute stats
    lines = transcript.strip().split("\n")
    d_lines = sum(1 for l in lines if l.startswith("D:"))
    p_lines = sum(1 for l in lines if l.startswith("P:"))

    # 7. Write metadata JSON
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_type": case_type,
        "seed": seed,
        "ccda_source": ccda_filename,
        "demographics": scenario["demographics"],
        "location": scenario.get("location"),
        "chief_complaint": scenario["chief_complaint"],
        "present_symptoms": scenario["present_symptoms"],
        "negated_symptoms": scenario["negated_symptoms"],
        "differential_system": scenario.get("differential_system"),
        "ccda_noise": {
            "conditions": scenario["ccda_noise"]["conditions"],
            "medications": scenario["ccda_noise"]["medications"],
            "allergies": scenario["ccda_noise"]["allergies"],
        },
        "transcript_stats": {
            "total_lines": len(lines),
            "doctor_lines": d_lines,
            "patient_lines": p_lines,
        },
        "transcript_file": transcript_filename,
        "scenario_file": scenario_filename,
    }

    metadata_filename = f"{prefix}_METADATA_v3_{patient_name}{seed_tag}.json"
    metadata_path = os.path.join(metadata_dir, metadata_filename)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 8. Clean up ground-truth temp file
    try:
        os.remove(gt_path)
    except OSError:
        pass

    return {
        "patient_name": scenario["demographics"]["name"],
        "filename": transcript_filename,
        "metadata_filename": metadata_filename,
        "scenario_filename": scenario_filename,
        "ccda_source": ccda_filename,
        "case_type": case_type,
        "differential_subtype": differential_system or "",
        "seed": seed,
        "present_symptom_count": len(scenario["present_symptoms"]),
        "negated_symptom_count": len(scenario["negated_symptoms"]),
        "present_symptoms": scenario["present_symptoms"],
        "negated_symptoms": scenario["negated_symptoms"],
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"],
    }


# ─── Main batch loop ────────────────────────────────────────────────────────

def run_batch(test_mode: bool = False) -> None:
    distribution = TEST_DISTRIBUTION if test_mode else DISTRIBUTION
    total = sum(c for _, _, c in distribution)
    mode_label = "TEST" if test_mode else "FULL"

    print("=" * 70)
    print(f"  Batch Transcript Generation — {mode_label} MODE ({total} transcripts)")
    print("=" * 70)

    # Show distribution
    print("\n  Distribution:")
    for case_type, diff_sys, count in distribution:
        label = _label(case_type, diff_sys)
        print(f"    {label:30s} {count:>4d}")
    print(f"    {'TOTAL':30s} {total:>4d}")

    # Setup directories
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    os.makedirs(CCDA_USED, exist_ok=True)

    # Load available CCDAs
    ccda_files = sorted(glob.glob(os.path.join(CCDA_TO_USE, "*.xml")))
    if len(ccda_files) < total:
        print(f"\n  ERROR: Need {total} CCDAs but only {len(ccda_files)} in {CCDA_TO_USE}")
        sys.exit(1)
    print(f"\n  CCDAs available: {len(ccda_files)}  (need {total})")

    # Shuffle CCDAs for diverse patient backgrounds
    random.shuffle(ccda_files)

    # Build case plan
    plan = _build_case_plan(distribution)
    print(f"  Case plan built: {len(plan)} cases (shuffled)\n")

    # Resume support
    _init_csv(CSV_PATH)
    completed = _load_completed(CSV_PATH)
    if completed:
        print(f"  Resuming: {len(completed)} already completed, {total - len(completed)} remaining\n")

    # ── Generation loop ──────────────────────────────────────────────
    ccda_idx = 0
    success_count = len(completed)
    error_count = 0
    start_time = time.time()

    for i, case in enumerate(plan):
        # Find next available CCDA (skip if already used in this run)
        while ccda_idx < len(ccda_files):
            ccda = ccda_files[ccda_idx]
            ccda_idx += 1
            # Check it hasn't been moved to used/ already
            if os.path.exists(ccda):
                break
        else:
            print(f"\n  ERROR: Ran out of CCDAs at case {i + 1}/{total}")
            break

        case_type = case["case_type"]
        diff_sys = case["differential_system"]
        seed = case["seed"]
        label = _label(case_type, diff_sys)

        # Progress header
        print(f"\n{'─' * 70}")
        print(f"  [{success_count + 1}/{total}]  {label}  |  seed={seed}  |  {os.path.basename(ccda)}")
        print(f"{'─' * 70}")

        # Generate with retries
        attempt = 0
        delay = BASE_DELAY_S
        gen_start = time.time()

        while attempt < MAX_RETRIES:
            try:
                result = generate_one(
                    ccda_path=ccda,
                    case_type=case_type,
                    seed=seed,
                    differential_system=diff_sys,
                    transcript_dir=TRANSCRIPT_DIR,
                    metadata_dir=METADATA_DIR,
                )
                break  # success
            except Exception as e:
                attempt += 1
                err_msg = str(e)
                # Check for rate limit (HTTP 429)
                if "429" in err_msg or "rate" in err_msg.lower():
                    wait = delay * (BACKOFF_FACTOR ** (attempt - 1))
                    print(f"  ⚠ Rate limited (attempt {attempt}/{MAX_RETRIES}). "
                          f"Waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    print(f"  ✗ Error (attempt {attempt}/{MAX_RETRIES}): {err_msg}")
                    if attempt >= MAX_RETRIES:
                        error_count += 1
                        print(f"  ✗ SKIPPING after {MAX_RETRIES} failures")
                        break
                    time.sleep(delay)
        else:
            # All retries exhausted
            error_count += 1
            continue

        gen_time = time.time() - gen_start

        # Log to CSV
        csv_row = {
            "index": success_count + 1,
            "patient_name": result["patient_name"],
            "filename": result["filename"],
            "metadata_filename": result["metadata_filename"],
            "scenario_filename": result["scenario_filename"],
            "ccda_source": result["ccda_source"],
            "case_type": result["case_type"],
            "differential_subtype": result["differential_subtype"],
            "seed": result["seed"],
            "present_symptom_count": result["present_symptom_count"],
            "negated_symptom_count": result["negated_symptom_count"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
            "generation_time_seconds": round(gen_time, 1),
        }
        _append_csv(CSV_PATH, csv_row)

        # Move CCDA to used/
        try:
            dest = os.path.join(CCDA_USED, os.path.basename(ccda))
            shutil.move(ccda, dest)
        except Exception as e:
            print(f"  ⚠ Could not move CCDA: {e}")

        success_count += 1

        # Progress summary
        elapsed = time.time() - start_time
        avg_time = elapsed / success_count if success_count else 0
        remaining_est = avg_time * (total - success_count)

        print(f"\n  ✓ {result['patient_name']}  |  {label}")
        print(f"    Present: {result['present_symptom_count']}  |  "
              f"Negated: {result['negated_symptom_count']}  |  "
              f"Time: {gen_time:.1f}s")
        print(f"    Progress: {success_count}/{total}  |  "
              f"Errors: {error_count}  |  "
              f"Elapsed: {elapsed / 60:.1f}m  |  "
              f"ETA: {remaining_est / 60:.1f}m")

        # Print symptoms for test mode
        if test_mode:
            print(f"\n    Present symptoms ({result['present_symptom_count']}):")
            for s in result["present_symptoms"]:
                print(f"      🟢 {s}")
            print(f"    Negated symptoms ({result['negated_symptom_count']}):")
            for s in result["negated_symptoms"]:
                print(f"      🔴 {s}")

        # Rate-limit sleep (skip after last)
        if success_count < total:
            _rate_limit_sleep(BASE_DELAY_S)

    # ── Final summary ────────────────────────────────────────────────
    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  BATCH COMPLETE — {mode_label} MODE")
    print("=" * 70)
    print(f"  Transcripts generated: {success_count}")
    print(f"  Errors / skipped:      {error_count}")
    print(f"  Total time:            {total_elapsed / 60:.1f} minutes")
    print(f"  CSV log:               {CSV_PATH}")
    print(f"  Transcripts:           {TRANSCRIPT_DIR}")
    print(f"  Metadata:              {METADATA_DIR}")
    print(f"  CCDAs remaining:       {len(glob.glob(os.path.join(CCDA_TO_USE, '*.xml')))}")
    print("=" * 70)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Batch generate 500 synthetic transcripts")
    parser.add_argument("--test", action="store_true",
                        help="Generate 8-9 test samples for review instead of full 500")
    args = parser.parse_args()

    run_batch(test_mode=args.test)


if __name__ == "__main__":
    main()
