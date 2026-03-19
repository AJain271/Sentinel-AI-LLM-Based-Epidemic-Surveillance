"""
test10.py — Generate 10 synthetic transcripts using the v3 pipeline
and record timing / token metrics for projecting larger batch costs.

Breakdown:
    4 × novel_virus
    2 × flu_like
    2 × healthy
    2 × differential

Outputs:
    - Transcripts + metadata in  .../test10_output/
    - Summary report printed to console and saved as test10_report.json
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # synthetic generation/
CCDA_DIR = os.path.join(PARENT_DIR, "..", "Synthea CCDAs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "test10_output")

# Ensure v3 modules are importable
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from scenario_builder_v3 import build_scenario
from generate_transcript_v3 import build_prompt, generate_transcript, _get_client

# ─── Case plan ───────────────────────────────────────────────────────────────
# (case_type, seed) — seeds for reproducibility
CASE_PLAN = [
    ("novel_virus",  1001),
    ("novel_virus",  1002),
    ("novel_virus",  1003),
    ("novel_virus",  1004),
    ("flu_like",     2001),
    ("flu_like",     2002),
    ("healthy",      3001),
    ("healthy",      3002),
    ("differential", 4001),
    ("differential", 4002),
]

# ─── Prefix map ──────────────────────────────────────────────────────────────
_CASE_TYPE_PREFIX = {
    "novel_virus":  "NOVEL",
    "flu_like":     "FLU",
    "differential": "DIFF",
    "healthy":      "HEALTHY",
}


def run_single(ccda_path: str, case_type: str, seed: int, idx: int) -> dict:
    """Run the pipeline for a single case and return metrics."""
    from scenario_builder_v3 import build_scenario
    from generate_transcript_v3 import build_prompt, _get_client

    prefix = _CASE_TYPE_PREFIX.get(case_type, "SYNTHETIC")
    ccda_filename = os.path.basename(ccda_path)

    # ── Step 1: Parse CCDA ───────────────────────────────────────────
    t_parse_start = time.perf_counter()
    ccda_parser = os.path.join(PARENT_DIR, "ccda_to_ground_truth.py")
    import subprocess
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run([sys.executable, ccda_parser, ccda_path],
                   check=True, capture_output=True, env=env)
    t_parse = time.perf_counter() - t_parse_start

    # Resolve ground-truth path
    base_json_name = ccda_filename.replace(".xml", "_ground_truth.json")
    gt_path = os.path.join(PARENT_DIR, base_json_name)

    # ── Step 2: Build scenario ───────────────────────────────────────
    t_scenario_start = time.perf_counter()
    scenario = build_scenario(gt_path, seed=seed, case_type=case_type)
    t_scenario = time.perf_counter() - t_scenario_start

    patient_name = scenario["demographics"]["name"].replace(" ", "_")

    # ── Step 3: Generate transcript (timed) ──────────────────────────
    system_prompt, user_prompt = build_prompt(scenario)

    client = _get_client()
    t_api_start = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.85,
        max_tokens=6000,
    )
    t_api = time.perf_counter() - t_api_start

    transcript = response.choices[0].message.content.strip()
    usage = response.usage

    # ── Save outputs ─────────────────────────────────────────────────
    tag = f"{idx:02d}_{prefix}_{patient_name}_s{seed}"

    transcript_path = os.path.join(OUTPUT_DIR, f"{tag}.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    scenario_path = os.path.join(OUTPUT_DIR, f"{tag}_scenario.json")
    with open(scenario_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)

    # Transcript stats
    lines = transcript.strip().split("\n")
    d_lines = sum(1 for l in lines if l.startswith("D:"))
    p_lines = sum(1 for l in lines if l.startswith("P:"))

    return {
        "index": idx,
        "case_type": case_type,
        "seed": seed,
        "ccda": ccda_filename,
        "patient": scenario["demographics"]["name"],
        "differential_system": scenario.get("differential_system"),
        "n_present": len(scenario["present_symptoms"]),
        "n_negated": len(scenario["negated_symptoms"]),
        "total_lines": len(lines),
        "doctor_lines": d_lines,
        "patient_lines": p_lines,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "time_parse_s": round(t_parse, 2),
        "time_scenario_s": round(t_scenario, 4),
        "time_api_s": round(t_api, 2),
        "time_total_s": round(t_parse + t_scenario + t_api, 2),
        "transcript_file": os.path.basename(transcript_path),
    }


def main() -> None:
    print("=" * 70)
    print("  v3 Test Batch — 10 Samples")
    print("  4 novel_virus | 2 flu_like | 2 healthy | 2 differential")
    print("=" * 70)

    # Gather CCDAs (cycle through them for 10 cases)
    xml_files = sorted(glob.glob(os.path.join(CCDA_DIR, "*.xml")))
    if not xml_files:
        print(f"ERROR: No CCDA XML files found in {CCDA_DIR}")
        sys.exit(1)
    print(f"\nFound {len(xml_files)} CCDAs — will cycle through them.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    batch_start = time.perf_counter()

    for i, (case_type, seed) in enumerate(CASE_PLAN):
        ccda = xml_files[i % len(xml_files)]
        print(f"\n{'─' * 70}")
        print(f"  [{i+1}/10] {case_type} | seed={seed} | {os.path.basename(ccda)}")
        print(f"{'─' * 70}")

        try:
            metrics = run_single(ccda, case_type, seed, i + 1)
            results.append(metrics)
            print(f"  ✓ {metrics['patient']} — {metrics['total_lines']} lines, "
                  f"{metrics['total_tokens']} tokens, {metrics['time_api_s']}s API, "
                  f"{metrics['time_total_s']}s total")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({
                "index": i + 1,
                "case_type": case_type,
                "seed": seed,
                "error": str(e),
            })

    batch_elapsed = time.perf_counter() - batch_start

    # ─── Summary report ──────────────────────────────────────────────────────
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    if not successful:
        print("  All cases failed!")
        return

    # Per-case breakdown
    print(f"\n  {'#':<4} {'Type':<15} {'Patient':<25} {'Lines':<7} {'Tokens':<8} {'API(s)':<8} {'Total(s)':<8}")
    print(f"  {'─'*4} {'─'*15} {'─'*25} {'─'*7} {'─'*8} {'─'*8} {'─'*8}")
    for r in successful:
        sys_tag = f" [{r['differential_system']}]" if r.get("differential_system") else ""
        print(f"  {r['index']:<4} {r['case_type'] + sys_tag:<15} {r['patient']:<25} "
              f"{r['total_lines']:<7} {r['total_tokens']:<8} {r['time_api_s']:<8} {r['time_total_s']:<8}")

    # Aggregate stats
    total_tokens = sum(r["total_tokens"] for r in successful)
    total_prompt = sum(r["prompt_tokens"] for r in successful)
    total_completion = sum(r["completion_tokens"] for r in successful)
    total_api_time = sum(r["time_api_s"] for r in successful)
    total_time = batch_elapsed
    avg_api_time = total_api_time / len(successful)
    avg_total_time = total_time / len(successful)
    avg_tokens = total_tokens / len(successful)
    avg_lines = sum(r["total_lines"] for r in successful) / len(successful)

    # GPT-4o pricing (as of 2024): $2.50/1M input, $10.00/1M output
    input_cost = (total_prompt / 1_000_000) * 2.50
    output_cost = (total_completion / 1_000_000) * 10.00
    total_cost = input_cost + output_cost
    cost_per_sample = total_cost / len(successful)

    print(f"\n  Successful: {len(successful)}/10  |  Failed: {len(failed)}/10")
    print(f"\n  ── Timing ──")
    print(f"  Total batch time:     {total_time:.1f}s  ({total_time/60:.1f} min)")
    print(f"  Total API time:       {total_api_time:.1f}s")
    print(f"  Avg API time/sample:  {avg_api_time:.1f}s")
    print(f"  Avg total time/sample:{avg_total_time:.1f}s")

    print(f"\n  ── Tokens ──")
    print(f"  Total tokens:         {total_tokens:,}")
    print(f"  Total prompt tokens:  {total_prompt:,}")
    print(f"  Total completion:     {total_completion:,}")
    print(f"  Avg tokens/sample:    {avg_tokens:,.0f}")

    print(f"\n  ── Cost (GPT-4o pricing: $2.50/1M in, $10/1M out) ──")
    print(f"  Total cost (10):      ${total_cost:.4f}")
    print(f"  Cost per sample:      ${cost_per_sample:.4f}")

    print(f"\n  ── Transcript Quality ──")
    print(f"  Avg lines/transcript: {avg_lines:.0f}")
    avg_d = sum(r["doctor_lines"] for r in successful) / len(successful)
    avg_p = sum(r["patient_lines"] for r in successful) / len(successful)
    print(f"  Avg doctor lines:     {avg_d:.0f}")
    print(f"  Avg patient lines:    {avg_p:.0f}")

    # ─── Projections ─────────────────────────────────────────────────────
    print(f"\n  ── Projections ──")
    print(f"  {'Batch Size':<12} {'Est. Time':<20} {'Est. Cost':<15}")
    print(f"  {'─'*12} {'─'*20} {'─'*15}")
    for n in [300, 500, 1000]:
        proj_time = avg_total_time * n
        proj_cost = cost_per_sample * n
        hrs = proj_time / 3600
        mins = proj_time / 60
        if hrs >= 1:
            time_str = f"{hrs:.1f} hrs ({mins:.0f} min)"
        else:
            time_str = f"{mins:.1f} min"
        print(f"  {n:<12} {time_str:<20} ${proj_cost:.2f}")

    print(f"\n  NOTE: Projections assume sequential generation (no parallelism).")
    print(f"        With async/parallel calls, actual time could be much lower.")

    # ─── Save report JSON ────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_size": 10,
        "successful": len(successful),
        "failed": len(failed),
        "batch_elapsed_s": round(batch_elapsed, 2),
        "total_api_time_s": round(total_api_time, 2),
        "avg_api_time_s": round(avg_api_time, 2),
        "avg_total_time_s": round(avg_total_time, 2),
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "avg_tokens_per_sample": round(avg_tokens),
        "total_cost_usd": round(total_cost, 4),
        "cost_per_sample_usd": round(cost_per_sample, 4),
        "avg_lines_per_transcript": round(avg_lines),
        "projections": {
            str(n): {
                "est_time_min": round((avg_total_time * n) / 60, 1),
                "est_cost_usd": round(cost_per_sample * n, 2),
            }
            for n in [300, 500, 1000]
        },
        "per_case": results,
    }

    report_path = os.path.join(OUTPUT_DIR, "test10_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("  Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
