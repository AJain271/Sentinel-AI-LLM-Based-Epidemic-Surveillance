"""
Configuration for the v3 LLM symptom extraction pipeline.

Changes from v2:
- Points to Batch 500 data (not Initial Batch Test)
- API key from env var only (no hardcoded keys)
- Dev-sample controls for initial iteration
"""

import os
from pathlib import Path

# ─── OpenAI settings ─────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "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")
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 4096

# ─── Paths ───────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # snhs symposium results

BATCH_500_DIR = _ROOT / "Synthetic Transcript Generation" / "Synthetic Transcripts" / "Batch 500"
TRANSCRIPT_DIR = BATCH_500_DIR / "transcripts"
METADATA_DIR = BATCH_500_DIR / "metadata"

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Symptom library source (for imports)
SYMPTOM_LIB_DIR = _ROOT / "Synthetic Transcript Generation" / "synthetic generation" / "v3"

# Few-shot examples directory
FEW_SHOT_DIR = Path(__file__).resolve().parent / "few_shot_examples"

# ─── Dev-sample settings ─────────────────────────────────────────────────────
DEV_SAMPLE_SIZE = 5
DEV_SAMPLE_SEED = 42

# ─── Benchmark settings ──────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BENCHMARK_SEED = 2026
NUM_TRIALS = 3
