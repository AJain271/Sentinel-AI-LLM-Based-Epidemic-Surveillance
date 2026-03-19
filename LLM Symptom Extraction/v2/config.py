"""
Configuration for the v2 zero-shot LLM symptom extraction pipeline.
"""

import os
from pathlib import Path

# ─── OpenAI settings ─────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 4096

# ─── Paths ───────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # snhs symposium results

TRANSCRIPT_DIR = _ROOT / "Synthetic Transcript Generation" / "Initial Batch Test" / "transcripts"
METADATA_DIR   = _ROOT / "Synthetic Transcript Generation" / "Initial Batch Test" / "metadata"
OUTPUT_DIR     = Path(__file__).resolve().parent / "output"

# Symptom library source (for imports)
SYMPTOM_LIB_DIR = _ROOT / "Synthetic Transcript Generation" / "synthetic generation" / "v3"
