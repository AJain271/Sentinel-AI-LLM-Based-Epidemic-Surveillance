"""
Load the 500 patient JSON files + pre-cached embeddings into a
simulation-ready DataFrame.  All heavy I/O is cached via Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    FEWSHOT_DIR,
    EMBEDDINGS_CACHE,
    OUTBREAK_START,
    INPUT_COST_PER_M,
    OUTPUT_COST_PER_M,
)


# ── helpers ─────────────────────────────────────────────────────────────────
def _day_number(ts_str: str) -> int:
    """Convert an ISO date string to a 1-based day index (Day 1 = Dec 26)."""
    start = datetime.strptime(OUTBREAK_START, "%Y-%m-%d")
    current = datetime.strptime(ts_str[:10], "%Y-%m-%d")
    return (current - start).days + 1


def _derive_case_type(filename: str) -> str:
    fn = filename.upper()
    if fn.startswith("NOVEL"):
        return "novel_virus"
    if fn.startswith("FLU"):
        return "flu_like"
    if fn.startswith("HEALTHY"):
        return "healthy"
    return "differential"


# ── main loader ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading patient data …")
def load_simulation_data() -> pd.DataFrame:
    """
    Parse every JSON in the fewshot results folder and return a flat
    DataFrame with one row per patient.
    """
    json_files = sorted(FEWSHOT_DIR.glob("*.json"))
    if not json_files:
        st.error(f"No JSON files found in {FEWSHOT_DIR}")
        st.stop()

    rows: list[dict] = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            rec = json.load(f)

        fname = rec.get("filename", jf.stem)
        hub   = rec.get("hospital_hub", {})
        unmapped = rec.get("unmapped_symptoms", [])

        # Extract anomaly term strings
        anomaly_terms = [
            s["term"] for s in unmapped
            if isinstance(s, dict) and s.get("status") == 1
        ]
        anomaly_quotes = [
            s.get("quote", "") for s in unmapped
            if isinstance(s, dict) and s.get("status") == 1
        ]

        prompt_tok = rec.get("prompt_tokens", 0)
        comp_tok   = rec.get("completion_tokens", 0)

        rows.append({
            "filename":        fname,
            "json_path":       str(jf),
            "case_type":       _derive_case_type(fname),
            "hub_id":          hub.get("hub_id", ""),
            "hospital":        hub.get("hospital", ""),
            "city":            hub.get("city", ""),
            "lat":             hub.get("lat", 0.0),
            "lng":             hub.get("lng", 0.0),
            "timestamp":       rec.get("timestamp", ""),
            "day":             _day_number(rec.get("timestamp", OUTBREAK_START)),
            "prompt_tokens":   prompt_tok,
            "completion_tokens": comp_tok,
            "total_tokens":    prompt_tok + comp_tok,
            "input_cost":      prompt_tok / 1_000_000 * INPUT_COST_PER_M,
            "output_cost":     comp_tok / 1_000_000 * OUTPUT_COST_PER_M,
            "total_cost":      (prompt_tok / 1_000_000 * INPUT_COST_PER_M
                                + comp_tok / 1_000_000 * OUTPUT_COST_PER_M),
            "unmapped_anomalies": anomaly_terms,
            "unmapped_quotes":    anomaly_quotes,
            "unmapped_text":   ", ".join(anomaly_terms) if anomaly_terms else "",
            "Novelty_Flag":    len(anomaly_terms) > 0,
            "novelty_count":   len(anomaly_terms),
        })

    df = pd.DataFrame(rows)
    return df


# ── embeddings loader ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading embeddings …")
def load_embeddings() -> np.ndarray:
    """
    Load pre-cached 768-dim nomic-embed-text embeddings.
    Row order matches sorted(glob("*.json")) from the fewshot dir.
    """
    if not EMBEDDINGS_CACHE.exists():
        st.error(f"Embeddings cache not found at {EMBEDDINGS_CACHE}")
        st.stop()
    emb = np.load(str(EMBEDDINGS_CACHE))
    return emb
