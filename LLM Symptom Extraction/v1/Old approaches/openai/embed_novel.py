"""
Novel symptom embedding + PCA reduction — ORIGINAL v1 (OpenAI / Ollama dual backend).

Archived copy of the original embedding pipeline.

Supports two backends:
  "openai" → text-embedding-3-small (1536-dim) via OpenAI API
  "ollama" → nomic-embed-text (768-dim) via local Ollama server

Takes a collection of novel symptom strings, embeds them, and
optionally reduces with PCA.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────

# Backend: "openai" or "ollama"
EMBED_BACKEND = "openai"

# OpenAI
OPENAI_EMBED_MODEL = "text-embedding-3-small"   # 1536-dim
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_EMBED_MODEL = "nomic-embed-text"          # 768-dim

PCA_DIMS = 8

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if EMBED_BACKEND == "ollama":
            _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        else:
            key = API_KEY or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError("No API key for embeddings.")
            _client = OpenAI(api_key=key)
    return _client


def _get_embed_model() -> str:
    return OLLAMA_EMBED_MODEL if EMBED_BACKEND == "ollama" else OPENAI_EMBED_MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of strings. Returns (N, D) array where D depends on model."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    client = _get_client()
    model = _get_embed_model()
    response = client.embeddings.create(model=model, input=texts)
    vecs = [item.embedding for item in response.data]
    return np.array(vecs, dtype=np.float32)


def pca_reduce(
    embeddings: np.ndarray,
    n_components: int = PCA_DIMS,
) -> np.ndarray:
    """Reduce embedding dimensionality with PCA.

    Uses a simple numpy implementation (no sklearn dependency).
    If there are fewer samples than n_components, returns as many
    components as possible.
    """
    if embeddings.shape[0] == 0:
        return np.empty((0, n_components), dtype=np.float32)

    n = min(n_components, embeddings.shape[0], embeddings.shape[1])

    # Center
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # Covariance via SVD (more numerically stable for wide matrices)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n]  # (n, D)

    reduced = centered @ components.T  # (N, n)

    # Pad with zeros if fewer components than requested
    if n < n_components:
        pad = np.zeros((reduced.shape[0], n_components - n), dtype=np.float32)
        reduced = np.hstack([reduced, pad])

    return reduced.astype(np.float32)


def embed_and_reduce(
    novel_symptoms: List[Dict[str, Any]],
    n_components: int = PCA_DIMS,
) -> List[Dict[str, Any]]:
    """Embed novel symptom strings and attach raw + PCA vectors.

    Parameters
    ----------
    novel_symptoms : list of dicts
        Each must have a "symptom" key (the text to embed).
    n_components : int
        PCA target dimensionality.

    Returns
    -------
    list of dicts — same items enriched with:
        embedding_raw : list[float]   (1536-dim)
        embedding_pca : list[float]   (n_components-dim)
    """
    if not novel_symptoms:
        return []

    texts = [ns["symptom"] for ns in novel_symptoms]
    raw = embed_texts(texts)
    reduced = pca_reduce(raw, n_components=n_components)

    enriched = []
    for i, ns in enumerate(novel_symptoms):
        enriched.append({
            **ns,
            "embedding_raw": raw[i].tolist(),
            "embedding_pca": reduced[i].tolist(),
        })
    return enriched


# ─── CLI for quick testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed novel symptom strings (ORIGINAL v1)")
    parser.add_argument("symptoms", nargs="+", help="Novel symptom strings to embed")
    parser.add_argument("--dims", type=int, default=PCA_DIMS, help="PCA dimensions")
    args = parser.parse_args()

    items = [{"symptom": s, "evidence": "", "score": 2} for s in args.symptoms]
    results = embed_and_reduce(items, n_components=args.dims)

    for r in results:
        print(f"\n{r['symptom']}:")
        print(f"  raw dims:  {len(r['embedding_raw'])}")
        print(f"  pca dims:  {len(r['embedding_pca'])}")
        print(f"  pca vals:  {r['embedding_pca']}")
