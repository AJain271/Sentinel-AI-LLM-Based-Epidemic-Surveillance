"""
UMAP + HDBSCAN Clustering Engine  (Dashboard Edition)
======================================================
Uses the validated UMAP + HDBSCAN (EOM) pipeline from
``Clustering/USED ALGORITHM/unsupervised_umap_hdbscan.py``
adapted for a **cumulative window** — for day *d* the engine
clusters all patients from Day 1 through Day *d*.

Pipeline per window
───────────────────
1. One-hot encode 70 symptoms × 3 classes (-1, 0, 1) → 210-dim
2. Load ClinicalBERT embeddings (768-dim, pre-cached)
3. Z-score normalise both matrices independently
4. Concatenate → (N, 978)
5. UMAP 10D reduction
6. HDBSCAN (EOM) clustering
7. UMAP 2D projection for visualisation
8. Outbreak detection — ≥50 % of cluster has novel symptoms present
"""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import FEWSHOT_DIR, CLUSTERING_CACHE

# ── Hyper-parameters (matching USED ALGORITHM) ──────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 10
UMAP_MIN_DIST = 0.0
UMAP_2D_MIN_DIST = 0.5
UMAP_2D_SPREAD = 2.0

NOVELTY_OUTBREAK_THRESHOLD = 0.50   # ≥ 50 % have novel symptoms
RANDOM_STATE = 42
WINDOW_SIZE = 3


# ── One-hot symptom matrix builder ──────────────────────────────────────────
@st.cache_data(show_spinner="Building one-hot symptom matrix …")
def build_onehot_matrix(_json_dir: str) -> tuple[np.ndarray, list[str]]:
    """
    Build (N, 210) one-hot matrix  — 70 symptoms × 3 classes (-1, 0, 1).
    Row order matches ``sorted(glob(*.json))``.
    """
    data_dir = Path(_json_dir)
    json_files = sorted(data_dir.glob("*.json"))
    all_scores: list[dict] = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            rec = json.load(f)
        all_scores.append(rec["checklist_scores"])

    symptom_names = sorted(all_scores[0].keys())
    n = len(all_scores)
    n_symptoms = len(symptom_names)
    onehot = np.zeros((n, n_symptoms * 3), dtype=np.float64)

    class_map = {-1: 0, 0: 1, 1: 2}
    for i, scores in enumerate(all_scores):
        for j, s in enumerate(symptom_names):
            val = scores.get(s, 0)
            col = j * 3 + class_map[val]
            onehot[i, col] = 1.0

    return onehot, symptom_names


@st.cache_data(show_spinner="Building categorical symptom matrix …")
def build_categorical_matrix(_json_dir: str) -> np.ndarray:
    """(N, 70) raw symptom scores for the symptom-analysis tab."""
    data_dir = Path(_json_dir)
    json_files = sorted(data_dir.glob("*.json"))
    all_scores: list[dict] = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            rec = json.load(f)
        all_scores.append(rec["checklist_scores"])

    symptom_names = sorted(all_scores[0].keys())
    n = len(all_scores)
    A = np.zeros((n, len(symptom_names)), dtype=np.int8)
    for i, scores in enumerate(all_scores):
        for j, s in enumerate(symptom_names):
            A[i, j] = scores.get(s, 0)
    return A


@st.cache_data(show_spinner=False)
def get_symptom_names(_json_dir: str) -> list[str]:
    """Return the sorted list of 70 symptom names from the first JSON."""
    data_dir = Path(_json_dir)
    jf = sorted(data_dir.glob("*.json"))[0]
    with open(jf, encoding="utf-8") as f:
        rec = json.load(f)
    return sorted(rec["checklist_scores"].keys())


# ── detail type builder (for cluster naming) ────────────────────────────────
@st.cache_data(show_spinner=False)
def _build_detail_types(_json_dir: str) -> list[str]:
    """Fine-grained labels: differential patients split into subtypes."""
    import csv
    data_dir = Path(_json_dir)
    workspace = data_dir.parent.parent

    master_csv = workspace / "LLM Symptom Extraction Full Run" / "results" / "master_results.csv"
    diff_map: dict[str, str] = {}
    if master_csv.exists():
        with open(master_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ds = row.get("differential_system", "").strip()
                if ds:
                    diff_map[row["filename"].strip()] = ds

    json_files = sorted(data_dir.glob("*.json"))
    detail: list[str] = []
    for jf in json_files:
        with open(jf, encoding="utf-8") as f:
            rec = json.load(f)
        fname = rec.get("filename", jf.stem)
        fn = fname.upper()
        if fn.startswith("NOVEL"):
            ct = "novel_virus"
        elif fn.startswith("FLU"):
            ct = "flu_like"
        elif fn.startswith("HEALTHY"):
            ct = "healthy"
        else:
            ct = "differential"

        if ct == "differential":
            detail.append(diff_map.get(fname, "differential"))
        else:
            detail.append(ct)
    return detail


# ── Build novelty flags per patient (for outbreak detection) ────────────────
@st.cache_data(show_spinner=False)
def _build_novelty_flags(_json_dir: str) -> np.ndarray:
    """Return boolean array: True if patient has non-blank unmapped_symptoms."""
    data_dir = Path(_json_dir)
    json_files = sorted(data_dir.glob("*.json"))
    flags = np.zeros(len(json_files), dtype=bool)
    for i, jf in enumerate(json_files):
        with open(jf, encoding="utf-8") as f:
            rec = json.load(f)
        unmapped = rec.get("unmapped_symptoms", [])
        has_novel = any(
            isinstance(s, dict) and s.get("status") == 1
            for s in unmapped
        )
        flags[i] = has_novel
    return flags


# ── UMAP + HDBSCAN core ────────────────────────────────────────────────────
def _run_umap_hdbscan(
    A_onehot: np.ndarray,
    B_semantic: np.ndarray,
    detail_types: np.ndarray,
    novelty_flags: np.ndarray,
) -> dict:
    """
    Run the full UMAP + HDBSCAN pipeline on the provided subset.

    Returns dict with: labels, label_names, coords_2d, silhouette,
                       n_clusters, n_noise, outbreak_label
    """
    import umap

    n = len(A_onehot)

    if n < 6:
        return {
            "labels": np.zeros(n, dtype=int),
            "label_names": {0: "mixed"},
            "coords_2d": np.zeros((n, 2)),
            "silhouette": None,
            "n_clusters": 1,
            "n_noise": 0,
            "outbreak_label": None,
        }

    # Z-score normalise independently
    scaler_oh = StandardScaler()
    A_norm = scaler_oh.fit_transform(A_onehot)

    scaler_sem = StandardScaler()
    B_norm = scaler_sem.fit_transform(B_semantic)

    # Concatenate → (N, 978)
    X = np.hstack([A_norm, B_norm])

    # UMAP 10D
    reducer = umap.UMAP(
        n_neighbors=min(UMAP_N_NEIGHBORS, n - 1),
        n_components=min(UMAP_N_COMPONENTS, n - 1),
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    X_umap = reducer.fit_transform(X)

    # UMAP 2D projection
    reducer_2d = umap.UMAP(
        n_neighbors=min(UMAP_N_NEIGHBORS, n - 1),
        n_components=2,
        min_dist=UMAP_2D_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
        spread=UMAP_2D_SPREAD,
    )
    coords_2d = reducer_2d.fit_transform(X_umap)

    # HDBSCAN (EOM) — FIXED parameters matching reference implementation
    # (unsupervised_umap_hdbscan.py uses fixed 15 / 5, no dynamic scaling)
    if n < HDBSCAN_MIN_CLUSTER_SIZE:
        # Too few points for meaningful HDBSCAN — everything would be noise
        labels = np.zeros(n, dtype=int)
        label_names: dict[int, str] = {0: "mixed"}
        return {
            "labels": labels,
            "label_names": label_names,
            "coords_2d": coords_2d,
            "silhouette": None,
            "n_clusters": 1,
            "n_noise": 0,
            "outbreak_label": None,
        }
    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_umap)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    sil = None
    if n_clusters >= 2 and (labels != -1).sum() > n_clusters:
        try:
            sil = float(silhouette_score(X_umap[labels != -1], labels[labels != -1]))
        except ValueError:
            pass

    # Label naming (by dominant detail type)
    label_names: dict[int, str] = {-1: "noise"}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        comps = Counter(detail_types[mask])
        dominant = comps.most_common(1)[0][0]
        label_names[cid] = dominant

    # Outbreak detection: ≥50% of cluster has novel symptoms present
    outbreak_label = _identify_outbreak(labels, novelty_flags)

    return {
        "labels": labels,
        "label_names": label_names,
        "coords_2d": coords_2d,
        "silhouette": sil,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "outbreak_label": outbreak_label,
    }


def _identify_outbreak(
    labels: np.ndarray,
    novelty_flags: np.ndarray,
) -> int | None:
    """Cluster where ≥50% of patients have novel symptoms present
    (non-blank unmapped_symptoms from LLM extraction)."""
    best_label, best_frac = None, 0.0
    for lbl in set(labels):
        if lbl == -1:
            continue
        mask = labels == lbl
        n_in_cluster = mask.sum()
        if n_in_cluster < 3:
            continue
        novel_frac = novelty_flags[mask].sum() / n_in_cluster
        if novel_frac > best_frac:
            best_frac = novel_frac
            best_label = lbl
    return best_label if best_frac >= NOVELTY_OUTBREAK_THRESHOLD else None


# ── public API ──────────────────────────────────────────────────────────────
def run_window_clustering(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    onehot: np.ndarray,
    detail_types: np.ndarray,
    novelty_flags: np.ndarray,
    current_day: int,
) -> dict:
    """
    UMAP + HDBSCAN on a cumulative window of patients (Day 1 → current_day).

    Returns dict:
        idx, labels, label_names, umap_2d, silhouette,
        n_clusters, n_noise, outbreak_label, window_start, window_end
    """
    window_start = 1
    window_end = current_day
    days = df["day"].values
    mask = (days >= window_start) & (days <= window_end)
    idx = np.where(mask)[0]

    A_sub = onehot[idx]
    B_sub = embeddings[idx]
    dt_sub = detail_types[idx]
    nf_sub = novelty_flags[idx]

    result = _run_umap_hdbscan(A_sub, B_sub, dt_sub, nf_sub)

    return {
        "idx": idx,
        "labels": result["labels"],
        "label_names": result["label_names"],
        "umap_2d": result["coords_2d"],
        "silhouette": result["silhouette"],
        "n_clusters": result["n_clusters"],
        "n_noise": result["n_noise"],
        "outbreak_label": result["outbreak_label"],
        "window_start": window_start,
        "window_end": window_end,
    }


# ── Data hash for cache staleness detection ─────────────────────────────────
def _compute_data_hash(df_json: str, emb_bytes: bytes, oh_bytes: bytes) -> str:
    h = hashlib.md5()
    h.update(df_json[:1000].encode())
    h.update(emb_bytes[:4096])
    h.update(oh_bytes[:4096])
    return h.hexdigest()


# ── pre-compute all days (with disk caching) ────────────────────────────────
@st.cache_data(show_spinner="Running UMAP + HDBSCAN clustering …")
def precompute_all_days(
    _df_json: str,
    _emb_bytes: bytes,
    _oh_bytes: bytes,
    _oh_shape: tuple[int, int],
    _detail_json: str,
    _novelty_json: str,
    num_days: int = 7,
) -> dict:
    """
    Pre-compute UMAP + HDBSCAN clustering for Days 1–7.
    Results are also written to disk so subsequent app starts are instant.
    """
    import io

    data_hash = _compute_data_hash(_df_json, _emb_bytes, _oh_bytes)
    cache_path = Path(CLUSTERING_CACHE)

    # Try loading from disk cache
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if cached.get("_hash") == data_hash:
                results = {k: v for k, v in cached.items() if k != "_hash"}
                return results
        except Exception:
            pass  # stale or corrupt — recompute

    df = pd.read_json(io.StringIO(_df_json))
    emb = np.frombuffer(_emb_bytes, dtype=np.float64).reshape(-1, 768)
    oh = np.frombuffer(_oh_bytes, dtype=np.float64).reshape(_oh_shape)
    detail = np.array(json.loads(_detail_json))
    novelty = np.array(json.loads(_novelty_json), dtype=bool)

    results: dict[int, dict] = {}
    for day in range(1, num_days + 1):
        results[day] = run_window_clustering(df, emb, oh, detail, novelty, day)

    # Persist to disk
    to_cache = dict(results)
    to_cache["_hash"] = data_hash
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(to_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass  # non-fatal

    return results
