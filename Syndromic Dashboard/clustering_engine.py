"""
Two-Stage HDBSCAN Clustering Engine  (Dashboard Edition)
=========================================================
Mirrors the methodology of ``Clustering/clustering_pipeline.py``
but uses a **rolling window** — for day *d* the engine clusters
patients from a 3-day window: Days max(1, d-2) … d.

Pipeline per window
───────────────────
1.  Build weighted distance matrix  D = w_cat * D_cat + w_sem * D_sem
2.  Stage 1  — HDBSCAN on D to identify healthy-dominated clusters
                 (≥ 70 % healthy ⇒ Cluster 0)
3.  Stage 2  — HDBSCAN on the non-healthy subset (fresh D_nh)
                 to isolate novel-virus, flu-like, and differential subtypes
4.  Merge labels  (healthy → 0, Stage 2 IDs offset +1, noise → -1)
5.  Noise recovery — assign each noise point to nearest cluster
6.  Name clusters by dominant case/detail type
7.  t-SNE on final D for visualisation
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import hdbscan

from config import FEWSHOT_DIR

# ── optimised hyper-parameters (from clustering_pipeline sweep) ─────────────
W_CAT = 0.78
W_SEM = 0.22

S1_MCS = 5          # Stage-1 min_cluster_size
S1_MS  = None       # Stage-1 min_samples ("auto" in the pipeline)
S2_MCS = 5          # Stage-2 min_cluster_size
S2_MS  = 3          # Stage-2 min_samples

HEALTHY_THRESHOLD = 0.70   # ≥ 70 % healthy → healthy cluster
RANDOM_STATE      = 42
WINDOW_SIZE       = 3      # rolling window width (days)


# ── categorical matrix builder ──────────────────────────────────────────────
@st.cache_data(show_spinner="Building categorical symptom matrix …")
def build_categorical_matrix(_json_dir: str) -> np.ndarray:
    """
    Build (N, 70) binary symptom matrix from ``checklist_scores``
    in each JSON.  Row order matches ``sorted(glob(*.json))``.
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
    """
    Return fine-grained labels aligned with sorted JSON order.
    Differential patients are split into subtypes via master_results.csv.
    """
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

        # derive case type
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


# ── distance matrix ─────────────────────────────────────────────────────────
def _compute_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Weighted Gower distance: w_cat * L1(A) + w_sem * cosine(B)."""
    n = A.shape[0]
    if n < 2:
        return np.zeros((1, 1))

    # categorical L1
    D_cat = squareform(pdist(A.astype(np.float64), metric="cityblock"))
    cat_max = D_cat.max()
    if cat_max > 0:
        D_cat /= cat_max

    # semantic cosine
    has_emb = np.linalg.norm(B, axis=1) > 0
    emb_idx = np.where(has_emb)[0]
    no_emb  = np.where(~has_emb)[0]
    D_sem   = np.zeros((n, n), dtype=np.float64)

    if len(emb_idx) > 1:
        B_sub = B[emb_idx]
        d_sub = squareform(pdist(B_sub, metric="cosine"))
        for ii, i in enumerate(emb_idx):
            for jj, j in enumerate(emb_idx):
                D_sem[i, j] = d_sub[ii, jj]
    for i in no_emb:
        for j in emb_idx:
            D_sem[i, j] = 1.0
            D_sem[j, i] = 1.0

    sem_max = D_sem.max()
    if sem_max > 0:
        D_sem /= sem_max

    D = W_CAT * D_cat + W_SEM * D_sem
    np.fill_diagonal(D, 0.0)
    return D


# ── t-SNE ───────────────────────────────────────────────────────────────────
def _run_tsne(D: np.ndarray) -> np.ndarray:
    """2-D projection from precomputed distance matrix."""
    n = D.shape[0]
    if n < 2:
        return np.zeros((n, 2))
    perp = min(30, max(2, n // 3))
    tsne = TSNE(
        n_components=2,
        metric="precomputed",
        perplexity=perp,
        random_state=RANDOM_STATE,
        init="random",
    )
    return tsne.fit_transform(D)


# ── two-stage HDBSCAN core ─────────────────────────────────────────────────
def _two_stage_hdbscan(
    A: np.ndarray,
    B: np.ndarray,
    case_types: np.ndarray,
    detail_types: np.ndarray,
) -> tuple[np.ndarray, dict[int, str], np.ndarray]:
    """
    Run the full two-stage pipeline on the provided subset.

    Returns
    -------
    labels       – final cluster label per patient
    label_names  – {cluster_id: cluster_name}
    D            – distance matrix (for t-SNE)
    """
    n = len(A)

    # ── too few patients → single cluster ──────────────────────────────
    if n < 6:
        labels = np.zeros(n, dtype=int)
        D = _compute_distance_matrix(A, B)
        return labels, {0: "mixed"}, D

    # ── full distance matrix ───────────────────────────────────────────
    D_full = _compute_distance_matrix(A, B)
    healthy_mask = (case_types == "healthy")

    # ═════════════════ STAGE 1: healthy vs non-healthy ═════════════════
    kw1 = dict(min_cluster_size=min(S1_MCS, max(2, n // 5)),
               metric="precomputed", cluster_selection_method="eom")
    if S1_MS is not None:
        kw1["min_samples"] = S1_MS
    s1_labels = hdbscan.HDBSCAN(**kw1).fit_predict(D_full.copy())

    # identify healthy-dominated clusters (≥ 70 % healthy)
    healthy_cluster_ids: set[int] = set()
    for cid in set(s1_labels):
        if cid == -1:
            continue
        mask = s1_labels == cid
        h_frac = healthy_mask[mask].sum() / mask.sum()
        if h_frac >= HEALTHY_THRESHOLD:
            healthy_cluster_ids.add(cid)

    # separate indices
    healthy_idx: list[int] = []
    nonhealthy_idx: list[int] = []
    for i in range(n):
        if s1_labels[i] in healthy_cluster_ids:
            healthy_idx.append(i)
        else:
            nonhealthy_idx.append(i)

    # ═════════════════ STAGE 2: sub-cluster non-healthy ════════════════
    if len(nonhealthy_idx) < 3:
        # too few non-healthy patients — everything healthy
        labels = np.zeros(n, dtype=int)
        return labels, {0: "healthy"}, D_full

    nh = np.array(nonhealthy_idx)
    A_nh, B_nh = A[nh], B[nh]
    D_nh = _compute_distance_matrix(A_nh, B_nh)

    kw2 = dict(min_cluster_size=min(S2_MCS, max(2, len(nh) // 5)),
               metric="precomputed", cluster_selection_method="eom")
    if S2_MS is not None:
        kw2["min_samples"] = min(S2_MS, max(1, len(nh) // 5))
    s2_labels = hdbscan.HDBSCAN(**kw2).fit_predict(D_nh.copy())

    # ═════════════════ MERGE ═══════════════════════════════════════════
    final_labels = np.full(n, -1, dtype=int)
    # healthy → 0
    for i in healthy_idx:
        final_labels[i] = 0
    # non-healthy → offset by +1
    for local_i, global_i in enumerate(nonhealthy_idx):
        if s2_labels[local_i] == -1:
            final_labels[global_i] = -1          # noise stays noise
        else:
            final_labels[global_i] = s2_labels[local_i] + 1

    # ═════════════════ NOISE RECOVERY ══════════════════════════════════
    noise_idx = np.where(final_labels == -1)[0]
    if len(noise_idx) > 0:
        cluster_ids = sorted(set(final_labels[final_labels >= 0]))
        if cluster_ids:
            for ni in noise_idx:
                best_cid, best_dist = -1, float("inf")
                for cid in cluster_ids:
                    members = np.where(final_labels == cid)[0]
                    avg_d = D_full[ni, members].mean()
                    if avg_d < best_dist:
                        best_dist = avg_d
                        best_cid = cid
                final_labels[ni] = best_cid

    # ═════════════════ LABEL NAMING ════════════════════════════════════
    label_names: dict[int, str] = {0: "healthy", -1: "noise"}
    for cid in sorted(set(s2_labels[s2_labels >= 0])):
        final_cid = cid + 1
        s2_mask = s2_labels == cid
        comps = Counter(detail_types[nh][s2_mask])
        dominant = comps.most_common(1)[0][0]
        label_names[final_cid] = dominant

    return final_labels, label_names, D_full


# ── public API ──────────────────────────────────────────────────────────────
def run_window_clustering(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    categorical: np.ndarray,
    detail_types: np.ndarray,
    current_day: int,
) -> dict:
    """
    Two-stage HDBSCAN on a rolling window of patients.

    Window: Days max(1, current_day - WINDOW_SIZE + 1) … current_day

    Returns dict:
        idx, labels, label_names, tsne, silhouette,
        n_clusters, n_noise, outbreak_label, window_start, window_end
    """
    window_start = max(1, current_day - WINDOW_SIZE + 1)
    window_end   = current_day
    days = df["day"].values
    mask = (days >= window_start) & (days <= window_end)
    idx  = np.where(mask)[0]

    A_sub = categorical[idx]
    B_sub = embeddings[idx]
    ct_sub = df["case_type"].values[idx]
    dt_sub = detail_types[idx]

    labels, label_names, D = _two_stage_hdbscan(A_sub, B_sub, ct_sub, dt_sub)
    coords = _run_tsne(D)

    unique = set(labels)
    n_clusters = len([l for l in unique if l >= 0])
    n_noise = int(np.sum(labels == -1))

    sil = None
    if n_clusters >= 2 and n_noise < len(labels):
        try:
            sil = float(silhouette_score(D, labels, metric="precomputed"))
        except ValueError:
            pass

    # identify outbreak cluster (highest novel_virus fraction)
    outbreak_label = _identify_outbreak(df, idx, labels)

    return {
        "idx":            idx,
        "labels":         labels,
        "label_names":    label_names,
        "tsne":           coords,
        "silhouette":     sil,
        "n_clusters":     n_clusters,
        "n_noise":        n_noise,
        "outbreak_label": outbreak_label,
        "window_start":   window_start,
        "window_end":     window_end,
    }


def _identify_outbreak(
    df: pd.DataFrame,
    idx: np.ndarray,
    labels: np.ndarray,
) -> int | None:
    """Cluster with highest novel_virus fraction (≥ 40 %)."""
    sub = df.iloc[idx]
    best_label, best_frac = None, 0.0
    for lbl in set(labels):
        if lbl <= 0:          # skip noise and healthy (cluster 0)
            continue
        group = sub.iloc[labels == lbl]
        nf = (group["case_type"] == "novel_virus").mean()
        if nf > best_frac:
            best_frac = nf
            best_label = lbl
    return best_label if best_frac >= 0.40 else None


# ── pre-compute all days ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Running two-stage HDBSCAN for all 7 days …")
def precompute_all_days(
    _df_json: str,
    _emb_bytes: bytes,
    _cat_bytes: bytes,
    _cat_shape: tuple[int, int],
    _detail_json: str,
    num_days: int = 7,
) -> dict:
    """
    Pre-compute two-stage clustering for Days 1 … 7 and cache.

    Serialised args keep Streamlit's caching happy.
    """
    import io
    df   = pd.read_json(io.StringIO(_df_json))
    emb  = np.frombuffer(_emb_bytes, dtype=np.float32).reshape(-1, 768)
    cat  = np.frombuffer(_cat_bytes, dtype=np.int8).reshape(_cat_shape)
    detail = np.array(json.loads(_detail_json))

    results: dict[int, dict] = {}
    for day in range(1, num_days + 1):
        results[day] = run_window_clustering(df, emb, cat, detail, day)
    return results
