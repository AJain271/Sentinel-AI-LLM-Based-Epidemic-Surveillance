"""
Unsupervised UMAP + HDBSCAN Pipeline
=====================================
1. One-hot encode 70 symptoms × 3 classes (-1, 0, 1) → 210-dim
2. Semantic embed unmapped/novel symptoms via ClinicalBERT → 768-dim (zero-padded if none)
3. Z-score normalise both matrices independently
4. Concatenate → (N, 978)
5. UMAP dimensionality reduction (euclidean)
6. HDBSCAN (EOM) clustering
7. Compare to ground truth ONLY at the final metrics step
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
import umap
import matplotlib.pyplot as plt

# ── Config ──
RANDOM_STATE = 42
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 10
UMAP_MIN_DIST = 0.0

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "LLM Symptom Extraction Full Run" / "results" / "fewshot"
MASTER_CSV = SCRIPT_DIR.parent.parent / "LLM Symptom Extraction Full Run" / "results" / "master_results.csv"
OUTPUT_DIR = SCRIPT_DIR

BERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"


# ── Data loading (reused logic from existing pipeline) ──

def derive_case_type(filename: str) -> str:
    fname = filename.upper()
    if fname.startswith("NOVEL_SYNTHETIC"):
        return "novel_virus"
    elif fname.startswith("FLU_SYNTHETIC"):
        return "flu_like"
    elif fname.startswith("DIFF_SYNTHETIC"):
        return "differential"
    elif fname.startswith("HEALTHY_SYNTHETIC"):
        return "healthy"
    return "unknown"


def load_patient_data(data_dir: Path):
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {data_dir}")

    filenames, case_types, all_scores, anomaly_texts = [], [], [], []
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            rec = json.load(f)
        fname = rec.get("filename", jf.stem)
        filenames.append(fname)
        case_types.append(derive_case_type(fname))
        all_scores.append(rec["checklist_scores"])

        parts = []
        for sym in rec.get("unmapped_symptoms", []):
            term = sym.get("term", "")
            defn = sym.get("definition", "")
            if term:
                parts.append(f"{term}: {defn}".strip())
        anomaly_texts.append(". ".join(parts))

    print(f"Loaded {len(filenames)} patients")
    return filenames, case_types, all_scores, anomaly_texts


def load_detail_types(filenames, case_types):
    diff_map = {}
    with open(MASTER_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ds = row.get("differential_system", "").strip()
            if ds:
                diff_map[row["filename"].strip()] = ds
    return [
        diff_map.get(fn, "differential") if ct == "differential" else ct
        for fn, ct in zip(filenames, case_types)
    ]


# ── Step 1: One-hot encode symptoms ──

def build_onehot_matrix(all_scores: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Each symptom has 3 classes: -1, 0, 1.
    One-hot → 70 × 3 = 210 columns.
    """
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

    print(f"One-hot matrix: {onehot.shape}  (70 symptoms × 3 classes)")
    return onehot, symptom_names


# ── Step 2: ClinicalBERT semantic embedding ──

def build_clinical_bert_embeddings(anomaly_texts: list[str]) -> np.ndarray:
    """
    Embed unmapped symptom texts via ClinicalBERT (mean-pooled last hidden state).
    Empty texts → zero vector.
    """
    print(f"Loading ClinicalBERT ({BERT_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    embed_dim = model.config.hidden_size  # 768

    n = len(anomaly_texts)
    B = np.zeros((n, embed_dim), dtype=np.float64)
    non_empty = [(i, t) for i, t in enumerate(anomaly_texts) if t.strip()]
    print(f"Embedding {len(non_empty)} non-empty anomaly texts...")

    with torch.no_grad():
        for count, (i, text) in enumerate(non_empty, 1):
            tokens = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512, padding=True).to(device)
            output = model(**tokens)
            # Mean-pool over token dimension (excluding padding)
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            B[i] = pooled.cpu().numpy().flatten()
            if count % 50 == 0 or count == len(non_empty):
                print(f"  [{count}/{len(non_empty)}]")

    print(f"Semantic matrix: {B.shape}")
    return B


# ── Main pipeline ──

def main():
    t0 = time.time()

    # ── Load data ──
    print("=" * 60)
    print("  LOADING DATA")
    print("=" * 60)
    filenames, case_types, all_scores, anomaly_texts = load_patient_data(DATA_DIR)

    # ── Step 1: One-hot ──
    print("\n" + "=" * 60)
    print("  STEP 1: ONE-HOT ENCODING (70 × 3)")
    print("=" * 60)
    A_onehot, symptom_names = build_onehot_matrix(all_scores)

    # ── Step 2: ClinicalBERT embeddings ──
    print("\n" + "=" * 60)
    print("  STEP 2: CLINICAL-BERT SEMANTIC EMBEDDINGS")
    print("=" * 60)

    cache_path = OUTPUT_DIR / "clinicalbert_embeddings_cache.npy"
    if cache_path.exists():
        B_semantic = np.load(cache_path)
        if B_semantic.shape[0] == len(filenames):
            print(f"Loaded cached ClinicalBERT embeddings: {B_semantic.shape}")
        else:
            B_semantic = build_clinical_bert_embeddings(anomaly_texts)
            np.save(cache_path, B_semantic)
            print(f"Cached → {cache_path}")
    else:
        B_semantic = build_clinical_bert_embeddings(anomaly_texts)
        np.save(cache_path, B_semantic)
        print(f"Cached → {cache_path}")

    # ── Step 3: Z-score normalise independently ──
    print("\n" + "=" * 60)
    print("  STEP 3: Z-SCORE NORMALISATION")
    print("=" * 60)

    scaler_oh = StandardScaler()
    A_norm = scaler_oh.fit_transform(A_onehot)
    print(f"One-hot normalised:   mean={A_norm.mean():.4f}, std={A_norm.std():.4f}")

    scaler_sem = StandardScaler()
    B_norm = scaler_sem.fit_transform(B_semantic)
    print(f"Semantic normalised:  mean={B_norm.mean():.4f}, std={B_norm.std():.4f}")

    # ── Step 4: Concatenate ──
    X = np.hstack([A_norm, B_norm])
    print(f"Combined feature matrix: {X.shape}")

    # ── Step 5: UMAP ──
    print("\n" + "=" * 60)
    print("  STEP 4: UMAP DIMENSIONALITY REDUCTION")
    print("=" * 60)
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    X_umap = reducer.fit_transform(X)
    print(f"UMAP embedding: {X_umap.shape}")

    # Project the 10D UMAP embedding → 2D for visualisation
    # (must project the SAME space HDBSCAN clustered on, not a separate fit)
    reducer_2d = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=2,
        min_dist=0.5,
        metric="euclidean",
        random_state=RANDOM_STATE,
        spread=2.0,
    )
    coords_2d = reducer_2d.fit_transform(X_umap)
    print(f"UMAP 10D → 2D projection: {coords_2d.shape}")

    # ── Step 6: HDBSCAN (EOM) ──
    print("\n" + "=" * 60)
    print("  STEP 5: HDBSCAN CLUSTERING (EOM)")
    print("=" * 60)
    clusterer = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X_umap)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points:   {n_noise}")

    for cid in sorted(set(labels)):
        tag = f"Cluster {cid}" if cid != -1 else "Noise (-1)"
        print(f"  {tag}: {(labels == cid).sum()}")

    # ── Step 7: Metrics (ground truth comparison — FINAL step only) ──
    print("\n" + "=" * 60)
    print("  STEP 6: EVALUATION METRICS (ground truth)")
    print("=" * 60)

    detail_types = load_detail_types(filenames, case_types)

    # Silhouette on UMAP space
    non_noise = labels != -1
    sil = None
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        sil = float(silhouette_score(X_umap[non_noise], labels[non_noise]))

    # External clustering metrics
    ari = float(adjusted_rand_score(detail_types, labels))
    nmi = float(normalized_mutual_info_score(detail_types, labels))

    # Per-cluster composition
    per_cluster = {}
    for cid in sorted(set(labels)):
        mask = labels == cid
        dt_list = [detail_types[i] for i in range(len(detail_types)) if mask[i]]
        ct_list = [case_types[i] for i in range(len(case_types)) if mask[i]]
        comp = Counter(dt_list)
        dominant, dominant_count = comp.most_common(1)[0]
        purity = dominant_count / int(mask.sum())
        per_cluster[str(cid)] = {
            "size": int(mask.sum()),
            "purity": round(purity, 4),
            "dominant_type": dominant,
            "composition": dict(Counter(ct_list)),
            "detail_composition": dict(comp),
        }

    # Novel recall
    novel_mask = np.array([ct == "novel_virus" for ct in case_types])
    novel_recall = 0.0
    if n_clusters > 0 and novel_mask.sum() > 0:
        for cid in set(labels[labels != -1]):
            novel_recall = max(novel_recall,
                               (labels[novel_mask] == cid).sum() / novel_mask.sum())

    # Per-category recall
    category_recall = {}
    all_cats = sorted(set(detail_types))
    for cat in all_cats:
        total = sum(1 for dt in detail_types if dt == cat)
        best_count = 0
        best_cid = None
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            mask = labels == cid
            in_cluster = sum(1 for i in range(len(labels)) if mask[i] and detail_types[i] == cat)
            if in_cluster > best_count:
                best_count = in_cluster
                best_cid = cid
        recall = best_count / total if total > 0 else 0.0
        category_recall[cat] = {
            "total": total,
            "best_cluster_count": best_count,
            "recall": round(recall, 4),
            "primary_cluster": int(best_cid) if best_cid is not None else None,
        }

    # Print summary
    print(f"\n  Silhouette:          {sil:.4f}" if sil else "  Silhouette:          N/A")
    print(f"  ARI:                 {ari:.4f}")
    print(f"  NMI:                 {nmi:.4f}")
    print(f"  Novel Recall:        {novel_recall:.4f}")
    print(f"\n  Per-cluster breakdown:")
    for cid_str, info in per_cluster.items():
        cid = int(cid_str)
        tag = f"C{cid}" if cid != -1 else "Noise"
        print(f"    {tag}: n={info['size']}, purity={info['purity']:.2%}, "
              f"dominant={info['dominant_type']}, detail={info['detail_composition']}")
    print(f"\n  Per-category recall:")
    for cat, info in category_recall.items():
        print(f"    {cat:<22} {info['total']:>4} → best cluster has "
              f"{info['best_cluster_count']}, recall={info['recall']:.2%}")

    # ── Save outputs ──
    print("\n" + "=" * 60)
    print("  SAVING OUTPUTS")
    print("=" * 60)

    # Metrics JSON
    metrics = {
        "algorithm": "UMAP + HDBSCAN (EOM)",
        "embedding": "One-hot (70×3) + ClinicalBERT (768)",
        "normalisation": "Z-score (independent)",
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "umap_n_components": UMAP_N_COMPONENTS,
        "umap_min_dist": UMAP_MIN_DIST,
        "umap_metric": "euclidean",
        "hdbscan_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE,
        "hdbscan_min_samples": HDBSCAN_MIN_SAMPLES,
        "hdbscan_cluster_selection_method": "eom",
        "n_patients": len(filenames),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": sil,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "novel_recall": round(novel_recall, 4),
        "per_cluster": per_cluster,
        "per_category_recall": category_recall,
    }
    metrics_path = OUTPUT_DIR / "cluster_metrics_unsupervised.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved → {metrics_path}")

    # Results CSV
    csv_path = OUTPUT_DIR / "clustering_results_unsupervised.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "case_type", "detail_type", "cluster_id",
                         "umap_1", "umap_2"])
        for i in range(len(filenames)):
            writer.writerow([
                filenames[i], case_types[i], detail_types[i], int(labels[i]),
                f"{coords_2d[i, 0]:.4f}", f"{coords_2d[i, 1]:.4f}",
            ])
    print(f"Saved → {csv_path}")

    # ── Plots ──
    type_colors = {
        "novel_virus": "#E63946", "flu_like": "#457B9D", "healthy": "#BBBBBB",
        "gastrointestinal": "#2A9D8F", "dermatological": "#E9C46A",
        "musculoskeletal": "#F4A261", "neurological": "#7B2D8E",
        "differential": "#2A9D8F",
    }
    type_labels = {
        "novel_virus": "Novel Virus", "flu_like": "Flu-Like", "healthy": "Healthy",
        "gastrointestinal": "GI", "dermatological": "Derm",
        "musculoskeletal": "MSK", "neurological": "Neuro",
    }

    # Ground truth scatter
    fig, ax = plt.subplots(figsize=(12, 10))
    for ct in ["healthy", "gastrointestinal", "dermatological",
               "musculoskeletal", "neurological", "flu_like", "novel_virus"]:
        mask = np.array([d == ct for d in detail_types])
        if mask.any():
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=type_colors.get(ct, "#000"),
                       label=f"{type_labels.get(ct, ct)} ({mask.sum()})",
                       s=50, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_title("Ground Truth (UMAP 2D)", fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=14, markerscale=1.5, framealpha=0.9)
    ax.set_xlabel("UMAP 1", fontsize=12); ax.set_ylabel("UMAP 2", fontsize=12)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "unsupervised_ground_truth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → unsupervised_ground_truth.png")

    # Cluster scatter
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_labels = sorted(set(labels))
    palette = plt.colormaps.get_cmap("tab10").resampled(max(n_clusters, 1))
    for idx, cid in enumerate(unique_labels):
        mask = labels == cid
        if cid == -1:
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c="#CCCCCC", label=f"Noise ({mask.sum()})",
                       s=30, alpha=0.4, edgecolors="white", linewidths=0.3)
        else:
            color = palette(idx if -1 not in unique_labels else idx - 1)
            ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                       c=[color], label=f"C{cid} ({mask.sum()})",
                       s=50, alpha=0.8, edgecolors="white", linewidths=0.4)
    ax.set_title("HDBSCAN Clusters (UMAP 2D, EOM)", fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=14, markerscale=1.5, framealpha=0.9)
    ax.set_xlabel("UMAP 1", fontsize=12); ax.set_ylabel("UMAP 2", fontsize=12)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "unsupervised_cluster_scatter.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → unsupervised_cluster_scatter.png")

    elapsed = time.time() - t0
    print(f"\nPipeline complete — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
