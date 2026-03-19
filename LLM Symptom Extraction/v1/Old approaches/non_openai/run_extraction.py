"""
Batch runner for v1 symptom extraction pipeline.

Processes a directory of transcripts, writes:
  1. CSV   — extraction_results.csv  (30 symptom columns + novel_count)
  2. JSON  — extraction_details/<Filename>.json  (reasoning, scores,
             novel symptoms with embeddings)

Resumable: skips transcripts that already have a JSON sidecar.

Usage:
    python run_extraction.py
    python run_extraction.py --input_dir "../../Clean Transcripts" --out_dir ./output
    python run_extraction.py --file single_transcript.txt
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from symptom_schema import SYMPTOM_KEYS, CSV_COLUMNS
from extract_symptoms import extract, API_KEY as EXTRACT_KEY
from embed_novel import embed_and_reduce, API_KEY as EMBED_KEY

# ─── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "..", "..", "..", "Clean Transcripts")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "output")
RATE_LIMIT_SECONDS = 0.3   # pause between API calls


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _share_api_key() -> None:
    """Propagate the API key and backend settings to sub-modules."""
    import extract_symptoms
    import embed_novel

    key = (
        extract_symptoms.API_KEY
        or embed_novel.API_KEY
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if key:
        extract_symptoms.API_KEY = key
        embed_novel.API_KEY = key


def _configure_backend(backend: str, model: str, embed_backend: str) -> None:
    """Set backend/model in sub-modules and reset cached clients."""
    import extract_symptoms
    import embed_novel

    extract_symptoms.BACKEND = backend
    extract_symptoms._client = None
    if model:
        if backend == "ollama":
            extract_symptoms.OLLAMA_MODEL = model
        else:
            extract_symptoms.OPENAI_MODEL = model

    embed_novel.EMBED_BACKEND = embed_backend
    embed_novel._client = None


def process_single(
    transcript_path: str,
    details_dir: str,
    pca_dims: int = 8,
) -> dict:
    """Extract symptoms from one transcript, embed novel symptoms, return result dict."""
    filename = os.path.basename(transcript_path)
    transcript = _load_transcript(transcript_path)

    # Stage 1+2: CoT extraction
    result = extract(transcript)
    time.sleep(RATE_LIMIT_SECONDS)

    # Stage 3: embed novel symptoms
    if result["novel_symptoms"]:
        result["novel_symptoms"] = embed_and_reduce(
            result["novel_symptoms"], n_components=pca_dims
        )
        time.sleep(RATE_LIMIT_SECONDS)

    # Save JSON sidecar
    sidecar = {
        "filename": filename,
        "reasoning": result["reasoning"],
        "scores": result["scores"],
        "novel_symptoms": result["novel_symptoms"],
    }
    sidecar_path = os.path.join(details_dir, filename.replace(".txt", ".json"))
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    return sidecar


def run_batch(
    input_dir: str,
    output_dir: str,
    pca_dims: int = 8,
) -> None:
    """Process all .txt files in input_dir, write CSV + JSON sidecars."""
    _share_api_key()

    details_dir = os.path.join(output_dir, "extraction_details")
    _ensure_dir(details_dir)

    csv_path = os.path.join(output_dir, "extraction_results.csv")

    # Find transcript files
    txt_files = sorted(
        f for f in os.listdir(input_dir)
        if f.endswith(".txt")
    )
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    # Check for already-processed files (resumable)
    already_done = set()
    if os.path.isdir(details_dir):
        already_done = {
            f.replace(".json", ".txt")
            for f in os.listdir(details_dir)
            if f.endswith(".json")
        }

    to_process = [f for f in txt_files if f not in already_done]
    total = len(txt_files)
    skipped = len(already_done & set(txt_files))

    print(f"Found {total} transcripts, {skipped} already processed, {len(to_process)} remaining.")
    print(f"Output: {output_dir}")
    print()

    # Process
    all_results: List[dict] = []

    # Load existing results from JSON sidecars
    for done_file in sorted(already_done & set(txt_files)):
        sidecar_path = os.path.join(details_dir, done_file.replace(".txt", ".json"))
        with open(sidecar_path, "r", encoding="utf-8") as f:
            all_results.append(json.load(f))

    for i, filename in enumerate(to_process, 1):
        filepath = os.path.join(input_dir, filename)
        print(f"[{i}/{len(to_process)}] Processing {filename}...", end=" ", flush=True)

        try:
            result = process_single(filepath, details_dir, pca_dims)
            all_results.append(result)

            novel_count = len(result["novel_symptoms"])
            present_count = sum(1 for v in result["scores"].values() if v == 2)
            negated_count = sum(1 for v in result["scores"].values() if v == 1)

            status = f"present={present_count}, negated={negated_count}"
            if novel_count:
                novel_names = [ns["symptom"] for ns in result["novel_symptoms"]]
                status += f", novel={novel_count} ({', '.join(novel_names)})"
            print(status)

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Write CSV
    all_results.sort(key=lambda r: r["filename"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in all_results:
            row = {"filename": r["filename"]}
            for key in SYMPTOM_KEYS:
                row[key] = r["scores"].get(key, 0)
            row["novel_count"] = len(r.get("novel_symptoms", []))
            writer.writerow(row)

    print(f"\n{'=' * 60}")
    print(f"Done! Processed {len(all_results)} transcripts total.")
    print(f"CSV:     {csv_path}")
    print(f"Details: {details_dir}/")

    # Summary stats
    total_novel = sum(len(r.get("novel_symptoms", [])) for r in all_results)
    if total_novel:
        all_novel = []
        for r in all_results:
            for ns in r.get("novel_symptoms", []):
                all_novel.append(ns["symptom"])
        unique_novel = set(all_novel)
        print(f"\nNovel symptoms found: {total_novel} total, {len(unique_novel)} unique")
        for ns in sorted(unique_novel):
            count = all_novel.count(ns)
            print(f"  - {ns} (x{count})")


def run_single_file(filepath: str, output_dir: str, pca_dims: int = 8) -> None:
    """Process a single file and print results."""
    _share_api_key()

    details_dir = os.path.join(output_dir, "extraction_details")
    _ensure_dir(details_dir)

    print(f"Processing: {filepath}")
    result = process_single(filepath, details_dir, pca_dims)

    print(f"\nReasoning: {result['reasoning']}")
    print(f"\nScores (non-zero):")
    for key, val in result["scores"].items():
        if val > 0:
            label = "NEGATED" if val == 1 else "PRESENT"
            print(f"  {key}: {label}")

    if result["novel_symptoms"]:
        print(f"\nNovel symptoms ({len(result['novel_symptoms'])}):")
        for ns in result["novel_symptoms"]:
            print(f"  - {ns['symptom']}: \"{ns['evidence']}\"")
            print(f"    PCA embedding: {ns.get('embedding_pca', [])}")
    else:
        print("\nNo novel symptoms detected.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="v1 Symptom Extraction Pipeline")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT,
                        help="Directory of .txt transcripts")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT,
                        help="Output directory for CSV + JSON")
    parser.add_argument("--file", default="",
                        help="Process a single file instead of batch")
    parser.add_argument("--pca_dims", type=int, default=8,
                        help="PCA dimensionality for novel embeddings")
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai",
                        help="LLM backend for extraction (default: openai)")
    parser.add_argument("--model", default="",
                        help="Model name override (e.g., mistral, gemma:2b, gpt-4o)")
    parser.add_argument("--embed_backend", choices=["openai", "ollama"], default=None,
                        help="Embedding backend (default: same as --backend)")
    args = parser.parse_args()

    embed_bk = args.embed_backend or args.backend
    _configure_backend(args.backend, args.model, embed_bk)
    _share_api_key()

    print(f"Backend:   {args.backend} ({args.model or 'default'})")
    print(f"Embedding: {embed_bk}")
    print()

    if args.file:
        run_single_file(args.file, args.out_dir, args.pca_dims)
    else:
        run_batch(args.input_dir, args.out_dir, args.pca_dims)


if __name__ == "__main__":
    main()
