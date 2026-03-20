# Sentinel AI — Deployment Guide

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| pip | 23.0+ |
| OpenAI API Key | Required for Conversation Demo tab only |

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/AJain271/Sentinel-AI-LLM-Based-Epidemic-Surveillance.git
cd Sentinel-AI-LLM-Based-Epidemic-Surveillance

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

## Required Python Packages

```
streamlit>=1.30
plotly>=5.18
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
umap-learn>=0.5.5
openai>=1.0
python-dotenv>=1.0
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Only for Conversation Demo | Your OpenAI API key |

The app loads environment variables from `Syndromic Dashboard/.env` via `python-dotenv`.
Create (or edit) that file:

```
OPENAI_API_KEY=sk-your-key-here
```

The `.env` file is git-ignored and will never be committed.

Alternatively, you can set the variable directly in your shell:

```bash
# Linux/Mac
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

## Running the Dashboard

```bash
# From the Syndromic Dashboard directory
cd "Syndromic Dashboard"
streamlit run app.py --server.port 8501
```

The dashboard will open in your default browser at `http://localhost:8501`.

## Data Requirements

The dashboard expects the following data to be pre-computed:

1. **Patient JSONs** — 500 patient profiles in `LLM Symptom Extraction Full Run/results/fewshot/`
2. **ClinicalBERT Embeddings** — Cached at `Clustering/clinicalbert_embeddings_cache.npy`
3. **Clean Transcripts** — Reference transcripts in `Clean Transcripts/` (for style reference)
4. **CCDA Ground-Truth JSONs** — In `Synthetic Transcript Generation/Synthea CCDAs/used/` (for Conversation Demo)

## Clustering Cache

On first run, the UMAP + HDBSCAN clustering is computed for all 7 simulation days and cached to
disk at `Syndromic Dashboard/clustering_cache.pkl`. Subsequent launches load from cache unless the
underlying data changes (detected via MD5 hash).

To force a re-cluster, delete `clustering_cache.pkl` and restart.

## Production Deployment

```bash
streamlit run app.py \
    --server.port 8501 \
    --server.maxUploadSize 200 \
    --server.headless true \
    --browser.gatherUsageStats false
```

For cloud deployment (e.g., Streamlit Community Cloud, AWS, GCP):

1. Push the repository to GitHub
2. Ensure `requirements.txt` is in the root
3. Set `OPENAI_API_KEY` as an environment secret
4. Point the deployment to `Syndromic Dashboard/app.py`
