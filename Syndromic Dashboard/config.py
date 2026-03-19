"""
Configuration constants for the Syndromic Surveillance Dashboard.
"""
from pathlib import Path

# ── Workspace root ──────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent

# ── Data paths ──────────────────────────────────────────────────────────────
FEWSHOT_DIR = WORKSPACE / "LLM Symptom Extraction Full Run" / "results" / "fewshot"
EMBEDDINGS_CACHE = WORKSPACE / "Clustering" / "embeddings_cache.npy"

# ── Outbreak timeline ───────────────────────────────────────────────────────
OUTBREAK_START = "2025-12-26"          # Day 1
OUTBREAK_END   = "2026-01-01"          # Day 7
NUM_DAYS       = 7

# ── LLM cost rates (USD per 1 M tokens) ────────────────────────────────────
INPUT_COST_PER_M  = 2.50
OUTPUT_COST_PER_M = 10.00

# ── Hospital hub definitions (7 SC hubs) ────────────────────────────────────
SC_HOSPITAL_HUBS = [
    {"hub_id": "H1", "name": "AnMed Health Medical Center",    "city": "Anderson",     "lat": 34.526,  "lng": -82.646},
    {"hub_id": "H2", "name": "Prisma Health Greenville",       "city": "Greenville",   "lat": 34.819,  "lng": -82.416},
    {"hub_id": "H3", "name": "Prisma Health Richland",         "city": "Columbia",     "lat": 34.032,  "lng": -81.033},
    {"hub_id": "H4", "name": "Aiken Regional Medical Center",  "city": "Aiken",        "lat": 33.565,  "lng": -81.763},
    {"hub_id": "H5", "name": "MUSC Health",                    "city": "Charleston",   "lat": 32.785,  "lng": -79.947},
    {"hub_id": "H6", "name": "McLeod Regional Medical Center", "city": "Florence",     "lat": 34.192,  "lng": -79.765},
    {"hub_id": "H7", "name": "Grand Strand Medical Center",    "city": "Myrtle Beach", "lat": 33.748,  "lng": -78.847},
]

HUB_LOOKUP = {h["hub_id"]: h for h in SC_HOSPITAL_HUBS}

# ── Color palette (Blue-accent, minimalist) ─────────────────────────────────
class Colors:
    PAGE_BG        = "#F8FAFC"
    CARD_BG        = "#FFFFFF"
    TEXT_PRIMARY   = "#0F172A"
    TEXT_SECONDARY = "#64748B"
    BORDER         = "#E2E8F0"

    # Accent
    ACCENT         = "#2563EB"       # vivid blue — primary accent
    ACCENT_LIGHT   = "#EFF6FF"       # light blue background
    ACCENT_BORDER  = "#93C5FD"       # soft blue border

    # Chart palette
    OUTBREAK       = "#DC2626"       # red — outbreak / safety alerts only
    OUTBREAK_DIM   = "rgba(220,38,38,0.18)"
    FLU            = "#94A3B8"       # slate gray — flu-like
    DIFFERENTIAL   = "#CBD5E1"       # cool gray  — differential
    HEALTHY        = "#E2E8F0"       # light gray — healthy
    NOISE          = "rgba(148,163,184,0.30)"
    CLUSTER_BASE   = "#60A5FA"       # medium blue — baseline cluster

    CASE_TYPE_MAP = {
        "novel_virus":  "#2563EB",
        "flu_like":     "#94A3B8",
        "differential": "#CBD5E1",
        "healthy":      "#E2E8F0",
    }

# ── SC map center ───────────────────────────────────────────────────────────
SC_CENTER_LAT = 33.85
SC_CENTER_LNG = -80.95
SC_ZOOM       = 6.3
