"""
Canonical symptom schema for the v1 extraction pipeline.

Defines 30 symptoms across 8 clinical categories. Each symptom has:
  - key:      short snake_case identifier (CSV column name)
  - name:     human-readable display name
  - category: clinical grouping
  - aliases:  list of phrases / rewordings the LLM should recognise as
              belonging to this symptom (prevents false "novel" flags)

Scoring convention (same as prior work):
    0 = Not Mentioned / Not Present
    1 = Negated (explicitly denied)
    2 = Present (explicitly endorsed)
"""

from __future__ import annotations
from typing import Dict, List, TypedDict


class SymptomDef(TypedDict):
    key: str
    name: str
    category: str
    aliases: List[str]


# ─── Canonical checklist ─────────────────────────────────────────────────────

SYMPTOM_SCHEMA: List[SymptomDef] = [
    # ── COVID CORE RESPIRATORY ───────────────────────────────────────
    {
        "key": "sore_throat",
        "name": "Sore Throat",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "sore throat", "throat pain", "pharyngitis", "scratchy throat",
            "razor-blade throat", "stabbing throat", "throat burning",
            "painful swallowing", "hurts to swallow",
        ],
    },
    {
        "key": "cough",
        "name": "Cough",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "cough", "coughing", "dry cough", "hacking cough",
            "persistent cough", "barking cough", "productive cough",
            "wet cough", "hacky cough",
        ],
    },
    {
        "key": "throat_tickle",
        "name": "Persistent Tickle in Throat",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "tickle in throat", "throat tickle", "itchy throat",
            "scratchy feeling in throat", "irritated throat",
        ],
    },
    {
        "key": "fever",
        "name": "Fever",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "fever", "febrile", "temperature", "burning up",
            "feeling hot", "low-grade fever", "high fever",
            "subjective fever", "chills and fever", "hot to touch",
        ],
    },
    {
        "key": "chills",
        "name": "Chills / Rigors",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "chills", "rigors", "shivering", "shaking",
            "cold sweats", "feeling cold", "teeth chattering",
        ],
    },
    {
        "key": "nasal_congestion",
        "name": "Nasal Congestion",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "nasal congestion", "congestion", "stuffy nose",
            "blocked nose", "sinus pressure", "plugged up nose",
            "can't breathe through nose", "stuffed up",
        ],
    },
    {
        "key": "rhinorrhea",
        "name": "Rhinorrhea (Runny Nose)",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "rhinorrhea", "runny nose", "nose running",
            "nasal discharge", "dripping nose", "post-nasal drip",
        ],
    },
    {
        "key": "sneezing",
        "name": "Frequent Sneezing",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "sneezing", "frequent sneezing", "sneeze",
            "keeps sneezing", "sneezing fits",
        ],
    },
    {
        "key": "wheezing",
        "name": "Wheezing",
        "category": "COVID_CORE_RESPIRATORY",
        "aliases": [
            "wheezing", "wheeze", "whistling breath",
            "noisy breathing", "rattling chest",
        ],
    },

    # ── COVID CORE SYSTEMIC ──────────────────────────────────────────
    {
        "key": "fatigue",
        "name": "Fatigue",
        "category": "COVID_CORE_SYSTEMIC",
        "aliases": [
            "fatigue", "tired", "exhausted", "bone-tired",
            "wiped out", "no energy", "worn out", "drained",
            "lethargic", "sluggish",
        ],
    },
    {
        "key": "malaise",
        "name": "General Malaise",
        "category": "COVID_CORE_SYSTEMIC",
        "aliases": [
            "malaise", "feeling unwell", "not feeling right",
            "feeling lousy", "generally unwell", "feeling crummy",
            "feeling off", "body just feels wrong",
        ],
    },
    {
        "key": "myalgia",
        "name": "Myalgia (Muscle Aches)",
        "category": "COVID_CORE_SYSTEMIC",
        "aliases": [
            "myalgia", "muscle aches", "muscle pain", "body aches",
            "sore muscles", "aching all over", "muscles hurt",
        ],
    },
    {
        "key": "headache",
        "name": "Headache",
        "category": "COVID_CORE_SYSTEMIC",
        "aliases": [
            "headache", "head pain", "frontal headache",
            "throbbing head", "head pressure", "migraine",
            "pounding headache", "tension headache",
        ],
    },
    {
        "key": "brain_fog",
        "name": "Brain Fog",
        "category": "COVID_CORE_SYSTEMIC",
        "aliases": [
            "brain fog", "mental cloudiness", "can't think straight",
            "foggy headed", "confused thinking", "difficulty concentrating",
            "trouble focusing", "fuzzy thinking",
        ],
    },

    # ── HIGH SPECIFICITY SENSORY ─────────────────────────────────────
    {
        "key": "anosmia",
        "name": "Anosmia (Loss of Smell)",
        "category": "HIGH_SPECIFICITY_SENSORY",
        "aliases": [
            "anosmia", "loss of smell", "can't smell",
            "no sense of smell", "smell gone",
        ],
    },
    {
        "key": "ageusia",
        "name": "Ageusia (Loss of Taste)",
        "category": "HIGH_SPECIFICITY_SENSORY",
        "aliases": [
            "ageusia", "loss of taste", "can't taste",
            "no sense of taste", "taste gone", "food tastes like nothing",
        ],
    },
    {
        "key": "dizziness",
        "name": "Dizziness / Lightheadedness",
        "category": "HIGH_SPECIFICITY_SENSORY",
        "aliases": [
            "dizziness", "dizzy", "lightheaded", "lightheadedness",
            "room spinning", "vertigo", "unsteady", "woozy",
        ],
    },

    # ── EMERGENCY / RED FLAGS ────────────────────────────────────────
    {
        "key": "dyspnea",
        "name": "Dyspnea (Shortness of Breath)",
        "category": "EMERGENCY_RED_FLAGS",
        "aliases": [
            "dyspnea", "shortness of breath", "short of breath",
            "trouble breathing", "difficulty breathing", "can't catch breath",
            "air hunger", "breathless", "winded", "labored breathing",
        ],
    },
    {
        "key": "chest_tightness",
        "name": "Chest Tightness / Pressure",
        "category": "EMERGENCY_RED_FLAGS",
        "aliases": [
            "chest tightness", "chest pressure", "tight chest",
            "chest discomfort", "squeezing in chest",
            "heaviness in chest", "band around chest",
        ],
    },
    {
        "key": "chest_pain",
        "name": "Chest Pain",
        "category": "EMERGENCY_RED_FLAGS",
        "aliases": [
            "chest pain", "pain in chest", "sharp chest pain",
            "stabbing chest", "aching chest",
        ],
    },
    {
        "key": "syncope",
        "name": "Syncope (Fainting)",
        "category": "EMERGENCY_RED_FLAGS",
        "aliases": [
            "syncope", "fainted", "fainting", "passed out",
            "blacked out", "lost consciousness",
        ],
    },

    # ── ATYPICAL / GI / DERM ────────────────────────────────────────
    {
        "key": "nausea",
        "name": "Nausea / Emesis",
        "category": "ATYPICAL_GI_DERM",
        "aliases": [
            "nausea", "nauseous", "feeling sick", "queasy",
            "vomiting", "throwing up", "emesis", "puking",
            "sick to stomach",
        ],
    },
    {
        "key": "diarrhea",
        "name": "Diarrhea",
        "category": "ATYPICAL_GI_DERM",
        "aliases": [
            "diarrhea", "loose stools", "watery stools",
            "frequent bowel movements", "runs", "the runs",
        ],
    },
    {
        "key": "abdominal_pain",
        "name": "Abdominal Pain",
        "category": "ATYPICAL_GI_DERM",
        "aliases": [
            "abdominal pain", "stomach pain", "belly pain",
            "stomach ache", "stomach cramps", "tummy ache",
            "gut pain", "epigastric pain",
        ],
    },
    {
        "key": "anorexia",
        "name": "Anorexia (Loss of Appetite)",
        "category": "ATYPICAL_GI_DERM",
        "aliases": [
            "anorexia", "loss of appetite", "no appetite",
            "not hungry", "can't eat", "don't feel like eating",
        ],
    },
    {
        "key": "rash",
        "name": "Rash / Skin Changes",
        "category": "ATYPICAL_GI_DERM",
        "aliases": [
            "rash", "skin rash", "hives", "pruritus", "itchy skin",
            "skin irritation", "lesions", "bumps on skin",
            "discolored toes", "covid toes",
        ],
    },
    {
        "key": "conjunctivitis",
        "name": "Conjunctivitis (Eye Irritation)",
        "category": "ATYPICAL_GI_DERM",
        "aliases": [
            "conjunctivitis", "pink eye", "watery eyes",
            "itchy eyes", "red eyes", "eye irritation",
        ],
    },

    # ── OTHER / CONTEXTUAL ───────────────────────────────────────────
    {
        "key": "exposure_to_sick",
        "name": "Exposure to Sick Contact",
        "category": "OTHER",
        "aliases": [
            "exposure", "sick contact", "been around someone sick",
            "coworker sick", "family member sick", "partner sick",
            "someone at work", "someone at home",
        ],
    },
    {
        "key": "edema",
        "name": "Peripheral Edema (Swelling)",
        "category": "OTHER",
        "aliases": [
            "edema", "swelling", "swollen ankles", "swollen feet",
            "swollen legs", "fluid retention", "puffy",
        ],
    },
    {
        "key": "back_pain",
        "name": "Back Pain",
        "category": "OTHER",
        "aliases": [
            "back pain", "lower back pain", "backache",
            "back strain", "lumbar pain", "sore back",
        ],
    },
]


# ─── Derived lookups ─────────────────────────────────────────────────────────

SYMPTOM_KEYS: List[str] = [s["key"] for s in SYMPTOM_SCHEMA]
SYMPTOM_NAMES: List[str] = [s["name"] for s in SYMPTOM_SCHEMA]
KEY_TO_DEF: Dict[str, SymptomDef] = {s["key"]: s for s in SYMPTOM_SCHEMA}
NAME_TO_KEY: Dict[str, str] = {s["name"]: s["key"] for s in SYMPTOM_SCHEMA}

CATEGORIES: List[str] = sorted(set(s["category"] for s in SYMPTOM_SCHEMA))

# Build a flat lookup: alias (lowercase) → symptom key
ALIAS_TO_KEY: Dict[str, str] = {}
for s in SYMPTOM_SCHEMA:
    for alias in s["aliases"]:
        ALIAS_TO_KEY[alias.lower()] = s["key"]

# CSV column order
CSV_COLUMNS: List[str] = ["filename"] + SYMPTOM_KEYS + ["novel_count"]
