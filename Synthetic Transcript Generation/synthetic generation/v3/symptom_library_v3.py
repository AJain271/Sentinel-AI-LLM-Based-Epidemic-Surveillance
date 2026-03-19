"""
Symptom library with feasibility-grouped sampling for multiple case types.

Provides the master SYMPTOM_LIBRARY dictionary (mirroring the project's
SYMPTOM_LIBRARY = {.txt) and a ``sample_symptoms`` function that returns
clinically coherent sets of present + negated symptoms.

Supported case types:
    novel_virus  — COVID-like viral respiratory illness (original behaviour)
    flu_like     — classic influenza profile (no anosmia/ageusia)
    differential — single non-respiratory body-system focus
    healthy      — routine wellness checkup (no present symptoms)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

# ─── Master symptom library ─────────────────────────────────────────────────
# Each string uses the format "Clinical term (patient-friendly description)"
# so the prompt can instruct the LLM to use the parenthetical phrasing.

# Only core COVID categories are used for sampling present symptoms.
# ATYPICAL_GI_DERM and extreme EMERGENCY_RED_FLAGS are excluded.
SYMPTOM_LIBRARY: Dict[str, list] = {
    "COVID_CORE_RESPIRATORY": [
        "Sore throat (stabbing/razor-blade pain)",
        "Dry, hacking cough",
        "Persistent tickle in throat",
        "Low-grade fever (100.4F - 102F)",
        "Subjective fever (feeling hot/burning up)",
        "Chills and rigors",
        "Nasal congestion (sinus pressure)",
        "Rhinorrhea (runny nose)",
        "Frequent sneezing",
    ],
    "COVID_CORE_SYSTEMIC": [
        "Profound fatigue (bone-tired)",
        "General malaise (feeling wiped out)",
        "Myalgia (deep muscle aches in back/thighs)",
        "Frontal headache (throbbing pressure)",
        "Brain fog (mental cloudiness)",
    ],
    "HIGH_SPECIFICITY_SENSORY": [
        "Anosmia (total loss of smell)",
        "Ageusia (total loss of taste)",
        "Dizziness/Lightheadedness upon standing",
    ],
    "EMERGENCY_RESPIRATORY": [
        "Dyspnea (shortness of breath)",
        "Chest tightness/pressure",
    ],
}

# ─── Novel-virus hallmark symptoms ──────────────────────────────────────────
# Unique symptoms that do NOT overlap with existing COVID/flu/differential
# libraries.  When case_type == "novel_virus", 2-4 of these are injected so
# they comprise ~30-50 % of the patient's present-symptom set, giving the
# clustering pipeline a distinct signal to detect.

NOVEL_VIRUS_HALLMARKS: List[str] = [
    "Lymphadenopathy (tender lumps in armpits)",
    "Dysgeusia (metallic/distorted taste — everything tastes wrong)",
    "Hemoptysis (coughing up blood-tinged mucus)",
    "Skin desquamation (peeling skin on palms/fingertips)",
    "Melanonychia (nails turning dark/black)",
]

# ─── Flu-like symptom library ────────────────────────────────────────────────
# Classic influenza profile — overlaps with some respiratory/systemic COVID
# symptoms but EXCLUDES high-specificity COVID hallmarks (anosmia, ageusia,
# brain fog).  Uses distinct severity descriptions to sound like flu.

FLU_LIKE_SYMPTOMS: Dict[str, list] = {
    "FLU_RESPIRATORY": [
        "Sore throat (stabbing/razor-blade pain)",
        "Nasal congestion (sinus pressure)",
        "Rhinorrhea (runny nose)",
        "Frequent sneezing",
    ],
    "FLU_SYSTEMIC": [
        "Sudden high fever (102F-104F, came on fast)",
        "Severe chills and shivering (teeth chattering)",
        "Myalgia (deep muscle aches in back/thighs)",
        "Frontal headache (throbbing pressure)",
        "Profound fatigue (bone-tired)",
        "General malaise (feeling wiped out)",
    ],
}

# Clinically relevant symptoms a doctor would actually rule out during a
# flu-like visit.  These are respiratory / viral differential symptoms that
# make sense to ask about — NOT random body-system complaints.
FLU_RELEVANT_NEGATIONS: List[str] = [
    # COVID hallmarks the doctor would rule out
    "Anosmia (total loss of smell)",
    "Ageusia (total loss of taste)",
    "Brain fog (mental cloudiness)",
    # Respiratory red flags
    "Dyspnea (shortness of breath)",
    "Chest tightness/pressure",
    # Other viral / respiratory differentials
    "Nausea (feeling queasy/sick to stomach)",
    "Diarrhea (loose/watery stools)",
    "Otalgia (ear ache/pressure)",
    "Nasal congestion (sinus pressure)",
    "Eye pain (pain behind eyes/hurts to move eyes)",
    "Wheezing (whistling sound when breathing)",
    "Night sweats (waking up drenched)",
    "Rash (red/raised patches on skin)",
    "Neck stiffness with headache (meningeal signs)",
]

# ─── Differential-distractor assessment phrases ─────────────────────────────
# Maps each non-COVID cluster name → (assessment phrase, plan phrase)
DIFFERENTIAL_ASSESSMENT: Dict[str, tuple] = {
    "gastrointestinal":  ("a gastrointestinal issue — could be a stomach bug or viral gastroenteritis",
                          "bland diet, fluids, and if it doesn't improve we can run some labs"),
    "dermatological":    ("a dermatological reaction — possibly contact dermatitis or an allergic reaction",
                          "topical treatment, avoid irritants, and follow up if it spreads"),
    "musculoskeletal":   ("a musculoskeletal issue — likely a strain or overuse injury",
                          "rest, ice, anti-inflammatory medication, and physical therapy if needed"),
    "cardiac":           ("something we want to look into more carefully from a cardiac standpoint",
                          "an EKG, some bloodwork, and a follow-up with cardiology"),
    "neurological":      ("a neurological concern worth investigating further",
                          "some imaging and bloodwork, and possibly a neurology referral"),
    "genitourinary":     ("a urinary issue — could be an infection or something else",
                          "a urinalysis, fluids, and we may start empiric treatment"),
    "constitutional":    ("some general constitutional symptoms we should keep an eye on",
                          "monitoring, bloodwork, and a follow-up visit"),
}

# Clusters eligible for differential-distractor cases
DIFFERENTIAL_CLUSTERS: List[str] = list(DIFFERENTIAL_ASSESSMENT.keys())

NON_COVID_DIFFERENTIALS: Dict[str, list] = {
    "CARDIAC": [
        "Palpitations (heart skipping beats)",
        "Peripheral edema (swollen ankles)",
        "Orthopnea (breathing trouble while lying flat)",
    ],
    "MUSCULOSKELETAL": [
        "Arthralgia (stiff/painful joints)",
        "Lower back strain",
        "Sciatica (shooting leg pain)",
        "Leg pain (calf/thigh aching)",
        "Neck stiffness (can't turn head)",
        "Shoulder pain (difficulty raising arm)",
    ],
    "PSYCH": [
        "Anxiety/Racing thoughts",
        "Sleep disturbance (difficulty falling/staying asleep)",
        "Work-related stress",
    ],
    "ENT": [
        "Tinnitus (ringing in ears)",
        "Otalgia (ear ache/pressure)",
        "Nasal congestion (sinus pressure)",
    ],
    "GASTROINTESTINAL": [
        "Nausea (feeling queasy/sick to stomach)",
        "Vomiting (throwing up/can't keep food down)",
        "Diarrhea (loose/watery stools)",
        "Abdominal pain (stomach cramps/belly ache)",
        "Loss of appetite (no desire to eat)",
        "Constipation (no bowel movement for days)",
        "Heartburn/GERD (burning in chest after eating)",
        "Bloating (belly feels swollen/distended)",
    ],
    "DERMATOLOGICAL": [
        "Rash (red/raised patches on skin)",
        "Pruritus (itchy skin all over)",
        "Hives/Urticaria (itchy welts/bumps)",
        "Skin lesions (sores/blisters on skin)",
        "Petechiae (tiny red/purple spots on skin)",
        "Skin discoloration (unusual color changes)",
    ],
    "NEUROLOGICAL": [
        "Confusion/Altered mental status (feeling out of it)",
        "Numbness/Tingling (pins and needles in hands/feet)",
        "Weakness (no strength in arms/legs)",
        "Photophobia (light sensitivity/hurts to look at bright light)",
        "Neck stiffness with headache (meningeal signs)",
        "Tremor (hands shaking involuntarily)",
        "Seizure (convulsions/blacking out with shaking)",
    ],
    "GENITOURINARY": [
        "Dysuria (burning/pain when urinating)",
        "Urinary frequency (going to bathroom constantly)",
        "Hematuria (blood in urine)",
        "Flank pain (pain in side/lower back near kidneys)",
    ],
    "OPHTHALMOLOGICAL": [
        "Conjunctivitis (red/irritated/pink eyes)",
        "Eye pain (pain behind eyes/hurts to move eyes)",
        "Blurred vision (can't see clearly)",
        "Watery/Tearing eyes (eyes won't stop watering)",
    ],
    "CONSTITUTIONAL": [
        "Night sweats (waking up drenched)",
        "Dehydration (dry mouth/not drinking enough)",
        "Unintentional weight loss (losing weight without trying)",
        "Sleep disturbance (difficulty falling/staying asleep)",
    ],
    "HEMATOLOGIC": [
        "Easy bruising (bruises appearing without injury)",
        "Unusual bleeding (nosebleeds/bleeding gums)",
    ],
}

# ─── Feasibility clusters ───────────────────────────────────────────────────
# Each cluster is a list of symptom strings that commonly co-occur.
# Sampling picks 1-2 primary clusters and draws from them, then optionally
# adds 1-2 symptoms from "adjacent" clusters for realism.

FEASIBILITY_CLUSTERS: Dict[str, List[str]] = {
    "upper_respiratory": [
        "Sore throat (stabbing/razor-blade pain)",
        "Dry, hacking cough",
        "Persistent tickle in throat",
        "Nasal congestion (sinus pressure)",
        "Rhinorrhea (runny nose)",
        "Frequent sneezing",
    ],
    "systemic": [
        "Profound fatigue (bone-tired)",
        "General malaise (feeling wiped out)",
        "Myalgia (deep muscle aches in back/thighs)",
        "Frontal headache (throbbing pressure)",
        "Brain fog (mental cloudiness)",
        "Low-grade fever (100.4F - 102F)",
        "Subjective fever (feeling hot/burning up)",
        "Chills and rigors",
    ],
    "sensory": [
        "Anosmia (total loss of smell)",
        "Ageusia (total loss of taste)",
    ],
    "respiratory_red_flag": [
        "Dyspnea (shortness of breath)",
        "Chest tightness/pressure",
    ],
    # ─── Non-COVID feasibility clusters ──────────────────────────────
    "gastrointestinal": [
        "Nausea (feeling queasy/sick to stomach)",
        "Vomiting (throwing up/can't keep food down)",
        "Diarrhea (loose/watery stools)",
        "Abdominal pain (stomach cramps/belly ache)",
        "Loss of appetite (no desire to eat)",
        "Constipation (no bowel movement for days)",
        "Heartburn/GERD (burning in chest after eating)",
        "Bloating (belly feels swollen/distended)",
    ],
    "dermatological": [
        "Rash (red/raised patches on skin)",
        "Pruritus (itchy skin all over)",
        "Hives/Urticaria (itchy welts/bumps)",
        "Skin lesions (sores/blisters on skin)",
    ],
    "musculoskeletal": [
        "Arthralgia (stiff/painful joints)",
        "Lower back strain",
        "Sciatica (shooting leg pain)",
        "Leg pain (calf/thigh aching)",
        "Neck stiffness (can't turn head)",
        "Shoulder pain (difficulty raising arm)",
    ],
    "cardiac": [
        "Palpitations (heart skipping beats)",
        "Peripheral edema (swollen ankles)",
        "Orthopnea (breathing trouble while lying flat)",
    ],
    "neurological": [
        "Confusion/Altered mental status (feeling out of it)",
        "Numbness/Tingling (pins and needles in hands/feet)",
        "Weakness (no strength in arms/legs)",
        "Photophobia (light sensitivity/hurts to look at bright light)",
        "Tremor (hands shaking involuntarily)",
    ],
    "genitourinary": [
        "Dysuria (burning/pain when urinating)",
        "Urinary frequency (going to bathroom constantly)",
        "Hematuria (blood in urine)",
        "Flank pain (pain in side/lower back near kidneys)",
    ],
    "constitutional": [
        "Night sweats (waking up drenched)",
        "Dehydration (dry mouth/not drinking enough)",
        "Sleep disturbance (difficulty falling/staying asleep)",
    ],
    # ─── Novel-virus hallmark cluster ────────────────────────────────
    "novel_hallmark": [
        "Lymphadenopathy (tender lumps in armpits)",
        "Dysgeusia (metallic/distorted taste — everything tastes wrong)",
        "Hemoptysis (coughing up blood-tinged mucus)",
        "Skin desquamation (peeling skin on palms/fingertips)",
        "Melanonychia (nails turning dark/black)",
    ],
}

# Which clusters are "adjacent" (clinically plausible to co-occur)
CLUSTER_ADJACENCY: Dict[str, List[str]] = {
    "upper_respiratory":    ["systemic", "sensory"],
    "systemic":             ["upper_respiratory", "sensory", "respiratory_red_flag"],
    "sensory":              ["upper_respiratory", "systemic"],
    "respiratory_red_flag": ["systemic", "upper_respiratory"],
    "gastrointestinal":     ["systemic", "constitutional"],
    "dermatological":       ["constitutional", "gastrointestinal"],
    "musculoskeletal":      ["neurological", "constitutional"],
    "cardiac":              ["respiratory_red_flag", "systemic", "neurological"],
    "neurological":         ["musculoskeletal", "systemic", "constitutional"],
    "genitourinary":        ["constitutional", "gastrointestinal"],
    "constitutional":       ["systemic", "gastrointestinal", "neurological"],
    "novel_hallmark":       ["systemic", "upper_respiratory", "sensory"],
}


# ─── Flu-like feasibility clusters ───────────────────────────────────────────

FLU_FEASIBILITY_CLUSTERS: Dict[str, List[str]] = {
    "flu_respiratory": FLU_LIKE_SYMPTOMS["FLU_RESPIRATORY"],
    "flu_systemic":    FLU_LIKE_SYMPTOMS["FLU_SYSTEMIC"],
}

FLU_CLUSTER_ADJACENCY: Dict[str, List[str]] = {
    "flu_respiratory": ["flu_systemic"],
    "flu_systemic":    ["flu_respiratory"],
}


# ─── Flat helpers ────────────────────────────────────────────────────────────

def _all_covid_symptoms() -> List[str]:
    """Return every COVID symptom across all library categories."""
    out: List[str] = []
    for symptoms in SYMPTOM_LIBRARY.values():
        out.extend(symptoms)
    return out


def _all_flu_symptoms() -> List[str]:
    """Return every flu-like symptom."""
    out: List[str] = []
    for symptoms in FLU_LIKE_SYMPTOMS.values():
        out.extend(symptoms)
    return out


def _all_non_covid_symptoms() -> List[str]:
    """Return every NON_COVID_DIFFERENTIAL symptom."""
    out: List[str] = []
    for symptoms in NON_COVID_DIFFERENTIALS.values():
        out.extend(symptoms)
    return out


# Valid case types
CASE_TYPES = ("novel_virus", "flu_like", "differential", "healthy")


# ─── Public sampling API ────────────────────────────────────────────────────

def sample_symptoms(
    rng: random.Random,
    case_type: str = "novel_virus",
    n_present_min: int = 4,
    n_present_max: int = 7,
    n_negated_min: int = 2,
    n_negated_max: int = 4,
    differential_system: Optional[str] = None,
) -> Tuple[List[str], str, List[str], Optional[str]]:
    """Sample a clinically coherent set of present and negated symptoms.

    Parameters
    ----------
    rng : random.Random
    case_type : str
        One of ``novel_virus``, ``flu_like``, ``differential``, ``healthy``.
    n_present_min, n_present_max : int
        Range for number of present symptoms (ignored for ``healthy``).
    n_negated_min, n_negated_max : int
        Range for number of negated symptoms.
    differential_system : str | None
        For ``differential`` cases, force a specific body system
        (e.g. "musculoskeletal"). ``None`` → random selection.

    Returns
    -------
    present : list[str]
        Symptoms the patient HAS.
    chief_complaint : str
        Reason for the visit.
    negated : list[str]
        Symptoms the patient DENIES.
    differential_system : str | None
        Body-system name for ``differential`` cases, else ``None``.
    """
    if case_type not in CASE_TYPES:
        raise ValueError(f"Unknown case_type {case_type!r}. Choose from {CASE_TYPES}")

    if case_type == "novel_virus":
        return _sample_novel_virus(rng, n_present_min, n_present_max, n_negated_min, n_negated_max)
    if case_type == "flu_like":
        return _sample_flu_like(rng, n_present_min, n_present_max, n_negated_min, n_negated_max)
    if case_type == "differential":
        return _sample_differential(rng, n_present_min, n_present_max, n_negated_min, n_negated_max, differential_system=differential_system)
    # healthy
    return _sample_healthy(rng, n_negated_min, n_negated_max)


# ─── Per-case-type samplers ──────────────────────────────────────────────────

def _sample_novel_virus(
    rng: random.Random,
    n_present_min: int, n_present_max: int,
    n_negated_min: int, n_negated_max: int,
) -> Tuple[List[str], str, List[str], None]:
    """COVID-like case with novel-hallmark symptoms mixed in.

    Total present count stays within n_present_min–n_present_max (4-7).
    Of those, 2-4 are novel hallmarks and the rest are standard COVID
    cluster symptoms, so the case reads as COVID-like with unusual extras.
    """
    covid_cluster_names = ["upper_respiratory", "systemic", "sensory", "respiratory_red_flag"]

    # 1. Decide total present count
    n_present = rng.randint(n_present_min, n_present_max)

    # 2. Pick 2-4 novel hallmarks (but leave at least 1 slot for COVID)
    n_novel = rng.randint(2, min(4, n_present - 1))
    n_novel = min(n_novel, len(NOVEL_VIRUS_HALLMARKS))
    novel_symptoms = rng.sample(NOVEL_VIRUS_HALLMARKS, k=n_novel)

    # 3. Fill remaining slots from COVID clusters
    n_covid = n_present - n_novel

    n_primary = rng.choice([1, 2])
    primary_clusters = rng.sample(covid_cluster_names, k=n_primary)

    pool: List[str] = []
    for c in primary_clusters:
        pool.extend(FEASIBILITY_CLUSTERS[c])
    seen = set(novel_symptoms)
    pool = [s for s in pool if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

    n_from_primary = min(len(pool), n_covid)
    covid_present = rng.sample(pool, k=n_from_primary)

    # Fill from adjacent clusters if needed
    if len(covid_present) < n_covid:
        adjacent_names: List[str] = []
        for c in primary_clusters:
            adjacent_names.extend(CLUSTER_ADJACENCY.get(c, []))
        adjacent_names = list(dict.fromkeys(adjacent_names))
        adjacent_names = [a for a in adjacent_names if a not in primary_clusters and a != "novel_hallmark"]

        adj_pool: List[str] = []
        for a in adjacent_names:
            adj_pool.extend(FEASIBILITY_CLUSTERS[a])
        adj_pool = [s for s in adj_pool if s not in seen]

        n_extra = min(len(adj_pool), n_covid - len(covid_present))
        if n_extra > 0:
            covid_present.extend(rng.sample(adj_pool, k=n_extra))

    # 4. Combine and shuffle
    present = covid_present + novel_symptoms
    rng.shuffle(present)

    # 5. Sensory pair rule
    sensory_pair = {
        "Anosmia (total loss of smell)",
        "Ageusia (total loss of taste)",
    }
    present_set = set(present)
    if present_set & sensory_pair and not sensory_pair <= present_set:
        missing = sensory_pair - present_set
        present.extend(missing)

    # 6. Chief complaint — lead with a COVID symptom + a novel symptom
    chief_items: List[str] = []
    if covid_present:
        chief_items.append(covid_present[0])
    if novel_symptoms:
        chief_items.append(novel_symptoms[0])
    if len(chief_items) < 2 and len(present) > len(chief_items):
        chief_items.append(present[-1])
    chief_complaint = " + ".join(chief_items[:3])

    # 7. Negated symptoms — other COVID symptoms the patient does NOT have
    all_covid = _all_covid_symptoms()
    neg_candidates = [s for s in all_covid if s not in set(present)]
    n_negated = min(rng.randint(n_negated_min, n_negated_max), len(neg_candidates))
    negated = rng.sample(neg_candidates, k=n_negated)

    return present, chief_complaint, negated, None


def _sample_flu_like(
    rng: random.Random,
    n_present_min: int, n_present_max: int,
    n_negated_min: int, n_negated_max: int,
) -> Tuple[List[str], str, List[str], None]:
    """Classic influenza — excludes COVID hallmarks (anosmia/ageusia/brain fog)."""
    flu_cluster_names = list(FLU_FEASIBILITY_CLUSTERS.keys())

    # 1. Pick 1-2 primary flu clusters
    n_primary = rng.choice([1, 2])
    primary_clusters = rng.sample(flu_cluster_names, k=n_primary)

    # 2. Pool
    pool: List[str] = []
    for c in primary_clusters:
        pool.extend(FLU_FEASIBILITY_CLUSTERS[c])
    seen = set()
    pool = [s for s in pool if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

    n_present = rng.randint(n_present_min, n_present_max)
    n_from_primary = min(len(pool), n_present)
    present = rng.sample(pool, k=n_from_primary)

    # 3. Fill from adjacent flu clusters if needed
    if len(present) < n_present:
        adj_names: List[str] = []
        for c in primary_clusters:
            adj_names.extend(FLU_CLUSTER_ADJACENCY.get(c, []))
        adj_names = list(dict.fromkeys(adj_names))
        adj_names = [a for a in adj_names if a not in primary_clusters]

        adj_pool: List[str] = []
        for a in adj_names:
            adj_pool.extend(FLU_FEASIBILITY_CLUSTERS[a])
        adj_pool = [s for s in adj_pool if s not in seen]

        n_extra = min(len(adj_pool), n_present - len(present))
        if n_extra > 0:
            present.extend(rng.sample(adj_pool, k=n_extra))

    # 4. Chief complaint
    n_chief = min(rng.randint(2, 3), len(present))
    chief_complaint = " + ".join(present[:n_chief])

    # 5. Negated — clinically relevant respiratory/viral differential symptoms
    present_set = set(present)
    neg_candidates = [s for s in FLU_RELEVANT_NEGATIONS if s not in present_set]

    n_negated = rng.randint(n_negated_min, n_negated_max)
    n_negated = min(n_negated, len(neg_candidates))
    negated = rng.sample(neg_candidates, k=n_negated)

    return present, chief_complaint, negated, None


# Per-system overrides for n_present_min (small symptom pools)
_SYSTEM_PRESENT_MIN: Dict[str, int] = {
    "dermatological": 3,  # pool has only 4 symptoms
}


def _sample_differential(
    rng: random.Random,
    n_present_min: int, n_present_max: int,
    n_negated_min: int, n_negated_max: int,
    differential_system: Optional[str] = None,
) -> Tuple[List[str], str, List[str], str]:
    """Single body-system focus (GI, cardiac, MSK, etc.)."""
    # 1. Pick one non-COVID system cluster (or use caller-specified system)
    system = differential_system if differential_system else rng.choice(DIFFERENTIAL_CLUSTERS)
    pool = list(FEASIBILITY_CLUSTERS[system])

    # 2. Sample present from that cluster (apply per-system min override)
    effective_min = _SYSTEM_PRESENT_MIN.get(system, n_present_min)
    n_present = rng.randint(effective_min, min(n_present_max, len(pool)))
    present = rng.sample(pool, k=n_present)

    # 3. Chief complaint
    n_chief = min(rng.randint(2, 3), len(present))
    chief_complaint = " + ".join(present[:n_chief])

    # 4. Negated — mainly from same cluster remainder + some from other clusters
    present_set = set(present)
    same_cluster_remaining = [s for s in pool if s not in present_set]
    other_pool: List[str] = []
    for cl in DIFFERENTIAL_CLUSTERS:
        if cl != system:
            other_pool.extend(FEASIBILITY_CLUSTERS[cl])
    other_pool = [s for s in other_pool if s not in present_set]

    n_negated = rng.randint(n_negated_min, n_negated_max)
    n_same = min(len(same_cluster_remaining), max(1, n_negated // 2))
    n_other = min(len(other_pool), n_negated - n_same)

    negated: List[str] = []
    if n_same > 0:
        negated.extend(rng.sample(same_cluster_remaining, k=n_same))
    if n_other > 0:
        negated.extend(rng.sample(other_pool, k=n_other))
    rng.shuffle(negated)

    return present, chief_complaint, negated, system


def _sample_healthy(
    rng: random.Random,
    n_negated_min: int, n_negated_max: int,
) -> Tuple[List[str], str, List[str], None]:
    """Healthy checkup — no present symptoms; broad ROS denials."""
    present: List[str] = []
    chief_complaint = "Routine checkup / annual wellness visit"

    # Broad ROS negations: sample from COVID + non-COVID pools so the doctor
    # can run through a realistic system-by-system screen.
    all_symptoms = _all_covid_symptoms() + _all_non_covid_symptoms()
    # Remove duplicates while preserving order
    seen: set = set()
    all_symptoms = [s for s in all_symptoms if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

    n_negated = rng.randint(max(n_negated_min, 6), max(n_negated_max, 10))
    n_negated = min(n_negated, len(all_symptoms))
    negated = rng.sample(all_symptoms, k=n_negated)

    return present, chief_complaint, negated, None
