"""
v2 Ground truth JSON + symptom catalog → scenario JSON.

Scenario outputs:
- present_symptoms (must be mentioned; CCDA-derived + injected)
- negated_symptoms (3–5; asked/denied in ROS)
- not_mentioned_symptoms (must not be mentioned anywhere)
- clinical_suspicion, is_covid_case
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class SymptomDef:
    id: str
    label: str
    category: str
    keywords: Tuple[str, ...]
    patient_phrases: Tuple[str, ...]
    covid_salient: bool = False


def _load_symptom_catalog(catalog_path: str) -> List[SymptomDef]:
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    symptoms: List[SymptomDef] = []
    for s in data.get("symptoms", []):
        symptoms.append(
            SymptomDef(
                id=s["id"],
                label=s.get("label", s["id"]),
                category=s.get("category", "unspecified"),
                keywords=tuple(s.get("keywords", [])),
                patient_phrases=tuple(s.get("patient_phrases", [])),
                covid_salient=bool(s.get("covid_salient", False)),
            )
        )
    return symptoms


def _normalize_text(x: str) -> str:
    x = x.lower()
    x = re.sub(r"[^a-z0-9\s\-/]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _collect_ccda_evidence_text(gt: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ["conditions", "vital_signs", "diagnostic_results", "functional_status", "care_plans", "procedures"]:
        items = gt.get(key, [])
        if not items:
            continue
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    for k in ["description", "panel", "name", "value", "unit", "code"]:
                        if k in item and item[k]:
                            parts.append(str(item[k]))
                    if "observations" in item and isinstance(item["observations"], list):
                        for obs in item["observations"]:
                            if isinstance(obs, dict):
                                parts.append(str(obs.get("name", "")))
                                parts.append(str(obs.get("code", "")))
                                parts.append(str(obs.get("value", "")))
                else:
                    parts.append(str(item))
    return _normalize_text(" ".join([p for p in parts if p]))


def _match_present_symptoms(
    symptoms: List[SymptomDef],
    evidence_text: str,
) -> Set[str]:
    present: Set[str] = set()
    for s in symptoms:
        keys = [k for k in s.keywords if k]
        if not keys:
            continue
        for kw in keys:
            kw_norm = _normalize_text(kw)
            if kw_norm and kw_norm in evidence_text:
                present.add(s.id)
                break
    return present


def _get_injected_symptoms(gt: Dict[str, Any]) -> List[str]:
    injected = []
    for s in gt.get("injected_symptoms", []) or []:
        if isinstance(s, dict) and s.get("description"):
            injected.append(str(s["description"]))
        elif isinstance(s, str):
            injected.append(s)
    return injected


def _map_injected_to_catalog(
    injected: List[str],
    symptoms: List[SymptomDef],
) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """
    Returns (mapped_symptom_ids, free_text_symptoms).
    If an injected symptom cannot be mapped, keep it as free-text present symptom.
    """
    mapped: Set[str] = set()
    free_text: List[Dict[str, Any]] = []
    sym_by_id = {s.id: s for s in symptoms}
    for raw in injected:
        txt = _normalize_text(raw)
        matched_id: Optional[str] = None
        for s in symptoms:
            if s.label and _normalize_text(s.label) == txt:
                matched_id = s.id
                break
            for kw in s.keywords:
                if kw and _normalize_text(kw) in txt:
                    matched_id = s.id
                    break
            if matched_id:
                break
        if matched_id and matched_id in sym_by_id:
            mapped.add(matched_id)
        else:
            free_text.append({"label": raw, "source": "injected_free_text"})
    return mapped, free_text


def _choose_negated(
    symptoms: List[SymptomDef],
    exclude_ids: Set[str],
    clinical_suspicion: str,
    rng: random.Random,
    n_min: int = 3,
    n_max: int = 5,
) -> List[str]:
    candidates = [s for s in symptoms if s.id not in exclude_ids]
    if not candidates:
        return []

    suspicion = (clinical_suspicion or "").lower()
    preferred_categories: Set[str]
    red_flag_ids: Set[str]

    if "resp" in suspicion or "viral" in suspicion:
        preferred_categories = {"respiratory", "cardiopulmonary", "constitutional", "ent"}
        red_flag_ids = {"chest_pain", "shortness_of_breath", "syncope", "confusion"}
    else:
        preferred_categories = {"constitutional", "gastrointestinal", "respiratory", "neurologic"}
        red_flag_ids = {"chest_pain", "syncope", "confusion", "hematuria"}

    scored: List[Tuple[str, float]] = []
    for s in candidates:
        w = 1.0
        if s.category in preferred_categories:
            w += 2.0
        if s.id in red_flag_ids:
            w += 2.0
        scored.append((s.id, w))

    k = min(len(scored), rng.randint(n_min, n_max))
    chosen: List[str] = []
    pool = scored[:]

    # Weighted sampling without replacement
    for _ in range(k):
        total_w = sum(w for _, w in pool)
        if total_w <= 0:
            break
        r = rng.random() * total_w
        upto = 0.0
        pick_idx = 0
        for i, (_, w) in enumerate(pool):
            upto += w
            if upto >= r:
                pick_idx = i
                break
        sid, _ = pool.pop(pick_idx)
        chosen.append(sid)

    return chosen


def _infer_is_covid_case_from_gt(gt: Dict[str, Any]) -> bool:
    conditions = gt.get("conditions", []) or []
    hay: List[str] = []
    for c in conditions:
        if isinstance(c, dict):
            hay.append(str(c.get("description", "")).lower())
            hay.append(str(c.get("code", "")).lower())
        elif isinstance(c, str):
            hay.append(c.lower())
    joined = " ".join(hay)
    return ("covid" in joined) or ("sars-cov" in joined) or ("sarscov" in joined) or ("cov-2" in joined)


def _infer_clinical_suspicion_from_gt(is_covid_case: bool) -> str:
    if is_covid_case:
        return "viral_respiratory"
    return "general_primary_care"


def _choose_not_mentioned(
    symptoms: List[SymptomDef],
    exclude_ids: Set[str],
    rng: random.Random,
    n_min: int = 5,
    n_max: int = 10,
) -> List[str]:
    candidates = [s.id for s in symptoms if s.id not in exclude_ids]
    if not candidates:
        return []
    k = min(len(candidates), rng.randint(n_min, n_max))
    return rng.sample(candidates, k=k)


def build_scenario(
    ground_truth_path: str,
    catalog_path: str,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    symptoms = _load_symptom_catalog(catalog_path)
    rng = random.Random(seed)

    case_meta = gt.get("case_metadata", {}) or {}
    if "is_covid_case" in case_meta:
        is_covid_case = bool(case_meta.get("is_covid_case", False))
    else:
        is_covid_case = _infer_is_covid_case_from_gt(gt)

    if "clinical_suspicion" in case_meta and case_meta.get("clinical_suspicion"):
        clinical_suspicion = str(case_meta.get("clinical_suspicion"))
    else:
        clinical_suspicion = _infer_clinical_suspicion_from_gt(is_covid_case)

    evidence_text = _collect_ccda_evidence_text(gt)
    present_from_ccda = _match_present_symptoms(symptoms, evidence_text)

    injected = _get_injected_symptoms(gt)
    mapped_injected_ids, injected_free_text = _map_injected_to_catalog(injected, symptoms)

    present_ids = set(present_from_ccda) | set(mapped_injected_ids)

    # If CCDA evidence doesn't surface any symptom terms, seed a minimal set
    # based on the clinical suspicion so conversations are not empty.
    seeded_from_syndrome: Set[str] = set()
    if not present_ids and clinical_suspicion == "viral_respiratory":
        for sid in ["cough", "fatigue", "fever"]:
            if any(s.id == sid for s in symptoms):
                seeded_from_syndrome.add(sid)
        present_ids |= seeded_from_syndrome

    negated_ids = _choose_negated(
        symptoms=symptoms,
        exclude_ids=present_ids,
        clinical_suspicion=clinical_suspicion,
        rng=rng,
    )

    exclude_for_not_mentioned = set(present_ids) | set(negated_ids)
    not_mentioned_ids = _choose_not_mentioned(
        symptoms=symptoms,
        exclude_ids=exclude_for_not_mentioned,
        rng=rng,
    )

    sym_index = {s.id: s for s in symptoms}
    ccda_ids = set(present_from_ccda)
    injected_ids = set(mapped_injected_ids)
    present_symptoms: List[Dict[str, Any]] = []
    for sid in sorted(present_ids):
        s = sym_index.get(sid)
        if not s:
            continue
        if sid in injected_ids:
            source = "injected_mapped"
        elif sid in ccda_ids:
            source = "ccda_evidence"
        elif sid in seeded_from_syndrome:
            source = "syndrome_seed"
        else:
            source = "other"
        present_symptoms.append(
            {
                "id": s.id,
                "label": s.label,
                "category": s.category,
                "patient_phrases": list(s.patient_phrases),
                "source": source,
            }
        )
    for ft in injected_free_text:
        present_symptoms.append({"id": None, **ft})

    demographics = gt.get("demographics", {}) or {}
    demo_summary = {
        "name": demographics.get("name", "Unknown"),
        "age": demographics.get("age", ""),
        "gender": demographics.get("gender", ""),
        "city": demographics.get("city", ""),
        "state": demographics.get("state", ""),
    }

    grounding = {
        "conditions": [c.get("description", "") for c in gt.get("conditions", []) if isinstance(c, dict)],
        "medications": [m.get("description", "") for m in gt.get("medications", []) if isinstance(m, dict)],
        "allergies": [a.get("description", "") for a in gt.get("allergies", []) if isinstance(a, dict)],
    }

    negated_expanded = [
        {
            "id": sid,
            "label": sym_index[sid].label if sid in sym_index else sid,
            "category": sym_index[sid].category if sid in sym_index else "unspecified",
        }
        for sid in negated_ids
    ]
    not_mentioned_expanded = [
        {
            "id": sid,
            "label": sym_index[sid].label if sid in sym_index else sid,
            "category": sym_index[sid].category if sid in sym_index else "unspecified",
        }
        for sid in not_mentioned_ids
    ]

    scenario = {
        "scenario_version": "v2",
        "source_ground_truth": os.path.basename(ground_truth_path),
        "is_covid_case": is_covid_case,
        "clinical_suspicion": clinical_suspicion,
        "demographics_summary": demo_summary,
        "present_symptoms": present_symptoms,
        "negated_symptoms": negated_ids,
        "not_mentioned_symptoms": not_mentioned_ids,
        "negated_symptoms_expanded": negated_expanded,
        "not_mentioned_symptoms_expanded": not_mentioned_expanded,
        "present_from_ccda_ids": sorted(ccda_ids),
        "present_from_injected_ids": sorted(injected_ids),
        "present_from_syndrome_seed_ids": sorted(seeded_from_syndrome),
        "grounding": grounding,
        "seed": seed,
    }
    return scenario


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate scenario_v2.json from ground truth + symptom catalog.")
    parser.add_argument("--ground_truth", required=True, help="Path to *_ground_truth_v2.json (or compatible) file")
    parser.add_argument("--catalog", required=True, help="Path to symptom_catalog_v2.json")
    parser.add_argument("--out", default="", help="Optional output scenario path")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 uses deterministic seed=0)")
    args = parser.parse_args()

    scenario = build_scenario(
        ground_truth_path=os.path.abspath(args.ground_truth),
        catalog_path=os.path.abspath(args.catalog),
        seed=args.seed,
    )

    out_path = os.path.abspath(args.out) if args.out else os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.basename(args.ground_truth).replace(".json", "_scenario_v2.json"),
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)

    print(out_path)


if __name__ == "__main__":
    main()

