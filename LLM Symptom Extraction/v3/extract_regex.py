"""
Lightweight regex / keyword-based symptom extraction (v3).

For each known symptom on the checklist, parses keywords from the
symptom string and searches the transcript sentence-by-sentence.
Uses a negation window to distinguish present vs denied symptoms.

Known limitation: cannot detect unmapped/novel symptoms (returns empty list).

Reuses core logic from v2/extract_regex.py.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config import TRANSCRIPT_DIR
from symptom_checklist import KNOWN_SYMPTOM_LIST


# ─── Negation helpers ────────────────────────────────────────────────────────

_NEGATION_TOKENS = {
    "no", "not", "don't", "dont", "doesn't", "doesnt",
    "haven't", "havent", "hasn't", "hasnt", "never",
    "denied", "denies", "deny", "without", "none", "nor",
    "didn't", "didnt", "wasn't", "wasnt", "weren't", "werent",
    "negative", "absent",
}

_NEGATION_WINDOW = 6  # words before match to check for negation


def _has_negation(text: str, match_start: int) -> bool:
    """Check if a negation token appears within the window before *match_start*."""
    prefix = text[:match_start].lower()
    words = prefix.split()
    window = words[-_NEGATION_WINDOW:] if len(words) >= _NEGATION_WINDOW else words
    return bool(set(window) & _NEGATION_TOKENS)


# ─── Keyword extraction from symptom strings ────────────────────────────────

_STOPWORDS = {
    "a", "an", "the", "in", "of", "on", "to", "for", "with", "and",
    "or", "is", "it", "that", "can", "t", "like", "up", "out", "when",
    "at", "by", "from", "its", "s", "won", "your", "you", "my",
}


def _extract_keywords(symptom_str: str) -> List[str]:
    """Extract meaningful keywords from a symptom string.

    E.g. "Rhinorrhea (runny nose)" → ["rhinorrhea", "runny", "nose"]
    """
    cleaned = re.sub(r"[()—,\.\"\']", " ", symptom_str)
    tokens = cleaned.lower().split()
    keywords = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return keywords


def _build_symptom_patterns(symptoms: List[str]) -> List[Tuple[str, List[re.Pattern]]]:
    """Build regex patterns for each symptom."""
    result = []
    for symptom in symptoms:
        keywords = _extract_keywords(symptom)
        patterns = []
        for kw in keywords:
            patterns.append(re.compile(r"\b" + re.escape(kw) + r"\w{0,3}\b", re.IGNORECASE))
        result.append((symptom, patterns))
    return result


# ─── Core extraction logic ───────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split transcript into rough sentence-like chunks."""
    lines = text.split("\n")
    sentences = []
    for line in lines:
        parts = re.split(r"(?<=[.!?])\s+", line.strip())
        sentences.extend(p for p in parts if p.strip())
    return sentences


def _score_symptom(
    sentences: List[str],
    full_text_lower: str,
    patterns: List[re.Pattern],
) -> int:
    """Score a single symptom: 1 (present), -1 (negated), 0 (not found)."""
    found_any = False
    all_negated = True

    for sentence in sentences:
        sentence_lower = sentence.lower()
        for pat in patterns:
            m = pat.search(sentence_lower)
            if m:
                found_any = True
                if not _has_negation(sentence_lower, m.start()):
                    all_negated = False
                break

    if not found_any:
        return 0
    return -1 if all_negated else 1


# ─── Public API ──────────────────────────────────────────────────────────────

def extract_single(transcript_path: Path) -> Dict[str, Any]:
    """Run regex extraction on a single transcript file."""
    t0 = time.time()

    text = transcript_path.read_text(encoding="utf-8")
    sentences = _split_sentences(text)
    full_lower = text.lower()

    symptom_patterns = _build_symptom_patterns(KNOWN_SYMPTOM_LIST)

    checklist_scores: Dict[str, int] = {}
    for symptom, patterns in symptom_patterns:
        checklist_scores[symptom] = _score_symptom(sentences, full_lower, patterns)

    extraction_time = round(time.time() - t0, 3)

    return {
        "filename": transcript_path.name,
        "checklist_scores": checklist_scores,
        "unmapped_symptoms": [],  # regex cannot detect truly unknown symptoms
        "extraction_time_sec": extraction_time,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def extract_all(
    transcript_paths: List[Path] | None = None,
    transcript_dir: Path | None = None,
    output_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Extract symptoms from transcripts using regex and save results."""

    if transcript_paths is None:
        transcript_dir = transcript_dir or TRANSCRIPT_DIR
        transcript_paths = sorted(transcript_dir.glob("*SYNTHETIC*.txt"))

    output_dir = output_dir or (Path(__file__).resolve().parent / "output_regex")
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for tp in transcript_paths:
        print(f"[Regex] Extracting: {tp.name} ...")
        result = extract_single(tp)
        results.append(result)

        out_path = output_dir / tp.with_suffix(".json").name
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        present = sum(1 for v in result["checklist_scores"].values() if v == 1)
        negated = sum(1 for v in result["checklist_scores"].values() if v == -1)
        print(f"  → {present} present, {negated} negated, "
              f"{len(result['unmapped_symptoms'])} unmapped")

    print(f"\n[Regex] Done. {len(results)} transcripts processed.")
    return results


if __name__ == "__main__":
    extract_all()
