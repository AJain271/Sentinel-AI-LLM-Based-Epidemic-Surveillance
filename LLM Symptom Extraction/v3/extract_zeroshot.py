"""
Zero-shot LLM symptom extraction with quote-binding (v3).

Reads a doctor-patient transcript, scores each symptom on the known
checklist as -1 (negated), 0 (not present), or 1 (present), and uses
quote-binding to identify unmapped symptoms with a clinical term,
definition, verbatim transcript quote, and status.

Changes from v2 extract.py:
- Unmapped symptoms require quote-binding (term + definition + quote + status)
- Field renamed: unlisted_symptoms → unmapped_symptoms
- API key from env var only
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI, RateLimitError, APIError

from config import OPENAI_API_KEY, MODEL, TEMPERATURE, MAX_TOKENS, TRANSCRIPT_DIR, OUTPUT_DIR
from symptom_checklist import KNOWN_SYMPTOM_LIST


# ─── OpenAI client ───────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Set your OpenAI API key in the OPENAI_API_KEY environment variable."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


# ─── Prompt construction ─────────────────────────────────────────────────────

def _build_checklist_block(symptoms: List[str]) -> str:
    """Format the symptom checklist as a numbered list for the prompt."""
    return "\n".join(f"{i}. {s}" for i, s in enumerate(symptoms, 1))


def build_prompt(transcript: str) -> Tuple[str, str]:
    """Return (system_message, user_message) for the extraction call."""

    checklist = _build_checklist_block(KNOWN_SYMPTOM_LIST)

    system_msg = (
        "You are a medical scribe. Your job is to extract symptom information "
        "from a doctor-patient transcript. Respond only with valid JSON."
    )

    user_msg = f"""Below is a doctor-patient clinical transcript. You have two tasks.

════════════════════════════════════════════════════════════════
PART 1 — Score Known Symptoms
════════════════════════════════════════════════════════════════
For EACH of the {len(KNOWN_SYMPTOM_LIST)} symptoms listed below, assign exactly one score:

  -1 = Negated  (the patient explicitly denies having it, or says no when asked)
   0 = Not Present  (the symptom is never mentioned in the conversation)
   1 = Present  (the patient reports having it or it is clearly present)

Patients often use everyday language rather than medical terms. For example,
"my nose won't stop running" means Rhinorrhea is present (score 1).

Here is the symptom checklist:

{checklist}

════════════════════════════════════════════════════════════════
PART 2 — Identify Unmapped Symptoms (Quote-Binding)
════════════════════════════════════════════════════════════════
After scoring all {len(KNOWN_SYMPTOM_LIST)} symptoms above, carefully re-read the
ENTIRE transcript from beginning to end and identify ANY additional
symptoms or clinical findings the patient mentions that are NOT covered
by any item on the checklist above.

Rules:
- Only include genuinely distinct symptoms — NOT rewordings of checklist
  items.  For example, "scratchy throat" is just Sore throat, so do NOT
  list it separately.
- Only include actual symptoms, not general descriptors like "feeling bad".
- For each unmapped symptom, you MUST provide all four fields:
  • "term": a short clinical name for this symptom (e.g. "Epistaxis")
  • "definition": a one-sentence plain-language definition of this term
    (e.g. "Nosebleeds or bleeding from the nasal passages")
  • "quote": the EXACT verbatim text from the transcript where the
    patient describes or mentions this symptom — copy word-for-word
  • "status": 1 (present) or -1 (negated/denied)

If there are no unmapped symptoms, return an empty list.

════════════════════════════════════════════════════════════════
OUTPUT FORMAT — strict JSON, no markdown fences
════════════════════════════════════════════════════════════════
Return a single JSON object with exactly these keys:

{{
  "checklist_scores": {{
    "<symptom string exactly as listed above>": <-1, 0, or 1>,
    ... (all {len(KNOWN_SYMPTOM_LIST)} symptoms)
  }},
  "unmapped_symptoms": [
    {{
      "term": "<clinical name>",
      "definition": "<one-sentence plain-language definition>",
      "quote": "<exact verbatim text from transcript>",
      "status": 1
    }}
  ]
}}

════════════════════════════════════════════════════════════════
TRANSCRIPT
════════════════════════════════════════════════════════════════
{transcript}
"""

    return system_msg, user_msg


# ─── Extraction ──────────────────────────────────────────────────────────────

def extract_single(transcript_path: Path, client: OpenAI) -> Dict[str, Any]:
    """Run extraction on a single transcript file and return structured output."""

    transcript_text = transcript_path.read_text(encoding="utf-8")
    system_msg, user_msg = build_prompt(transcript_text)

    t0 = time.time()
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            break
        except (RateLimitError, APIError) as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt * 10  # 10, 20, 40, 80, 160 seconds
            print(f"  ⚠ Rate limit hit, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
    extraction_time = round(time.time() - t0, 3)

    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    raw = response.choices[0].message.content
    result = json.loads(raw)

    return {
        "filename": transcript_path.name,
        "checklist_scores": result.get("checklist_scores", {}),
        "unmapped_symptoms": result.get("unmapped_symptoms", []),
        "extraction_time_sec": extraction_time,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def extract_all(
    transcript_paths: List[Path] | None = None,
    transcript_dir: Path | None = None,
    output_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Extract symptoms from transcripts and save results as JSON.

    Parameters
    ----------
    transcript_paths : list of Path, optional
        Explicit list of transcript files to process. If provided,
        transcript_dir is ignored.
    transcript_dir : Path, optional
        Directory to glob *.txt from. Used only if transcript_paths is None.
    output_dir : Path, optional
        Where to write output JSONs. Defaults to OUTPUT_DIR from config.
    """
    if transcript_paths is None:
        transcript_dir = transcript_dir or TRANSCRIPT_DIR
        transcript_paths = sorted(transcript_dir.glob("*SYNTHETIC*.txt"))

    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    results: List[Dict[str, Any]] = []

    for tp in transcript_paths:
        print(f"[Zero-Shot] Extracting: {tp.name} ...")
        result = extract_single(tp, client)
        results.append(result)

        out_path = output_dir / tp.with_suffix(".json").name
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        present = sum(1 for v in result["checklist_scores"].values() if v == 1)
        negated = sum(1 for v in result["checklist_scores"].values() if v == -1)
        print(f"  → {present} present, {negated} negated, "
              f"{len(result['unmapped_symptoms'])} unmapped")

    print(f"\n[Zero-Shot] Done. {len(results)} transcripts processed.")
    return results


if __name__ == "__main__":
    extract_all()
