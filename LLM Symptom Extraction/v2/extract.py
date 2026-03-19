"""
Zero-shot LLM symptom extraction using OpenAI GPT-4o.

Reads a doctor-patient transcript, scores each symptom on the known
checklist as -1 (negated), 0 (not present), or 1 (present), and
identifies any additional symptoms not on the checklist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from config import OPENAI_API_KEY, MODEL, TEMPERATURE, MAX_TOKENS, TRANSCRIPT_DIR, OUTPUT_DIR
from symptom_checklist import KNOWN_SYMPTOM_LIST


# ─── OpenAI client ───────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    if OPENAI_API_KEY == "YOUR_KEY_HERE":
        raise RuntimeError(
            "Set your OpenAI API key in config.py or the OPENAI_API_KEY env var."
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
PART 2 — Identify Unlisted Symptoms
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
- Use your medical knowledge to assign a short clinical name to each
  unlisted symptom.
- For each unlisted symptom, provide:
  • "symptom": a short clinical name
  • "status": 1 (present) or -1 (negated/denied)
  • "evidence": a brief quote from the transcript supporting this

If there are no unlisted symptoms, return an empty list.

════════════════════════════════════════════════════════════════
OUTPUT FORMAT — strict JSON, no markdown fences
════════════════════════════════════════════════════════════════
Return a single JSON object with exactly these keys:

{{
  "checklist_scores": {{
    "<symptom string exactly as listed above>": <-1, 0, or 1>,
    ... (all {len(KNOWN_SYMPTOM_LIST)} symptoms)
  }},
  "unlisted_symptoms": [
    {{
      "symptom": "<name>",
      "status": 1,
      "evidence": "<quote>"
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

    raw = response.choices[0].message.content
    result = json.loads(raw)

    return {
        "filename": transcript_path.name,
        "checklist_scores": result.get("checklist_scores", {}),
        "unlisted_symptoms": result.get("unlisted_symptoms", []),
    }


def extract_all(
    transcript_dir: Path | None = None,
    output_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Extract symptoms from all transcripts and save results as JSON."""

    transcript_dir = transcript_dir or TRANSCRIPT_DIR
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    transcripts = sorted(transcript_dir.glob("*.txt"))
    results: List[Dict[str, Any]] = []

    for tp in transcripts:
        print(f"Extracting: {tp.name} ...")
        result = extract_single(tp, client)
        results.append(result)

        out_path = output_dir / tp.with_suffix(".json").name
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  → Saved {out_path.name}  "
              f"({len(result['checklist_scores'])} checklist scores, "
              f"{len(result['unlisted_symptoms'])} unlisted)")

    print(f"\nDone. {len(results)} transcripts processed.")
    return results


if __name__ == "__main__":
    extract_all()
