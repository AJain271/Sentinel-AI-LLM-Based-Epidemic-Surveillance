"""
Chain-of-Thought LLM symptom extraction using OpenAI GPT-4o.

Single-pass approach: the LLM must provide brief reasoning for each
symptom score, then output a structured JSON. This encourages more
careful disambiguation (e.g., hemoptysis vs productive cough).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from config import OPENAI_API_KEY, MODEL, TEMPERATURE, TRANSCRIPT_DIR
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
    return "\n".join(f"{i}. {s}" for i, s in enumerate(symptoms, 1))


def build_prompt(transcript: str) -> Tuple[str, str]:
    """Return (system_message, user_message) for the CoT extraction call."""

    checklist = _build_checklist_block(KNOWN_SYMPTOM_LIST)

    system_msg = (
        "You are a meticulous medical scribe. For every clinical judgment "
        "you make, you first state your reasoning in 1-2 sentences, then "
        "assign a score. Respond only with valid JSON."
    )

    user_msg = f"""Below is a doctor-patient clinical transcript. You have two tasks.

════════════════════════════════════════════════════════════════
PART 1 — Score Known Symptoms with Reasoning
════════════════════════════════════════════════════════════════
For EACH of the {len(KNOWN_SYMPTOM_LIST)} symptoms listed below, you must:
1. State 1-2 sentences of reasoning — cite what the patient said (or
   didn't say) that led to your decision.
2. Assign exactly one score:
     -1 = Negated  (patient explicitly denies it)
      0 = Not Present  (never mentioned)
      1 = Present  (patient reports having it)

IMPORTANT SCORING GUIDELINES:
• Be precise — score a checklist symptom as present ONLY if the patient's
  description genuinely matches that specific symptom. If what the patient
  describes is related but clinically distinct, score the checklist item
  as 0 (Not Present) and capture the actual finding as an unlisted symptom.
• A symptom is only NEGATED (-1) if the patient explicitly denies it
  when asked, or volunteers that they do NOT have it. If the symptom
  is simply never discussed, score it 0 (Not Present).

Here is the symptom checklist:

{checklist}

════════════════════════════════════════════════════════════════
PART 2 — Identify Unlisted Symptoms with Reasoning
════════════════════════════════════════════════════════════════
After scoring all checklist symptoms, carefully re-read the ENTIRE
transcript and identify ANY additional symptoms the patient mentions
that are NOT covered by any checklist item.

For each unlisted symptom found, provide:
• "symptom": a short clinical name
• "reasoning": why you believe this is a genuinely distinct symptom
  not covered by any checklist item
• "status": 1 (present) or -1 (negated)
• "evidence": a brief quote from the transcript

If there are no unlisted symptoms, return an empty list.

════════════════════════════════════════════════════════════════
OUTPUT FORMAT — strict JSON, no markdown fences
════════════════════════════════════════════════════════════════
Return a single JSON object:

{{
  "checklist_results": [
    {{
      "symptom": "<symptom string exactly as listed>",
      "reasoning": "<1-2 sentence justification>",
      "score": <-1, 0, or 1>
    }},
    ... (all {len(KNOWN_SYMPTOM_LIST)} symptoms, in order)
  ],
  "unlisted_symptoms": [
    {{
      "symptom": "<name>",
      "reasoning": "<why this is distinct>",
      "status": <1 or -1>,
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

def _post_process(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert CoT output to the standard schema used by evaluate.py.

    The CoT model outputs checklist_results (list of dicts with reasoning),
    but evaluate.py expects checklist_scores (dict mapping symptom → score).
    We produce both so the full reasoning is preserved alongside the
    evaluation-compatible format.
    """
    checklist_results = raw_result.get("checklist_results", [])

    # Build the flat scores dict
    checklist_scores: Dict[str, int] = {}
    for item in checklist_results:
        symptom = item.get("symptom", "")
        score = max(-1, min(1, int(item.get("score", 0))))
        checklist_scores[symptom] = score

    # Ensure every known symptom is present (default 0 if LLM skipped it)
    for symptom in KNOWN_SYMPTOM_LIST:
        if symptom not in checklist_scores:
            checklist_scores[symptom] = 0

    return {
        "checklist_scores": checklist_scores,
        "checklist_results_cot": checklist_results,  # preserve reasoning
        "unlisted_symptoms": raw_result.get("unlisted_symptoms", []),
    }


def extract_single(transcript_path: Path, client: OpenAI) -> Dict[str, Any]:
    """Run CoT extraction on a single transcript file."""

    transcript_text = transcript_path.read_text(encoding="utf-8")
    system_msg, user_msg = build_prompt(transcript_text)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=8192,  # larger to accommodate reasoning text
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)
    processed = _post_process(result)

    return {
        "filename": transcript_path.name,
        **processed,
    }


def extract_all(
    transcript_dir: Path | None = None,
    output_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    """Extract symptoms from all transcripts using CoT and save results."""

    transcript_dir = transcript_dir or TRANSCRIPT_DIR
    output_dir = output_dir or (Path(__file__).resolve().parent / "output_cot")
    output_dir.mkdir(parents=True, exist_ok=True)

    client = _get_client()
    transcripts = sorted(transcript_dir.glob("*.txt"))
    results: List[Dict[str, Any]] = []

    for tp in transcripts:
        print(f"[CoT] Extracting: {tp.name} ...")
        result = extract_single(tp, client)
        results.append(result)

        out_path = output_dir / tp.with_suffix(".json").name
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        present = sum(1 for v in result["checklist_scores"].values() if v == 1)
        negated = sum(1 for v in result["checklist_scores"].values() if v == -1)
        print(f"  → {present} present, {negated} negated, "
              f"{len(result['unlisted_symptoms'])} unlisted")

    print(f"\n[CoT] Done. {len(results)} transcripts processed.")
    return results


if __name__ == "__main__":
    extract_all()
