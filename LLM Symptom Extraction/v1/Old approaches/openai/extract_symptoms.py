"""
Chain-of-Thought symptom extractor — ORIGINAL v1 (OpenAI GPT-4o optimised).

Archived copy of the full two-stage CoT extraction prompt that was designed
for GPT-4o with response_format=json_object.

Two-stage extraction within a single prompt:
  Stage 1 — Score each of the 30 canonical symptoms as 0 / 1 / 2.
  Stage 2 — Identify truly novel symptoms that don't map to any known
            symptom (even via rewording / alias).

Returns a structured dict per transcript.

Backends:
  "openai"  → GPT-4o via OpenAI API
  "ollama"  → Any Ollama model via its OpenAI-compatible endpoint
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from symptom_schema import SYMPTOM_SCHEMA, SYMPTOM_KEYS, NAME_TO_KEY

# ─── Configuration ───────────────────────────────────────────────────────────

# Backend: "openai" or "ollama"
BACKEND = "openai"

# OpenAI settings
OPENAI_MODEL = "gpt-4o"
API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "mistral"   # change to any pulled model

TEMPERATURE = 0.0
MAX_TOKENS = 4000

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if BACKEND == "ollama":
            _client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        else:
            key = API_KEY or os.environ.get("OPENAI_API_KEY", "")
            if not key:
                raise RuntimeError(
                    "No API key. Set API_KEY in extract_symptoms.py or "
                    "the OPENAI_API_KEY environment variable."
                )
            _client = OpenAI(api_key=key)
    return _client


def _get_model() -> str:
    return OLLAMA_MODEL if BACKEND == "ollama" else OPENAI_MODEL


# ─── Prompt construction ─────────────────────────────────────────────────────

def _build_symptom_reference() -> str:
    """Build the symptom checklist block for the prompt, including aliases."""
    lines: List[str] = []
    for i, s in enumerate(SYMPTOM_SCHEMA, 1):
        alias_str = ", ".join(f'"{a}"' for a in s["aliases"][:5])
        lines.append(
            f'{i}. {s["name"]} (key: "{s["key"]}", category: {s["category"]})\n'
            f'   Common phrasings: {alias_str}'
        )
    return "\n".join(lines)


def build_prompt(transcript: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the extraction call.

    This is the ORIGINAL full two-stage CoT prompt, optimised for GPT-4o.
    It includes detailed instructions for:
    - Scoring 30 canonical symptoms (0/1/2)
    - Detecting truly novel symptoms with anti-rewording logic
    - Strict JSON output format with reasoning
    """

    symptom_ref = _build_symptom_reference()

    system_prompt = (
        "You are an expert medical scribe specialised in symptom extraction "
        "from doctor-patient transcripts. You think step-by-step but output "
        "the final result as strict JSON."
    )

    user_prompt = f"""Below is a doctor-patient clinical transcript. Your task has TWO stages.

═══════════════════════════════════════════════════════════════
STAGE 1 — Score Known Symptoms
═══════════════════════════════════════════════════════════════
For each of the 30 canonical symptoms listed below, assign a score:
  0 = Not Mentioned (the symptom never comes up in the conversation)
  1 = Negated (the patient explicitly denies having it, or the doctor
      asks and the patient says no)
  2 = Present (the patient reports having it, or it is clearly present)

IMPORTANT: Patients often describe symptoms in everyday language, NOT
medical terms.  Use the "Common phrasings" to recognise rewordings.
For example, "I feel wiped out" → Fatigue (present, score 2).

Here are the 30 symptoms to score:

{symptom_ref}

═══════════════════════════════════════════════════════════════
STAGE 2 — Detect Novel Symptoms
═══════════════════════════════════════════════════════════════
After scoring all 30 known symptoms, carefully re-read the transcript
and identify ANY additional symptoms the patient mentions that do NOT
fit under any of the 30 symptoms above.

For each candidate novel symptom you find:
1. Check whether it is just a rewording of a known symptom (use the
   aliases and common phrasings above).  If yes → do NOT flag it.
2. Check whether it is a general descriptor rather than a distinct
   symptom (e.g., "feeling bad" is too vague).  If yes → do NOT flag it.
3. Only flag it as novel if it is a genuinely distinct symptom that
   cannot be mapped to any of the 30 known symptoms.

Examples of NOVEL symptoms: "pink feet", "hair loss", "blood in urine"
Examples of NOT novel: "scratchy throat" (= sore_throat), "stomach
turning" (= nausea), "achy" (= myalgia)

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — strict JSON, no markdown
═══════════════════════════════════════════════════════════════
Return a JSON object with exactly these keys:

{{
  "reasoning": "Brief summary of your step-by-step analysis (2-4 sentences).",
  "scores": {{
    "sore_throat": 0,
    "cough": 2,
    ... (all 30 symptom keys, in order)
  }},
  "novel_symptoms": [
    {{
      "symptom": "pink feet",
      "evidence": "Patient says 'my feet have been turning pink'",
      "score": 2
    }}
  ]
}}

Rules:
- "scores" must contain ALL 30 keys, no more, no less.
- "novel_symptoms" is an empty list [] if none are found.
- Each novel symptom must include "symptom" (short name), "evidence"
  (quote from transcript), and "score" (1 for negated, 2 for present).
- Do NOT include anything outside this JSON structure.

═══════════════════════════════════════════════════════════════
TRANSCRIPT
═══════════════════════════════════════════════════════════════
{transcript}
"""
    return system_prompt, user_prompt


# ─── Extraction call ─────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    """Parse JSON from the LLM response, handling markdown fences and extra text."""
    text = raw.strip()
    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first { ... } block (greedy match for outermost braces)
    brace_match = re.search(r"\{", text)
    if brace_match:
        start = brace_match.start()
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")


def extract(transcript: str) -> Dict[str, Any]:
    """Run the CoT extraction on a single transcript.

    Returns
    -------
    dict with keys:
        reasoning  : str
        scores     : dict[str, int]   (30 keys → 0/1/2)
        novel_symptoms : list[dict]   (each has symptom, evidence, score)
    """
    client = _get_client()
    model = _get_model()
    system_prompt, user_prompt = build_prompt(transcript)

    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    # Only OpenAI supports response_format reliably
    if BACKEND == "openai":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)

    raw = response.choices[0].message.content
    data = _parse_json_response(raw)

    # Validate and normalise scores
    scores: Dict[str, int] = {}
    raw_scores = data.get("scores", {})
    for key in SYMPTOM_KEYS:
        val = raw_scores.get(key, 0)
        scores[key] = int(val) if val in (0, 1, 2) else 0

    # Validate novel symptoms
    novel: List[Dict[str, Any]] = []
    for item in data.get("novel_symptoms", []):
        if isinstance(item, dict) and "symptom" in item:
            novel.append({
                "symptom": str(item["symptom"]),
                "evidence": str(item.get("evidence", "")),
                "score": int(item.get("score", 2)),
            })

    return {
        "reasoning": data.get("reasoning", ""),
        "scores": scores,
        "novel_symptoms": novel,
    }


# ─── CLI for single-file testing ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract symptoms from a transcript (ORIGINAL v1 prompt)")
    parser.add_argument("file", help="Path to a .txt transcript")
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai",
                        help="LLM backend (default: openai)")
    parser.add_argument("--model", default="",
                        help="Model name override (e.g., mistral, gemma:2b)")
    args = parser.parse_args()

    BACKEND = args.backend
    if args.model:
        if BACKEND == "ollama":
            OLLAMA_MODEL = args.model
        else:
            OPENAI_MODEL = args.model
    # Reset cached client
    _client = None

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    result = extract(text)
    print(json.dumps(result, indent=2))
