"""
Chain-of-Thought symptom extractor — supports OpenAI and local (Ollama) models.

Two-stage extraction within a single prompt:
  Stage 1 — Score each of the 30 canonical symptoms as 0 / 1 / 2.
  Stage 2 — Identify truly novel symptoms that don't map to any known
            symptom (even via rewording / alias).

Returns a structured dict per transcript.

Backends:
  "openai"  → GPT-4o via OpenAI API
  "ollama"  → Any Ollama model via its OpenAI-compatible endpoint
             (default: llama3.1:8b)

Original GPT-4o-optimised code archived in extract_symptoms_v1_openai.py.
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
OLLAMA_MODEL = "llama3.1:8b"

TEMPERATURE = 0.0
MAX_TOKENS = 4000

# Retry settings for local models (which may fail to output valid JSON)
LOCAL_MAX_RETRIES = 3

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


def _build_symptom_reference_compact() -> str:
    """Build a compact symptom list for local models — less verbose, easier to follow."""
    lines: List[str] = []
    for i, s in enumerate(SYMPTOM_SCHEMA, 1):
        aliases = ", ".join(s["aliases"][:3])
        lines.append(f'{i}. "{s["key"]}": {s["name"]} (aliases: {aliases})')
    return "\n".join(lines)


def _build_all_keys_json_template() -> str:
    """Build a complete JSON template with all 30 keys set to 0."""
    pairs = [f'    "{k}": 0' for k in SYMPTOM_KEYS]
    return "{\n" + ",\n".join(pairs) + "\n  }"


def build_prompt(transcript: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the extraction call.

    Uses a full two-stage CoT prompt for OpenAI, and a simplified
    directive prompt for local (Ollama) models.
    """

    if BACKEND == "ollama":
        return _build_prompt_local(transcript)
    return _build_prompt_openai(transcript)


def _build_prompt_openai(transcript: str) -> tuple[str, str]:
    """Full two-stage CoT prompt optimised for GPT-4o."""

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


def _build_prompt_local(transcript: str) -> tuple[str, str]:
    """Prompt for local models — kept for API compatibility but extract()
    uses the two-call approach below when BACKEND == 'ollama'."""

    symptom_ref = _build_symptom_reference_compact()
    scores_template = _build_all_keys_json_template()

    system_prompt = (
        "You are a medical symptom extractor. You carefully read EVERY line "
        "of a doctor-patient transcript and score 30 symptoms. "
        "Output ONLY valid JSON."
    )

    user_prompt = f"""Read the transcript and score each of 30 symptoms:
- 0 = NEVER mentioned
- 1 = NEGATED (patient denies it or says no when asked)
- 2 = PRESENT (patient has it)

THE 30 SYMPTOMS:
{symptom_ref}

TRANSCRIPT:
{transcript}

Respond with ONLY this JSON:
{{
  "reasoning": "summary",
  "scores": {scores_template},
  "novel_symptoms": []
}}"""
    return system_prompt, user_prompt


def _extract_evidence_local(client: OpenAI, model: str, transcript: str) -> str:
    """Pass 1 for local models: extract all symptom-related exchanges from transcript."""

    system_prompt = (
        "You are a medical transcript analyst. You read transcripts line by "
        "line and find every place where a symptom or physical complaint is "
        "discussed. You are extremely thorough and never skip exchanges."
    )

    user_prompt = f"""Read this doctor-patient transcript. Find EVERY exchange where the doctor asks about or the patient mentions ANY of these topics:

LOOK FOR ALL OF THESE:
- Pain (chest pain, back pain, leg pain, headache, sore throat, stomach pain)
- Breathing problems (shortness of breath, wheezing, trouble breathing)
- Cough (dry cough, wet cough, coughing blood)
- Fever, chills, night sweats
- Nose symptoms (congestion, stuffy nose, runny nose)
- Fatigue, feeling tired, feeling unwell
- Nausea, vomiting, diarrhea, loss of appetite
- Dizziness, fainting, lightheadedness
- Skin changes (rash, swelling, edema)
- Smell or taste changes
- Vision or hearing changes
- Confusion, brain fog
- Heart racing, palpitations
- Weight loss
- Exposure to sick people
- Sneezing

For EACH exchange found, write:
- What symptom was discussed
- Whether the patient SAID YES (has it) or SAID NO (denied it)
- A brief quote

IMPORTANT: When the doctor asks "Any X?" and the patient says "No" — that counts! Those are DENIED symptoms and you MUST include them.

TRANSCRIPT:
{transcript}

Now list EVERY symptom exchange. There should be at least 10-20 items. Be thorough:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def _score_from_evidence_local(
    client: OpenAI, model: str, evidence: str
) -> dict:
    """Pass 2 for local models: convert evidence list into scored JSON."""

    symptom_ref = _build_symptom_reference_compact()
    scores_template = _build_all_keys_json_template()

    system_prompt = (
        "You convert a symptom evidence list into a JSON object with scores. "
        "Output ONLY valid JSON. No other text."
    )

    user_prompt = f"""Below is a list of symptoms found in a doctor-patient transcript, with whether each was PRESENT or DENIED.

EVIDENCE:
{evidence}

Now score each of the 30 symptoms below:
- 0 = not in the evidence list above (never mentioned)
- 1 = DENIED in the evidence list
- 2 = PRESENT in the evidence list

THE 30 SYMPTOMS:
{symptom_ref}

Also check: did the evidence list mention any symptom that is NOT one of the 30 above? If so, add it to "novel_symptoms".

Respond with ONLY this JSON:

{{
  "reasoning": "Brief summary: which symptoms were present and which were denied",
  "scores": {scores_template},
  "novel_symptoms": []
}}

For novel symptoms use: {{"symptom": "name", "evidence": "quote from evidence", "score": 2}}"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )
    return _parse_json_response(response.choices[0].message.content)


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

    For OpenAI: single-call with full CoT prompt.
    For Ollama: two-call approach —
        Pass 1: extract all symptom evidence as free text
        Pass 2: convert evidence to scored JSON

    Returns
    -------
    dict with keys:
        reasoning  : str
        scores     : dict[str, int]   (30 keys → 0/1/2)
        novel_symptoms : list[dict]   (each has symptom, evidence, score)
    """
    client = _get_client()
    model = _get_model()

    if BACKEND == "ollama":
        # Two-call approach for local models
        max_retries = LOCAL_MAX_RETRIES
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Pass 1: Extract evidence (free text, no JSON constraint)
                evidence = _extract_evidence_local(client, model, transcript)
                # Pass 2: Score from evidence (JSON output)
                data = _score_from_evidence_local(client, model, evidence)
                break
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  [retry {attempt}/{max_retries}] failed, retrying...")
                    continue
                raise ValueError(
                    f"Failed after {max_retries} attempts. Last error: {last_error}"
                )
    else:
        # Single-call for OpenAI
        system_prompt, user_prompt = build_prompt(transcript)
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
        )
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

    parser = argparse.ArgumentParser(description="Extract symptoms from a transcript")
    parser.add_argument("file", help="Path to a .txt transcript")
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai",
                        help="LLM backend (default: openai)")
    parser.add_argument("--model", default="",
                        help="Model name override (e.g., llama3.1:8b, mistral)")
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
