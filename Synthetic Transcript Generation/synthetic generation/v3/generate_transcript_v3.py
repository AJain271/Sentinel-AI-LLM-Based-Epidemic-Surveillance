"""
v3 One-shot synthetic transcript generator.

Takes a v3 scenario dict (from scenario_builder_v3) and produces a full
doctor–patient transcript in a single GPT-4o call, mimicking the natural
conversational style of real OSCE transcripts (filler words, hesitations,
colloquial phrasing) while following a 7-stage clinical outline.

Usage (standalone):
    python generate_transcript_v3.py --scenario scenario_v3.json [--out transcript.txt]

Requires OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Tuple

# ─── Config ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_TRANSCRIPTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "..", "Clean Transcripts")
STYLE_REFERENCE_FILE = os.path.join(CLEAN_TRANSCRIPTS_DIR, "CAR0001.txt")

DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0.85
DEFAULT_MAX_TOKENS = 6000

# ─── Paste your OpenAI API key here ─────────────────────────────────────────
API_KEY = "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"

# ─── OpenAI client ───────────────────────────────────────────────────────────

def _get_client():
    """Create OpenAI client using the hardcoded API_KEY or OPENAI_API_KEY env var."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.  pip install openai")
        sys.exit(1)

    api_key = API_KEY if API_KEY != "YOUR_API_KEY_HERE" else os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "No API key found. Either set API_KEY in generate_transcript_v3.py "
            "or set the OPENAI_API_KEY environment variable."
        )
    return OpenAI(api_key=api_key)


# ─── Style reference ────────────────────────────────────────────────────────

def _load_style_reference(max_lines: int = 40) -> str:
    """Load the first N lines of a real transcript as a style example."""
    if not os.path.exists(STYLE_REFERENCE_FILE):
        return ""
    with open(STYLE_REFERENCE_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[:max_lines]
    return "".join(lines)


# ─── Prompt construction ────────────────────────────────────────────────────

# Shared format rules used by ALL case types
def _format_rules_block(style_excerpt: str) -> str:
    return f"""═══════════════════════════════════════════════════════════════
FORMAT RULES  (follow exactly)
═══════════════════════════════════════════════════════════════
- Each doctor line starts with "D: " (capital D, colon, space).
- Each patient line starts with "P: " (capital P, colon, space).
- Alternate between D and P lines.
- Put exactly ONE blank line between each speaker turn.
- Do NOT include any headers, titles, labels, stage markers, or metadata.
- Do NOT include stage directions like [coughs] or (pauses).
- The patient should speak like a REAL person — use filler words naturally
  (um, uh, yeah, I mean, you know, like, so, honestly, kinda).
  Vary them; don't over-use the same filler.
- The patient should occasionally stumble, self-correct, or trail off
  mid-sentence just like in a real clinical interview.
- The doctor should be professional but warm, occasionally say "OK" or
  "Alright" to acknowledge, and use brief clarifying follow-ups.

Here is an example of the EXACT style and format to follow:

{style_excerpt}"""


def _symptom_rules_block(present_list: str, negated_list: str) -> str:
    """Symptom rules for any case type that has present symptoms."""
    return f"""═══════════════════════════════════════════════════════════════
SYMPTOM RULES  (CRITICAL — read carefully)
═══════════════════════════════════════════════════════════════
The patient has ONLY the following symptoms.  Do NOT invent, add, or
imply ANY symptoms beyond this exact list:

{present_list}

The parenthetical text tells YOU (the generator) which symptom is meant.
The patient should describe each symptom naturally, the way a real person
would in a doctor's office.  Simple, plain language is fine — a patient
CAN say "my nose is running" or "I've got a runny nose."  What they
should NOT do is repeat the exact clinical phrasing word-for-word from
the list above (e.g. do not say "I have rhinorrhea" or "I'm experiencing
nasal congestion with sinus pressure").  Keep it conversational and
authentic — some patients are vague, some are specific, and not every
description needs to be creative or unusual.

PACING — this is extremely important:
- In the OPENING turn the patient mentions the chief complaint symptoms
  (2-3 concerns that brought them in).  They may hint at feeling
  generally unwell but must NOT list ALL their symptoms yet.
- Each subsequent patient turn reveals AT MOST 1 new symptom, and only
  when the doctor asks a relevant follow-up.
- Some symptoms should emerge only after the doctor specifically asks
  ("Have you noticed any changes in taste or smell?" → patient then
  reveals it).
- The doctor should spend 2-4 exchanges per symptom: asking about onset,
  duration, severity, timing, what makes it worse/better.

The patient does NOT have any of the following related symptoms.
The doctor should ask about each during ROS (these are clinically
related symptoms a doctor would naturally ask about), and the patient
should deny them naturally:

{negated_list}

If the doctor asks about ANY symptom not on either list, the patient
should deny it.  The patient must NEVER volunteer symptoms that are not
on the present list."""


def _user_prompt_demographics(demo: dict) -> list:
    """Return user-prompt lines for patient demographics."""
    lines = []
    lines.append("=== PATIENT DEMOGRAPHICS ===")
    lines.append(f"Name: {demo.get('name', 'Unknown')}")
    lines.append(f"Age: {demo.get('age', 'Unknown')}")
    gender_label = "Female" if demo.get("gender") == "F" else "Male"
    lines.append(f"Gender: {gender_label}")
    if demo.get("race") or demo.get("ethnicity"):
        lines.append(f"Race/Ethnicity: {demo.get('race', '')}, {demo.get('ethnicity', '')}")
    if demo.get("city") or demo.get("state"):
        lines.append(f"Location: {demo.get('city', '')}, {demo.get('state', '')}")
    return lines


def _user_prompt_ccda(noise: dict) -> list:
    """Return user-prompt lines for CCDA background data."""
    lines = []
    lines.append("\n=== CCDA BACKGROUND (weave naturally THROUGHOUT the conversation) ===")
    lines.append("NOTE: Do NOT dump these in one block. Sprinkle them during HPI and context.")
    lines.append("E.g., doctor asks 'Are you on any medications?' mid-HPI → patient lists meds.")
    lines.append("E.g., doctor: 'Any other health conditions I should know about?' → patient mentions a condition.")

    if noise.get("conditions"):
        lines.append("\nPast Medical Conditions (weave into conversation when relevant):")
        for c in noise["conditions"]:
            lines.append(f"  - {c}")

    if noise.get("medications"):
        lines.append("\nCurrent Medications (patient mentions when doctor asks about treatments):")
        for m in noise["medications"]:
            lines.append(f"  - {m}")

    if noise.get("allergies"):
        lines.append("\nAllergies (doctor asks, patient answers):")
        for a in noise["allergies"]:
            lines.append(f"  - {a}")

    if noise.get("social_history"):
        lines.append("\nSocial History (mention naturally if relevant):")
        for s in noise["social_history"][:3]:
            lines.append(f"  - {s}")

    if noise.get("immunizations"):
        lines.append("\nImmunizations (may mention briefly):")
        for im in noise["immunizations"][:3]:
            lines.append(f"  - {im}")

    if noise.get("injected_symptoms"):
        lines.append("\n⚠️ NOVEL/EMERGING SYMPTOMS (patient MUST report these vividly):")
        for s in noise["injected_symptoms"]:
            lines.append(f"  - {s}")

    return lines


# ─── Case-type–specific prompt builders ──────────────────────────────────────

def _build_prompt_novel_virus(scenario: dict, style_excerpt: str) -> tuple:
    """Prompt for novel-virus (COVID-like) cases — original v3 behaviour."""
    demo = scenario["demographics"]
    present = scenario["present_symptoms"]
    negated = scenario["negated_symptoms"]
    chief = scenario["chief_complaint"]
    noise = scenario["ccda_noise"]

    present_list = "\n".join(f"   {i}. {s}" for i, s in enumerate(present, 1))
    negated_list = "\n".join(f"   {i}. {s}" for i, s in enumerate(negated, 1))

    system_prompt = f"""You are a medical transcript generator. Your job is to produce a realistic
doctor–patient clinical encounter transcript for a patient presenting with
symptoms consistent with a viral respiratory illness.

{_format_rules_block(style_excerpt)}

{_symptom_rules_block(present_list, negated_list)}

═══════════════════════════════════════════════════════════════
CLINICAL OUTLINE  (follow this natural flow — do NOT label stages)
═══════════════════════════════════════════════════════════════
1. GREETING (~2-4 lines) — Doctor introduces themselves briefly and asks
   an open-ended question ("What brings you in today?").

2. CHIEF COMPLAINT (~4-6 lines) — The patient's opening centers on:
   "{chief}"
   described in their own everyday words.  The patient mentions these
   2-3 main concerns that brought them in — like a real patient who
   comes in saying "I've had this cough and sore throat and I just
   feel wiped out."  They should NOT list every symptom yet.

3. HISTORY OF PRESENT ILLNESS (~50-70 lines, the BULK of the conversation)
   — The doctor systematically explores each present symptom one by one.
   For EACH symptom spend 2-4 exchanges covering:
     • When did it start?  How long has it lasted?
     • How severe is it? (use scales, comparisons)
     • Is it constant or does it come and go?
     • What makes it worse? What makes it better?
     • Any related symptoms? (this is how the next symptom gets revealed)
   The patient reveals symptoms GRADUALLY — some only when directly asked.
   Weave in occasional CCDA background naturally during HPI:
     • When asking about medications: "Are you taking anything for it?"
       → patient mentions current meds
     • When discussing severity: patient might mention how a pre-existing
       condition interacts ("well, with my anxiety it makes it worse")
     • These details should feel organic, not like a separate checklist.

4. REVIEW OF SYSTEMS (~8-12 lines) — Doctor asks about negated symptoms.
   Space them out — don't rapid-fire.  Mix in brief acknowledgments and
   transitions between questions.  Patient denies naturally with variety
   ("No, nothing like that" / "Nah" / "No, that's been fine").

5. CONTEXT (~10-16 lines) — Doctor weaves in remaining background items
   NOT yet covered (allergies, social history, sick contacts, relevant
   conditions).  This should feel like natural follow-up questions, not
   a data dump.  Example:
     • "Do you smoke at all?" → "No, never."
     • "Any allergies I should know about?"
     • "Have you been around anyone who's been sick?"
   If a CCDA item (like a condition or social detail) already came up
   naturally during HPI, do NOT ask about it again.

6. ASSESSMENT (~4-6 lines) — Doctor gives a preliminary, NON-SPECIFIC
   assessment.  Use phrases like "looks like a viral respiratory
   infection" or "something viral going around."  Do NOT say the word
   "COVID" and do NOT give a definitive diagnosis.

7. PLAN (~6-10 lines) — Rest, fluids, OTC symptom relief, diagnostic
   testing suggestion, return precautions, follow-up scheduling,
   and a warm closing.

═══════════════════════════════════════════════════════════════
LENGTH & PACING RULES  (CRITICAL)
═══════════════════════════════════════════════════════════════
- The conversation MUST be 150–250 lines long (75–125 exchanges).
  A conversation shorter than 150 lines is a FAILURE.
- HPI should be roughly 50-60% of the total conversation.
- The patient mentions ONLY the present symptoms listed above — NO OTHERS.
- Every present symptom must appear in the conversation.
- Every negated symptom must be asked about and denied.
- CCDA background items should be woven throughout naturally — NOT
  dumped in a single section.  Sprinkle them during HPI and context.
- Neither the doctor NOR the patient may say the word "COVID" at any
  point.  The disease is unnamed — treat it as a generic viral
  respiratory illness.  No character should reference COVID, coronavirus,
  or any specific pandemic by name.
- Make the conversation feel REAL — not every question gets a perfect
  answer; the patient can be vague, ask for clarification, or need
  prompting."""

    # User prompt
    user_lines = []
    user_lines.append("Generate a full doctor–patient transcript for the patient below.")
    user_lines.append("Start directly with \"D: \" — no preamble.\n")
    user_lines.extend(_user_prompt_demographics(demo))

    user_lines.append(f"\n=== 🟢 PRESENT SYMPTOMS (patient HAS these — describe vividly) ===")
    user_lines.append(f"Chief complaint: {chief}")
    for i, s in enumerate(present, 1):
        user_lines.append(f"  {i}. {s}")

    user_lines.append(f"\n=== 🔴 NEGATED SYMPTOMS (doctor asks, patient DENIES) ===")
    for i, s in enumerate(negated, 1):
        user_lines.append(f"  {i}. {s}")

    user_lines.extend(_user_prompt_ccda(noise))

    return system_prompt, "\n".join(user_lines)


def _build_prompt_flu_like(scenario: dict, style_excerpt: str) -> tuple:
    """Prompt for flu-like (seasonal influenza) cases."""
    demo = scenario["demographics"]
    present = scenario["present_symptoms"]
    negated = scenario["negated_symptoms"]
    chief = scenario["chief_complaint"]
    noise = scenario["ccda_noise"]

    present_list = "\n".join(f"   {i}. {s}" for i, s in enumerate(present, 1))
    negated_list = "\n".join(f"   {i}. {s}" for i, s in enumerate(negated, 1))

    system_prompt = f"""You are a medical transcript generator. Your job is to produce a realistic
doctor–patient clinical encounter transcript for a patient presenting with
symptoms consistent with a seasonal respiratory illness such as influenza.

{_format_rules_block(style_excerpt)}

{_symptom_rules_block(present_list, negated_list)}

═══════════════════════════════════════════════════════════════
CLINICAL OUTLINE  (follow this natural flow — do NOT label stages)
═══════════════════════════════════════════════════════════════
1. GREETING (~2-4 lines) — Doctor introduces themselves briefly and asks
   an open-ended question ("What brings you in today?").

2. CHIEF COMPLAINT (~4-6 lines) — The patient's opening centers on:
   "{chief}"
   described in their own everyday words.  Emphasise how SUDDENLY the
   symptoms hit — flu often comes on fast ("I was fine yesterday and
   woke up feeling like I got hit by a truck").  Do NOT list every
   symptom yet.

3. HISTORY OF PRESENT ILLNESS (~50-70 lines, the BULK of the conversation)
   — The doctor systematically explores each present symptom one by one.
   For EACH symptom spend 2-4 exchanges covering:
     • When did it start?  How long has it lasted?
     • How severe is it? (use scales, comparisons)
     • Is it constant or does it come and go?
     • What makes it worse? What makes it better?
     • Any related symptoms? (this is how the next symptom gets revealed)
   The patient reveals symptoms GRADUALLY — some only when directly asked.
   Weave in occasional CCDA background naturally during HPI:
     • When asking about medications: "Are you taking anything for it?"
       → patient mentions current meds
     • When discussing severity: patient might mention how a pre-existing
       condition interacts
     • These details should feel organic, not like a separate checklist.

4. REVIEW OF SYSTEMS (~8-12 lines) — Doctor asks about negated symptoms.
   Space them out — don't rapid-fire.  Mix in brief acknowledgments and
   transitions between questions.  Patient denies naturally with variety
   ("No, nothing like that" / "Nah" / "No, that's been fine").

5. CONTEXT (~10-16 lines) — Doctor weaves in remaining background items
   NOT yet covered (allergies, social history, sick contacts, relevant
   conditions).  This should feel like natural follow-up questions, not
   a data dump.  The doctor should ask about flu-shot history and sick
   contacts.
   If a CCDA item already came up naturally during HPI, do NOT repeat it.

6. ASSESSMENT (~4-6 lines) — Doctor gives a preliminary assessment.
   Use phrases like "this looks like influenza" or "a classic flu
   presentation" or "influenza-like illness."  The doctor may mention
   that "there's a lot of flu going around right now."
   Do NOT say the word "COVID" and do NOT give a definitive diagnosis
   without testing.

7. PLAN (~6-10 lines) — Rest, fluids, OTC symptom relief (acetaminophen
   or ibuprofen for fever/aches).  Doctor may discuss a rapid flu test
   or suggest antivirals (e.g., oseltamivir/Tamiflu) if within 48 hrs
   of symptom onset.  Return precautions (worsening breathing, high
   fever not responding to meds), follow-up scheduling, warm closing.

═══════════════════════════════════════════════════════════════
LENGTH & PACING RULES  (CRITICAL)
═══════════════════════════════════════════════════════════════
- The conversation MUST be 150–250 lines long (75–125 exchanges).
  A conversation shorter than 150 lines is a FAILURE.
- HPI should be roughly 50-60% of the total conversation.
- The patient mentions ONLY the present symptoms listed above — NO OTHERS.
- Every present symptom must appear in the conversation.
- Every negated symptom must be asked about and denied.
- CCDA background items should be woven throughout naturally — NOT
  dumped in a single section.  Sprinkle them during HPI and context.
- The doctor must NOT say the word "COVID."
- Make the conversation feel REAL — not every question gets a perfect
  answer; the patient can be vague, ask for clarification, or need
  prompting."""

    # User prompt
    user_lines = []
    user_lines.append("Generate a full doctor–patient transcript for the patient below.")
    user_lines.append("Start directly with \"D: \" — no preamble.\n")
    user_lines.extend(_user_prompt_demographics(demo))

    user_lines.append(f"\n=== 🟢 PRESENT SYMPTOMS (patient HAS these — describe vividly) ===")
    user_lines.append(f"Chief complaint: {chief}")
    for i, s in enumerate(present, 1):
        user_lines.append(f"  {i}. {s}")

    user_lines.append(f"\n=== 🔴 NEGATED SYMPTOMS (doctor asks, patient DENIES) ===")
    for i, s in enumerate(negated, 1):
        user_lines.append(f"  {i}. {s}")

    user_lines.extend(_user_prompt_ccda(noise))

    return system_prompt, "\n".join(user_lines)


def _build_prompt_differential(scenario: dict, style_excerpt: str) -> tuple:
    """Prompt for differential-distractor cases (single body-system focus)."""
    from symptom_library_v3 import DIFFERENTIAL_ASSESSMENT

    demo = scenario["demographics"]
    present = scenario["present_symptoms"]
    negated = scenario["negated_symptoms"]
    chief = scenario["chief_complaint"]
    noise = scenario["ccda_noise"]
    system_name = scenario.get("differential_system", "general")

    assessment_phrase, plan_phrase = DIFFERENTIAL_ASSESSMENT.get(
        system_name,
        ("something we should look into further", "some tests and a follow-up visit"),
    )

    present_list = "\n".join(f"   {i}. {s}" for i, s in enumerate(present, 1))
    negated_list = "\n".join(f"   {i}. {s}" for i, s in enumerate(negated, 1))

    system_prompt = f"""You are a medical transcript generator. Your job is to produce a realistic
doctor–patient clinical encounter transcript for a patient presenting with
{system_name} complaints.

{_format_rules_block(style_excerpt)}

{_symptom_rules_block(present_list, negated_list)}

═══════════════════════════════════════════════════════════════
CLINICAL OUTLINE  (follow this natural flow — do NOT label stages)
═══════════════════════════════════════════════════════════════
1. GREETING (~2-4 lines) — Doctor introduces themselves briefly and asks
   an open-ended question ("What brings you in today?").

2. CHIEF COMPLAINT (~4-6 lines) — The patient's opening centers on:
   "{chief}"
   described in their own everyday words.  The patient mentions these
   2-3 main concerns that brought them in.  They should NOT list every
   symptom yet.

3. HISTORY OF PRESENT ILLNESS (~50-70 lines, the BULK of the conversation)
   — The doctor systematically explores each present symptom one by one.
   Focus on {system_name}-specific questioning:
   For EACH symptom spend 2-4 exchanges covering:
     • When did it start?  How long has it lasted?
     • How severe is it? (use scales, comparisons)
     • Is it constant or does it come and go?
     • What makes it worse? What makes it better?
     • Any related symptoms? (this is how the next symptom gets revealed)
   The patient reveals symptoms GRADUALLY — some only when directly asked.
   Weave in occasional CCDA background naturally during HPI.

4. REVIEW OF SYSTEMS (~8-12 lines) — Doctor asks about negated symptoms
   to rule out related conditions.  Space them out — don't rapid-fire.
   Patient denies naturally with variety.

5. CONTEXT (~10-16 lines) — Doctor weaves in remaining background items
   NOT yet covered (allergies, social history, relevant conditions).
   This should feel like natural follow-up questions, not a data dump.
   If a CCDA item already came up naturally during HPI, do NOT repeat it.

6. ASSESSMENT (~4-6 lines) — Doctor gives a preliminary assessment:
   "{assessment_phrase}."
   The assessment should sound clinical but not overly definitive.
   Do NOT mention any viral respiratory illness, influenza, or COVID.

7. PLAN (~6-10 lines) — {plan_phrase}.
   Return precautions, follow-up scheduling, warm closing.

═══════════════════════════════════════════════════════════════
LENGTH & PACING RULES  (CRITICAL)
═══════════════════════════════════════════════════════════════
- The conversation MUST be 150–250 lines long (75–125 exchanges).
  A conversation shorter than 150 lines is a FAILURE.
- HPI should be roughly 50-60% of the total conversation.
- The patient mentions ONLY the present symptoms listed above — NO OTHERS.
- Every present symptom must appear in the conversation.
- Every negated symptom must be asked about and denied.
- CCDA background items should be woven throughout naturally — NOT
  dumped in a single section.  Sprinkle them during HPI and context.
- Do NOT mention COVID, flu, or any viral respiratory illness.
- Make the conversation feel REAL — not every question gets a perfect
  answer; the patient can be vague, ask for clarification, or need
  prompting."""

    # User prompt
    user_lines = []
    user_lines.append("Generate a full doctor–patient transcript for the patient below.")
    user_lines.append("Start directly with \"D: \" — no preamble.\n")
    user_lines.extend(_user_prompt_demographics(demo))

    user_lines.append(f"\n=== 🟢 PRESENT SYMPTOMS [{system_name.upper()}] (patient HAS these — describe vividly) ===")
    user_lines.append(f"Chief complaint: {chief}")
    for i, s in enumerate(present, 1):
        user_lines.append(f"  {i}. {s}")

    user_lines.append(f"\n=== 🔴 NEGATED SYMPTOMS (doctor asks, patient DENIES) ===")
    for i, s in enumerate(negated, 1):
        user_lines.append(f"  {i}. {s}")

    user_lines.extend(_user_prompt_ccda(noise))

    return system_prompt, "\n".join(user_lines)


def _build_prompt_healthy(scenario: dict, style_excerpt: str) -> tuple:
    """Prompt for routine wellness/annual checkup — no present symptoms."""
    demo = scenario["demographics"]
    negated = scenario["negated_symptoms"]
    noise = scenario["ccda_noise"]

    negated_list = "\n".join(f"   {i}. {s}" for i, s in enumerate(negated, 1))

    system_prompt = f"""You are a medical transcript generator. Your job is to produce a realistic
doctor–patient clinical encounter transcript for a ROUTINE WELLNESS
CHECKUP / ANNUAL PHYSICAL.  The patient is generally healthy and has
NO active complaints.

{_format_rules_block(style_excerpt)}

═══════════════════════════════════════════════════════════════
WELLNESS VISIT RULES  (CRITICAL — read carefully)
═══════════════════════════════════════════════════════════════
This is NOT a sick visit.  The patient feels fine and is here for a
routine checkup.  There are NO present symptoms.

The doctor should go through a brief Review of Systems.  For each of
the following symptoms, the doctor asks and the patient DENIES naturally:

{negated_list}

If the doctor asks about ANY other symptom, the patient should also
deny it.  The patient is healthy and has no complaints.

═══════════════════════════════════════════════════════════════
CLINICAL OUTLINE  (follow this natural flow — do NOT label stages)
═══════════════════════════════════════════════════════════════
1. GREETING (~2-4 lines) — Doctor greets the patient warmly.  Opens with
   something like "Good to see you again" or "How have you been?" rather
   than "What brings you in?" since this is a scheduled wellness visit.

2. OPENING / REASON FOR VISIT (~4-6 lines) — Patient confirms they're
   here for their annual checkup / routine visit.  They feel fine
   overall.  Doctor acknowledges and says they'll go through everything.

3. MEDICAL HISTORY REVIEW (~20-30 lines) — Doctor reviews the patient's
   existing conditions and medications from their chart.  Asks about any
   changes, new concerns, how medications are working, side effects,
   refills needed.  This should feel conversational:
     • "I see you're on [medication] — how's that been working?"
     • "Any changes with your [condition]?"
     • "Last time we talked about [topic] — how's that going?"

4. LIFESTYLE & PREVENTION (~15-20 lines) — Doctor asks about:
     • Exercise habits
     • Diet
     • Sleep quality
     • Stress levels
     • Smoking / alcohol / substance use
     • Preventive screenings (mammogram, colonoscopy, etc. if age-appropriate)
     • Vaccination status
   Patient answers naturally — some areas good, some room for improvement.

5. REVIEW OF SYSTEMS (~15-20 lines) — Doctor runs through a brief
   system-by-system screen.  The patient denies everything.  Doctor
   should cover: respiratory, cardiac, GI, neurological, musculoskeletal,
   and any other relevant systems.  Space questions out naturally — do NOT
   rapid-fire.  Patient denies with variety ("Nope" / "No, that's been
   fine" / "Nothing like that" / "All good there").

6. ASSESSMENT (~4-6 lines) — Doctor summarises: "Everything looks good"
   / "You're in good shape" / "Nothing concerning."  Mentions any lab
   results or vitals briefly if relevant.

7. PLAN (~6-10 lines) — Continue current medications, schedule any due
   screenings, routine bloodwork if appropriate, schedule next annual
   visit, warm closing.

═══════════════════════════════════════════════════════════════
LENGTH & PACING RULES  (CRITICAL)
═══════════════════════════════════════════════════════════════
- The conversation MUST be 100–180 lines long (50–90 exchanges).
  A conversation shorter than 100 lines is a FAILURE.
- Medical history review + lifestyle should be ~50-60% of the conversation.
- Every negated symptom in the list must be asked about and denied.
- CCDA background items (conditions, medications, allergies, social
  history) are the MAIN content of this visit — weave them throughout
  the medical history review and lifestyle sections.
- Do NOT mention COVID, flu, or any illness — the patient is healthy.
- Make the conversation feel REAL and warm — the doctor knows this
  patient and there's a friendly rapport."""

    # User prompt
    user_lines = []
    user_lines.append("Generate a full doctor–patient WELLNESS CHECKUP transcript for the patient below.")
    user_lines.append("Start directly with \"D: \" — no preamble.\n")
    user_lines.extend(_user_prompt_demographics(demo))

    user_lines.append("\nThis is a ROUTINE WELLNESS VISIT. The patient has NO active complaints.")

    user_lines.append(f"\n=== 🔴 REVIEW OF SYSTEMS (doctor asks briefly, patient DENIES all) ===")
    for i, s in enumerate(negated, 1):
        user_lines.append(f"  {i}. {s}")

    user_lines.extend(_user_prompt_ccda(noise))

    return system_prompt, "\n".join(user_lines)


# ─── Main dispatcher ────────────────────────────────────────────────────────

def build_prompt(scenario: Dict[str, Any]) -> Tuple[str, str]:
    """Build (system_prompt, user_prompt) from a v3 scenario dict."""
    style_excerpt = _load_style_reference()
    case_type = scenario.get("case_type", "novel_virus")

    if case_type == "novel_virus":
        return _build_prompt_novel_virus(scenario, style_excerpt)
    if case_type == "flu_like":
        return _build_prompt_flu_like(scenario, style_excerpt)
    if case_type == "differential":
        return _build_prompt_differential(scenario, style_excerpt)
    if case_type == "healthy":
        return _build_prompt_healthy(scenario, style_excerpt)
    # Fallback to novel_virus for unknown types
    return _build_prompt_novel_virus(scenario, style_excerpt)


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_transcript(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Call OpenAI to generate the transcript in one shot."""
    client = _get_client()

    print("Calling OpenAI API to generate synthetic transcript...")
    print("(This may take 30-60 seconds)\n")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    transcript = response.choices[0].message.content.strip()

    usage = response.usage
    token_usage = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    print(
        f"Tokens used — prompt: {usage.prompt_tokens}, "
        f"completion: {usage.completion_tokens}, "
        f"total: {usage.total_tokens}"
    )

    return transcript, token_usage


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a v3 synthetic transcript from a scenario JSON.")
    parser.add_argument("--scenario", required=True, help="Path to scenario_v3.json")
    parser.add_argument("--out", default="", help="Output transcript path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    with open(os.path.abspath(args.scenario), "r", encoding="utf-8") as f:
        scenario = json.load(f)

    system_prompt, user_prompt = build_prompt(scenario)
    transcript, _token_usage = generate_transcript(
        system_prompt, user_prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    out_path = args.out or os.path.abspath(args.scenario).replace(".json", ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
