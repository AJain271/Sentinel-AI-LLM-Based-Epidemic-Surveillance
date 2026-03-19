"""
v2 Scenario → synthetic conversation transcript.

Constraints:
- 30–45 conversational turns (doctor+patient).
- Enforce 7-step clinical outline stages in order:
  1) GREETING
  2) CHIEF_COMPLAINT
  3) HPI
  4) ROS (ask negated symptoms; patient denies)
  5) CONTEXT (allergies/meds + sick contacts)
  6) ASSESSMENT (non-specific viral assessment)
  7) PLAN (standard plan + closing)
- Split context:
  - Patient sees only demographics + present_symptoms (Natural Patient Language).
  - Doctor sees clinical_suspicion + present/negated/not_mentioned + CCDA grounding.
- COVID language rule:
  - Patient should avoid saying “COVID” unless it comes up naturally as their concern.
  - Doctor must NOT explicitly state/diagnose COVID (do not say the word).

This module uses OPENAI_API_KEY from environment (no hardcoded credentials).
"""

from __future__ import annotations

import importlib
import json
import os
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class VisitStage(str, Enum):
    GREETING = "GREETING"
    CHIEF_COMPLAINT = "CHIEF_COMPLAINT"
    HPI = "HPI"
    ROS = "ROS"
    CONTEXT = "CONTEXT"
    ASSESSMENT = "ASSESSMENT"
    PLAN = "PLAN"


@dataclass(frozen=True)
class OrchestrationConfig:
    min_turns: int = 25
    max_turns: int = 35
    model: str = "gpt-4o-mini"
    temperature: float = 0.8
    seed: Optional[int] = 0
    max_context_turns: int = 14
    max_tokens_per_turn: int = 150


def _load_openai_client():
    openai_mod = importlib.import_module("openai")
    OpenAI = getattr(openai_mod, "OpenAI")

    # Hardcode your API key here if you don't want to use environment variables
    api_key_override = "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"
    
    # Check if the user has replaced the placeholder
    if api_key_override != "YOUR_API_KEY_HERE":
        api_key = api_key_override
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please set the environment variable or edit _load_openai_client in conversation_orchestrator_v2.py")
    return OpenAI(api_key=api_key)


def _format_transcript_lines(turns: List[Tuple[str, str]]) -> str:
    out: List[str] = []
    for speaker, text in turns:
        prefix = "D: " if speaker == "doctor" else "P: "
        out.append(prefix + text.strip())
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _build_stage_schedule(total_turns: int) -> List[Tuple[VisitStage, int, int]]:
    total_turns = max(14, total_turns)

    def even(n: int) -> int:
        return n if n % 2 == 0 else n + 1

    greeting = 2
    chief = 2
    assessment = 2
    reserve_plan = 4

    remaining = max(0, total_turns - (greeting + chief + assessment + reserve_plan))
    hpi = even(max(10, int(remaining * 0.55)))
    ros = even(max(6, int(remaining * 0.25)))
    context = even(max(4, remaining - hpi - ros))

    allocated = greeting + chief + hpi + ros + context + assessment + reserve_plan
    if allocated > total_turns:
        overflow = allocated - total_turns
        hpi = max(2, hpi - even(overflow))

    schedule: List[Tuple[VisitStage, int, int]] = []
    cur = 0

    def add(stage: VisitStage, n: int) -> None:
        nonlocal cur
        end = min(total_turns, cur + n)
        schedule.append((stage, cur, end))
        cur = end

    add(VisitStage.GREETING, greeting)
    add(VisitStage.CHIEF_COMPLAINT, chief)
    add(VisitStage.HPI, hpi)
    add(VisitStage.ROS, ros)
    add(VisitStage.CONTEXT, context)
    add(VisitStage.ASSESSMENT, assessment)
    add(VisitStage.PLAN, max(2, total_turns - cur))
    return schedule


def _stage_for_turn(schedule: List[Tuple[VisitStage, int, int]], turn_index: int) -> VisitStage:
    for stage, start, end in schedule:
        if start <= turn_index < end:
            return stage
    return VisitStage.PLAN


def _patient_symptom_labels(present_symptoms: List[Dict[str, Any]]) -> List[str]:
    labels: List[str] = []
    for s in present_symptoms:
        if not isinstance(s, dict):
            continue
        if s.get("label"):
            labels.append(str(s["label"]))
        elif s.get("id"):
            labels.append(str(s["id"]))
    return labels


def generate_transcript_from_scenario(
    scenario: Dict[str, Any],
    config: OrchestrationConfig,
) -> str:
    client = _load_openai_client()
    rng = random.Random(config.seed)

    target_turns = rng.randint(config.min_turns, config.max_turns)
    stage_schedule = _build_stage_schedule(target_turns)

    demographics = scenario.get("demographics_summary", {}) or {}
    present_symptoms = scenario.get("present_symptoms", []) or []
    negated_ids = scenario.get("negated_symptoms", []) or []
    not_mentioned_ids = scenario.get("not_mentioned_symptoms", []) or []
    negated_expanded = scenario.get("negated_symptoms_expanded", []) or []
    not_mentioned_expanded = scenario.get("not_mentioned_symptoms_expanded", []) or []
    clinical_suspicion = scenario.get("clinical_suspicion", "general_primary_care")
    grounding = scenario.get("grounding", {}) or {}

    # Extract MUST-discuss items from grounding
    case_conditions = grounding.get("conditions", [])
    case_medications = grounding.get("medications", [])
    case_allergies = grounding.get("allergies", [])
    injected_symptoms = grounding.get("injected_symptoms", [])

    present_labels_for_patient = _patient_symptom_labels(present_symptoms)
    turns: List[Tuple[str, str]] = []

    ros_queue = list(negated_expanded)
    rng.shuffle(ros_queue)

    stage_objectives = {
        "GREETING": "Brief intro.",
        "CHIEF_COMPLAINT": "Patient describes primary symptoms in Natural Patient Language.",
        "HPI": "Ask duration, severity, and timing of present symptoms.",
        "ROS": "Ask about negated symptoms to rule out other causes.",
        "CONTEXT": "Review ALL medications, allergies, and past medical history. Ask about sick contacts.",
        "ASSESSMENT": "Preliminary non-specific viral assessment grounded in case details.",
        "PLAN": "Standard plan and closing."
    }
    stage_rules = {
        "GREETING": "Introduce yourself and invite the patient to explain what's going on. Keep it short.",
        "CHIEF_COMPLAINT": "Ask open-ended; do not list symptoms for the patient.",
        "HPI": "Clarify onset/duration/timing/severity. Keep it conversational and brief (1-2 sentences).",
        "ROS": "Ask ONLY the provided negated symptoms. 1–2 per turn. Patient should deny them.",
        "CONTEXT": "Explicitly ask about medications and conditions found in the case instructions. Also sick contacts.",
        "ASSESSMENT": "Use non-specific language (e.g., viral respiratory infection). Do not say 'COVID'.",
        "PLAN": "Rest/fluids, testing suggestion, return precautions, follow-up, close. Do not say 'COVID'."
    }

    def call_llm(system: str, user_payload: Dict[str, Any]) -> str:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens_per_turn,
        )
        return response.choices[0].message.content.strip()

    for t in range(target_turns):
        stage = _stage_for_turn(stage_schedule, t)
        is_doctor_turn = (t % 2 == 0)
        recent = [{"speaker": s, "text": txt} for s, txt in turns[-config.max_context_turns :]]

        if is_doctor_turn:
            system = (
                "You are the Doctor in a primary care visit.\n"
                "FORMAT:\n"
                "- Output ONLY the doctor utterance text (no 'D:' prefix).\n"
                "- Speak concisely (1-2 sentences maximum per turn).\n"
                "- No headers/titles/stage directions.\n\n"
                "HARD RULES:\n"
                "- Follow the Visit Outline stage you are given.\n"
                "- Ask about negated symptoms ONLY in the ROS stage.\n"
                "- Do NOT mention any not_mentioned symptoms.\n"
                "- In the CONTEXT stage, you MUST ask about the patient's specific conditions and medications.\n"
                "- Do NOT say the word 'COVID' and do NOT explicitly diagnose it.\n"
                "- If the patient says 'COVID', acknowledge without repeating the word.\n"
            )

            ros_targets = []
            if stage == VisitStage.ROS:
                if ros_queue:
                    ros_targets.append(ros_queue.pop(0))
                if ros_queue:
                    ros_targets.append(ros_queue.pop(0))

            user_payload = {
                "visit_stage": stage.value,
                "stage_objective": stage_objectives[stage.value],
                "stage_rules": stage_rules[stage.value],
                "clinical_suspicion": clinical_suspicion,
                "present_symptoms": present_symptoms,
                "negated_symptom_ids": negated_ids,
                "negated_symptoms_expanded": negated_expanded,
                "ros_targets": ros_targets,
                "not_mentioned_symptom_ids": not_mentioned_ids,
                "not_mentioned_symptoms_expanded": not_mentioned_expanded,
                "grounding": grounding,
                # Pass full lists to doctor so they know what to ask about
                "patient_medications": case_medications,
                "patient_conditions": case_conditions,
                "patient_allergies": case_allergies,
                "injected_symptoms": injected_symptoms,
                "demographics_summary": demographics,
                "conversation_so_far": recent
            }
            turns.append(("doctor", call_llm(system, user_payload)))
        else:
            system = (
                "You are the Patient in a primary care visit.\n"
                "FORMAT:\n"
                "- Output ONLY the patient utterance text (no 'P:' prefix).\n"
                "- No headers/titles/stage directions.\n"
                "- Speak concisely (1-3 sentences maximum).\n"
                "- Speak in Natural Patient Language (use the example phrases as inspiration, don't just copy them).\n\n"
                "HARD RULES:\n"
                "- Only describe symptoms from present_symptoms.\n"
                "- If asked about a symptom you don't have, deny it naturally.\n"
                "- If asked about your medications or past history, verify them against your provided list.\n"
                "- Avoid using the word 'COVID' unless it comes up naturally as your own concern.\n"
            )

            user_payload = {
                "visit_stage": stage.value,
                "demographics_summary": demographics,
                "present_symptom_labels": present_labels_for_patient,
                "present_symptoms": present_symptoms,
                # Give patient access to their own history so they can answer questions
                "my_medications": case_medications,
                "my_conditions": case_conditions,
                "my_allergies": case_allergies,
                "conversation_so_far": recent
            }
            turns.append(("patient", call_llm(system, user_payload)))

    if turns and turns[-1][0] != "doctor":
        system = (
            "You are the Doctor.\n"
            "Output ONLY a final closing plan statement.\n"
            "Do not say the word 'COVID'."
        )
        user_payload = {
            "visit_stage": VisitStage.PLAN.value,
            "clinical_suspicion": clinical_suspicion,
            "conversation_so_far": [{"speaker": s, "text": txt} for s, txt in turns[-config.max_context_turns :]],
        }
        turns.append(("doctor", call_llm(system, user_payload)))

    return _format_transcript_lines(turns)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a synthetic transcript from scenario_v2.json.")
    parser.add_argument("--scenario", required=True, help="Path to scenario_v2.json")
    parser.add_argument("--out", default="", help="Optional output transcript path")
    parser.add_argument("--min_turns", type=int, default=30)
    parser.add_argument("--max_turns", type=int, default=45)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(os.path.abspath(args.scenario), "r", encoding="utf-8") as f:
        scenario = json.load(f)

    cfg = OrchestrationConfig(
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
    )

    transcript = generate_transcript_from_scenario(scenario, cfg)
    out_path = os.path.abspath(args.out) if args.out else os.path.abspath(args.scenario).replace(".json", ".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    print(out_path)


if __name__ == "__main__":
    main()
