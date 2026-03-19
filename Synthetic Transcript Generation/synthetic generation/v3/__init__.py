"""
Synthetic transcript generation pipeline (v3).

Combines v2's structured symptom sampling with v1's natural one-shot
generation style.  Each transcript samples present COVID symptoms from
SYMPTOM_LIBRARY, negates symptoms from NON_COVID_DIFFERENTIALS, and
generates the entire conversation in a single GPT-4o call.
"""
