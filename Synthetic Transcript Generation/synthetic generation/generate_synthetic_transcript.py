"""
Generate a synthetic doctor-patient transcript from a C-CDA XML file.
Uses the ChatGPT API to simulate a realistic COVID-19 clinical encounter,
producing output in the same format as the existing OSCE transcripts.
"""

import xml.etree.ElementTree as ET
import os
import json
import sys
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "Synthetic Transcripts")

api_key = os.environ.get("OPENAI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=api_key)

# ─── Parse C-CDA ─────────────────────────────────────────────────────────────

def parse_ccda(path):
    """Extract clinically relevant info from a C-CDA XML file."""
    tree = ET.parse(path)
    root = tree.getroot()

    # Handle CDA namespace
    ns = {'cda': 'urn:hl7-org:v3'}

    info = {}

    # --- Demographics ---
    patient = root.find('.//cda:patient', ns)
    if patient is not None:
        given = patient.findtext('cda:name/cda:given', '', ns)
        family = patient.findtext('cda:name/cda:family', '', ns)
        info['name'] = f"{given} {family}".strip()

        gender_el = patient.find('cda:administrativeGenderCode', ns)
        info['gender'] = gender_el.get('code', '') if gender_el is not None else ''

        birth_el = patient.find('cda:birthTime', ns)
        if birth_el is not None:
            bv = birth_el.get('value', '')
            if len(bv) >= 4:
                birth_year = int(bv[:4])
                info['age'] = 2025 - birth_year
                info['birth_year'] = birth_year

        race_el = patient.find('cda:raceCode', ns)
        info['race'] = race_el.get('displayName', '') if race_el is not None else ''

        eth_el = patient.find('cda:ethnicGroupCode', ns)
        info['ethnicity'] = eth_el.get('displayName', '') if eth_el is not None else ''

    # --- Address ---
    addr = root.find('.//cda:patientRole/cda:addr', ns)
    if addr is not None:
        info['city'] = addr.findtext('cda:city', '', ns)
        info['state'] = addr.findtext('cda:state', '', ns)

    # --- Conditions / Problems ---
    conditions = []
    # Look in the text table for condition descriptions
    for td in root.iter('{urn:hl7-org:v3}td'):
        td_id = td.get('ID', '')
        if td_id.startswith('conditions-desc'):
            text = td.text or ''
            if text:
                conditions.append(text)
    info['conditions'] = conditions

    # --- Medications ---
    medications = []
    for td in root.iter('{urn:hl7-org:v3}td'):
        td_id = td.get('ID', '')
        if td_id.startswith('medications-desc'):
            text = td.text or ''
            if text:
                medications.append(text)
    info['medications'] = medications

    # --- Immunizations ---
    immunizations = []
    for td in root.iter('{urn:hl7-org:v3}td'):
        td_id = td.get('ID', '')
        if td_id.startswith('immunizations-desc'):
            text = td.text or ''
            if text:
                immunizations.append(text)
    info['immunizations'] = immunizations

    return info


# ─── Load from JSON ground truth ─────────────────────────────────────────────

def load_from_json(json_path):
    """Load patient info from a ground_truth.json file instead of parsing CCDA.
    This allows injected symptoms to flow into the prompt automatically.
    Returns items grouped by priority tier."""
    with open(json_path, 'r', encoding='utf-8') as f:
        gt = json.load(f)

    info = {}
    demo = gt.get('demographics', {})
    info['name'] = demo.get('name', 'Unknown')
    info['gender'] = demo.get('gender', '')
    info['age'] = demo.get('age', '')
    info['race'] = demo.get('race', '')
    info['ethnicity'] = demo.get('ethnicity', '')
    info['city'] = demo.get('city', '')
    info['state'] = demo.get('state', '')

    # ── MUST-discuss items ────────────────────────────────────────────
    # Conditions (deduplicate by description)
    seen_conditions = set()
    must_conditions = []
    for c in gt.get('conditions', []):
        desc = c['description']
        if desc not in seen_conditions:
            seen_conditions.add(desc)
            must_conditions.append(desc)
    info['conditions'] = must_conditions

    # Medications (deduplicate)
    seen_meds = set()
    info['medications'] = []
    for m in gt.get('medications', []):
        desc = m['description']
        if desc not in seen_meds:
            seen_meds.add(desc)
            info['medications'].append(desc)

    # Allergies
    info['allergies'] = [a['description'] for a in gt.get('allergies', [])]

    # Injected symptoms (always MUST)
    info['injected_symptoms'] = [
        s['description'] for s in gt.get('injected_symptoms', [])
    ]

    # ── SHOULD-discuss items ──────────────────────────────────────────
    info['social_history'] = []
    for s in gt.get('social_history', []):
        desc = s.get('description', '')
        val = s.get('value', '')
        if desc and val:
            info['social_history'].append(f"{desc}: {val}")
        elif desc:
            info['social_history'].append(desc)

    info['immunizations'] = [i['description'] for i in gt.get('immunizations', [])]

    return info


# ─── Build Prompt ────────────────────────────────────────────────────────────

def build_prompt(patient_info):
    """Construct the system + user prompts for ChatGPT using priority tiers."""

    # ── Build structured patient summary by priority ──────────────────
    summary_lines = []
    summary_lines.append("=== PATIENT DEMOGRAPHICS ===")
    summary_lines.append(f"Name: {patient_info.get('name', 'Unknown')}")
    summary_lines.append(f"Age: {patient_info.get('age', 'Unknown')}")
    summary_lines.append(f"Gender: {'Female' if patient_info.get('gender') == 'F' else 'Male'}")
    summary_lines.append(f"Race/Ethnicity: {patient_info.get('race', '')}, {patient_info.get('ethnicity', '')}")
    summary_lines.append(f"Location: {patient_info.get('city', '')}, {patient_info.get('state', '')}")

    # 🔴 MUST-discuss
    summary_lines.append("\n=== 🔴 MUST DISCUSS (each item MUST be mentioned) ===")

    if patient_info.get('conditions'):
        summary_lines.append("\nMedical Conditions (doctor must ask about EACH, patient confirms or denies):")
        for i, c in enumerate(patient_info['conditions'], 1):
            summary_lines.append(f"  {i}. {c}")

    if patient_info.get('medications'):
        summary_lines.append("\nMedications (doctor must review):")
        for m in patient_info['medications']:
            summary_lines.append(f"  - {m}")

    if patient_info.get('allergies'):
        summary_lines.append("\nAllergies (doctor must ask):")
        for a in patient_info['allergies']:
            summary_lines.append(f"  - {a}")

    if patient_info.get('injected_symptoms'):
        summary_lines.append("\n⚠️ NOVEL/EMERGING SYMPTOMS (patient MUST report these, describe vividly):")
        for s in patient_info['injected_symptoms']:
            summary_lines.append(f"  - {s}")

    # 🟡 SHOULD-discuss
    summary_lines.append("\n=== 🟡 SHOULD DISCUSS (include naturally) ===")

    if patient_info.get('social_history'):
        summary_lines.append("\nSocial History:")
        for s in patient_info['social_history']:
            summary_lines.append(f"  - {s}")

    if patient_info.get('immunizations'):
        summary_lines.append("\nImmunization History:")
        for i in patient_info['immunizations'][:5]:
            summary_lines.append(f"  - {i}")

    patient_summary = "\n".join(summary_lines)

    # Load a sample transcript for style reference
    sample_path = os.path.join(SCRIPT_DIR, "..", "..", "Clean Transcripts", "CAR0001.txt")
    sample_excerpt = ""
    if os.path.exists(sample_path):
        with open(sample_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:40]  # first ~20 exchanges
            sample_excerpt = "".join(lines)

    system_prompt = f"""You are a medical transcript generator. Your job is to produce a realistic
doctor-patient clinical encounter transcript for a patient presenting with COVID-19 symptoms.

FORMAT RULES (CRITICAL — follow exactly):
- Each doctor line starts with "D: " (capital D, colon, space)
- Each patient line starts with "P: " (capital P, colon, space)
- Alternate between D and P lines
- Put exactly one blank line between each speaker turn
- Do NOT include any headers, titles, labels, or metadata
- Do NOT include stage directions like [coughs] or (pauses)
- Use natural, conversational language with filler words (um, uh, yeah, I mean)
- The patient should speak like a real person, not a textbook

Here is an example of the exact style to follow:

{sample_excerpt}

CLINICAL CONTENT RULES (CRITICAL — follow the priority system):

🔴 MUST-DISCUSS ITEMS:
- The patient is presenting with COVID-19
- EVERY medical condition listed in the MUST section must be mentioned in the
  conversation. The doctor should ask about each, and the patient should either
  confirm they have/had it OR deny it ("No, I haven't had that").
- ALL current medications must be reviewed by the doctor.
- Allergies must be explicitly asked about and confirmed/denied.
- Any NOVEL/EMERGING SYMPTOMS listed must be described vividly by the patient.
  The patient should bring these up and describe them in detail using everyday
  language (not medical terminology).

🟡 SHOULD-DISCUSS ITEMS:
- Social history (smoking, alcohol, living situation, work) should come up
  naturally during the history-taking portion.
- Immunization status may be briefly mentioned.

🟢 COVID SYMPTOMS TO INCLUDE:
- Include realistic COVID symptoms: fever, dry cough, fatigue, loss of taste/smell,
  body aches, headache, shortness of breath, sore throat
- The doctor should systematically ask about symptom onset, duration, severity
- Include questions about exposure history (contact with sick people)
- Some COVID symptoms should be NEGATED (patient says they don't have them)
  e.g. "No, I haven't had any chest pain" or "No rashes or anything like that"

GENERAL:
- The conversation should be 150-250 lines long (75-125 exchanges)
- Include family history briefly
- Make it realistic — not every symptom is present"""

    user_prompt = f"""Generate a doctor-patient transcript for the following patient presenting
with COVID-19 symptoms. Use the priority system below — every 🔴 MUST item needs
to appear in the conversation. Use their real demographics and medical history
to make the conversation authentic.

{patient_summary}

Generate the full transcript now. Start directly with "D: " — no preamble."""

    return system_prompt, user_prompt


# ─── Generate Transcript ────────────────────────────────────────────────────

def generate_transcript(system_prompt, user_prompt):
    """Call ChatGPT to generate the transcript."""
    print("Calling ChatGPT API to generate synthetic transcript...")
    print("(This may take 30-60 seconds)\n")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.85,
        max_tokens=4000
    )

    transcript = response.choices[0].message.content.strip()

    # Print token usage
    usage = response.usage
    print(f"Tokens used — prompt: {usage.prompt_tokens}, "
          f"completion: {usage.completion_tokens}, "
          f"total: {usage.total_tokens}")

    return transcript


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Synthetic Transcript Generator (COVID-19)")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccda", required=False, default=os.path.join(SCRIPT_DIR, "..", "Synthea CCDAs", "Adela471_Danica886_Schmitt836_0c1a1859-29c7-4f11-f21c-20a7099e8613.xml"))
    args = parser.parse_args()
    ccda_path = args.ccda

    base_dir = os.path.dirname(os.path.abspath(ccda_path))
    ccda_filename = os.path.basename(ccda_path)
    base_json_name = ccda_filename.replace(".xml", "_ground_truth.json")
    modified_json_name = ccda_filename.replace(".xml", "_modified_ground_truth.json")
    
    modified_json = os.path.join(SCRIPT_DIR, modified_json_name)
    base_json = os.path.join(SCRIPT_DIR, base_json_name)

    if os.path.exists(modified_json):
        print(f"\n1. Loading from modified ground truth: {os.path.basename(modified_json)}")
        patient_info = load_from_json(modified_json)
        print(f"   (includes {len(patient_info.get('injected_symptoms', []))} injected symptoms)")
    elif os.path.exists(base_json):
        print(f"\n1. Loading from ground truth: {os.path.basename(base_json)}")
        patient_info = load_from_json(base_json)
    else:
        print(f"\n1. Parsing C-CDA: {os.path.basename(ccda_path)}")
        patient_info = parse_ccda(ccda_path)

    print(f"   Patient: {patient_info.get('name', '?')}, "
          f"Age: {patient_info.get('age', '?')}, "
          f"Gender: {patient_info.get('gender', '?')}")
    print(f"   Conditions: {len(patient_info.get('conditions', []))} found")
    print(f"   Medications: {len(patient_info.get('medications', []))} found")

    # 2. Build prompt
    print("\n2. Building prompt...")
    system_prompt, user_prompt = build_prompt(patient_info)

    # 3. Generate
    print("\n3. Generating transcript...")
    transcript = generate_transcript(system_prompt, user_prompt)

    # 4. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Use a COVID-specific filename
    patient_name = patient_info.get('name', 'Unknown').replace(' ', '_')
    output_file = os.path.join(OUTPUT_DIR, f"COVID_SYNTHETIC_{patient_name}.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(transcript)

    print(f"\n4. Saved to: {output_file}")

    # 5. Quick stats
    lines = transcript.strip().split('\n')
    d_lines = sum(1 for l in lines if l.startswith('D:'))
    p_lines = sum(1 for l in lines if l.startswith('P:'))
    print(f"\n   Stats: {d_lines} doctor lines, {p_lines} patient lines, "
          f"{len(lines)} total lines")

    # 6. Preview
    print("\n" + "=" * 60)
    print("  PREVIEW (first 20 lines)")
    print("=" * 60)
    for line in lines[:20]:
        print(line)
    print("...")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
