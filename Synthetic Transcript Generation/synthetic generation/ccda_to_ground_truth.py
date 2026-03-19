"""
Parse a Synthea C-CDA XML file and export ALL clinical data to a structured
JSON ground-truth file.  Dynamically discovers every section — nothing is
hard-coded.

Usage:
    python ccda_to_ground_truth.py                         # uses default CCDA_PATH
    python ccda_to_ground_truth.py path/to/some_ccda.xml   # explicit path
"""

import xml.etree.ElementTree as ET
import json
import os
import sys
import re
from datetime import datetime

# ─── Default path ────────────────────────────────────────────────────────────
DEFAULT_CCDA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "Synthea CCDAs",
    "Adela471_Danica886_Schmitt836_0c1a1859-29c7-4f11-f21c-20a7099e8613.xml",
)

NS = {"cda": "urn:hl7-org:v3"}
CDA = "{urn:hl7-org:v3}"

# ─── Priority tier definitions ───────────────────────────────────────────────
#   must    = MUST be discussed in the transcript (conditions, meds, allergies)
#   should  = SHOULD come up naturally (social history, immunizations)
#   may     = MAY appear if relevant (vitals, diagnostics)
#   exclude = NOT expected in conversation (procedures, encounters, care plans)

SECTION_PRIORITIES = {
    "conditions":        "must",
    "medications":       "must",
    "allergies":         "must",
    "social_history":    "should",
    "immunizations":     "should",
    "vital_signs":       "may",
    "diagnostic_results":"may",
    "encounters":        "exclude",
    "procedures":        "exclude",
    "care_plans":        "exclude",
    "functional_status": "exclude",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _fmt_date(raw: str) -> str:
    """Turn a CDA timestamp like '20220910062602' into '2022-09-10'."""
    if not raw:
        return ""
    try:
        if len(raw) >= 8:
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
        return raw[:4]
    except Exception:
        return raw


def _collect_table_entries(root, prefix: str) -> list[dict]:
    """Collect all <td> entries whose @ID starts with *prefix*-desc-N.

    Returns a list of dicts with 'description' and 'code' keys, preserving
    the original order in the XML.
    """
    descs: dict[int, str] = {}
    codes: dict[int, str] = {}

    for td in root.iter(f"{CDA}td"):
        td_id = td.get("ID", "")
        if td_id.startswith(f"{prefix}-desc-"):
            idx = int(td_id.split("-")[-1])
            descs[idx] = (td.text or "").strip()
        elif td_id.startswith(f"{prefix}-code-"):
            idx = int(td_id.split("-")[-1])
            codes[idx] = (td.text or "").strip()

    items = []
    for idx in sorted(descs.keys()):
        code_raw = codes.get(idx, "")
        # Parse "http://snomed.info/sct 43878008" → system + code
        code_system, code_val = "", ""
        if code_raw:
            parts = code_raw.rsplit(" ", 1)
            if len(parts) == 2:
                code_system, code_val = parts
            else:
                code_val = code_raw
        items.append({
            "description": descs[idx],
            "code_system": code_system,
            "code": code_val,
            "source": "ccda",
        })
    return items


def _collect_table_entries_with_dates(root, prefix: str) -> list[dict]:
    """Like _collect_table_entries but also grabs Start/Stop dates from the
    adjacent <td> cells in the same table row."""
    # We already have desc/code.  For dates we parse the HTML table rows.
    items = _collect_table_entries(root, prefix)

    # Try to pair dates from the table.  The Synthea tables have the pattern:
    #   <tr>  <td>start</td>  <td>stop</td>  <td ID="prefix-desc-N">…</td>  <td ID="prefix-code-N">…</td>  </tr>
    for tr in root.iter(f"{CDA}tr"):
        tds = list(tr.iter(f"{CDA}td"))
        # Find which row index this belongs to
        for td in tds:
            td_id = td.get("ID", "")
            if td_id.startswith(f"{prefix}-desc-"):
                idx = int(td_id.split("-")[-1]) - 1  # 0-based into items
                if 0 <= idx < len(items):
                    # First two <td> children are start and stop
                    all_tds = [t for t in tr.findall(f"{CDA}td")]
                    if len(all_tds) >= 2:
                        items[idx]["start"] = (all_tds[0].text or "").strip()
                        items[idx]["stop"] = (all_tds[1].text or "").strip()
                break
    return items


# ─── Main parser ─────────────────────────────────────────────────────────────

def parse_ccda(path: str) -> dict:
    """Parse a Synthea C-CDA XML and return a comprehensive dict."""
    tree = ET.parse(path)
    root = tree.getroot()

    gt: dict = {
        "source_file": os.path.basename(path),
        "parsed_at": datetime.now().isoformat(),
        "demographics": {},
        "conditions": [],
        "medications": [],
        "encounters": [],
        "immunizations": [],
        "vital_signs": [],
        "procedures": [],
        "diagnostic_results": [],
        "care_plans": [],
        "functional_status": [],
        "social_history": [],
        "allergies": [],
        "injected_symptoms": [],
    }

    # ── Demographics ──────────────────────────────────────────────────────
    patient = root.find(".//cda:patient", NS)
    if patient is not None:
        given = patient.findtext("cda:name/cda:given", "", NS)
        family = patient.findtext("cda:name/cda:family", "", NS)
        gt["demographics"]["name"] = f"{given} {family}".strip()

        gender_el = patient.find("cda:administrativeGenderCode", NS)
        gt["demographics"]["gender"] = (
            gender_el.get("code", "") if gender_el is not None else ""
        )

        birth_el = patient.find("cda:birthTime", NS)
        if birth_el is not None:
            bv = birth_el.get("value", "")
            gt["demographics"]["birth_date"] = _fmt_date(bv)
            if len(bv) >= 4:
                gt["demographics"]["age"] = 2025 - int(bv[:4])

        race_el = patient.find("cda:raceCode", NS)
        gt["demographics"]["race"] = (
            race_el.get("displayName", "") if race_el is not None else ""
        )

        eth_el = patient.find("cda:ethnicGroupCode", NS)
        gt["demographics"]["ethnicity"] = (
            eth_el.get("displayName", "") if eth_el is not None else ""
        )

    addr = root.find(".//cda:patientRole/cda:addr", NS)
    if addr is not None:
        gt["demographics"]["street"] = addr.findtext("cda:streetAddressLine", "", NS)
        gt["demographics"]["city"] = addr.findtext("cda:city", "", NS)
        gt["demographics"]["state"] = addr.findtext("cda:state", "", NS)
        gt["demographics"]["zip"] = addr.findtext("cda:postalCode", "", NS)

    # ── Table-based sections ──────────────────────────────────────────────
    gt["conditions"] = _collect_table_entries(root, "conditions")
    gt["medications"] = _collect_table_entries_with_dates(root, "medications")
    gt["encounters"] = _collect_table_entries_with_dates(root, "encounters")
    gt["immunizations"] = _collect_table_entries_with_dates(root, "immunizations")
    gt["procedures"] = _collect_table_entries(root, "procedures")
    gt["care_plans"] = _collect_table_entries(root, "careplans")
    gt["functional_status"] = _collect_table_entries(root, "functional-status")

    # ── Vital signs (observations with values) ────────────────────────────
    obs_entries = _collect_table_entries(root, "observations")
    # Enrich with actual values from the structured entries
    for section in root.iter(f"{CDA}section"):
        title_el = section.find(f"{CDA}title")
        if title_el is not None and title_el.text and "Vital" in title_el.text:
            # Parse each organizer → observations
            for organizer in section.iter(f"{CDA}organizer"):
                eff = organizer.find(f".//{CDA}effectiveTime")
                date = ""
                if eff is not None:
                    low = eff.find(f"{CDA}low")
                    date = _fmt_date(low.get("value", "")) if low is not None else _fmt_date(eff.get("value", ""))

                for obs in organizer.iter(f"{CDA}observation"):
                    code_el = obs.find(f"{CDA}code")
                    value_el = obs.find(f"{CDA}value")
                    if code_el is not None and value_el is not None:
                        gt["vital_signs"].append({
                            "description": code_el.get("displayName", ""),
                            "code_system": "http://loinc.org",
                            "code": code_el.get("code", ""),
                            "value": value_el.get("value", ""),
                            "unit": value_el.get("unit", ""),
                            "date": date,
                            "source": "ccda",
                        })
            break

    # ── Diagnostic results (lab panels with observations) ─────────────────
    for section in root.iter(f"{CDA}section"):
        title_el = section.find(f"{CDA}title")
        if title_el is not None and title_el.text and "Diagnostic" in title_el.text:
            for organizer in section.iter(f"{CDA}organizer"):
                panel_code = organizer.find(f"{CDA}code")
                panel_name = panel_code.get("displayName", "") if panel_code is not None else ""

                eff = organizer.find(f"{CDA}effectiveTime")
                date = ""
                if eff is not None:
                    low = eff.find(f"{CDA}low")
                    date = _fmt_date(low.get("value", "")) if low is not None else ""

                observations = []
                for obs in organizer.iter(f"{CDA}observation"):
                    code_el = obs.find(f"{CDA}code")
                    value_el = obs.find(f"{CDA}value")
                    if code_el is not None:
                        entry = {
                            "name": code_el.get("displayName", ""),
                            "code": code_el.get("code", ""),
                        }
                        if value_el is not None:
                            entry["value"] = value_el.get("value", "")
                            entry["unit"] = value_el.get("unit", "")
                        observations.append(entry)

                if observations:
                    gt["diagnostic_results"].append({
                        "panel": panel_name,
                        "date": date,
                        "observations": observations,
                        "source": "ccda",
                    })
            break

    # ── Allergies ─────────────────────────────────────────────────────────
    for section in root.iter(f"{CDA}section"):
        title_el = section.find(f"{CDA}title")
        if title_el is not None and title_el.text and "Allergi" in title_el.text:
            text_el = section.find(f"{CDA}text")
            if text_el is not None and text_el.text:
                gt["allergies"].append({
                    "description": text_el.text.strip(),
                    "source": "ccda",
                })
            break

    # ── Social History ────────────────────────────────────────────────────
    for section in root.iter(f"{CDA}section"):
        title_el = section.find(f"{CDA}title")
        if title_el is not None and title_el.text and "Social" in title_el.text:
            for obs in section.iter(f"{CDA}observation"):
                code_el = obs.find(f"{CDA}code")
                value_el = obs.find(f"{CDA}value")
                if code_el is not None:
                    entry = {
                        "description": code_el.get("displayName", ""),
                        "code": code_el.get("code", ""),
                        "source": "ccda",
                    }
                    if value_el is not None:
                        entry["value"] = value_el.get("displayName", value_el.get("value", ""))
                    gt["social_history"].append(entry)
            break

    # ── Assign priority tiers ─────────────────────────────────────────────
    gt["priority_scheme"] = dict(SECTION_PRIORITIES)
    for section_name, priority in SECTION_PRIORITIES.items():
        if section_name in gt and isinstance(gt[section_name], list):
            for item in gt[section_name]:
                if isinstance(item, dict):
                    # Injected symptoms always get 'must'
                    if item.get("source") == "injected":
                        item["priority"] = "must"
                    elif "priority" not in item:
                        item["priority"] = priority

    return gt


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ccda_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CCDA

    if not os.path.exists(ccda_path):
        print(f"ERROR: File not found: {ccda_path}")
        sys.exit(1)

    print(f"Parsing: {os.path.basename(ccda_path)}")
    gt = parse_ccda(ccda_path)

    # Save JSON
    ccda_filename = os.path.basename(ccda_path)
    base_json_name = ccda_filename.replace(".xml", "_ground_truth.json")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), base_json_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Ground Truth Export Complete")
    print(f"{'='*60}")
    print(f"  Patient:          {gt['demographics'].get('name', '?')}")
    print(f"  Age:              {gt['demographics'].get('age', '?')}")
    print(f"  Conditions:       {len(gt['conditions'])}")
    print(f"  Medications:      {len(gt['medications'])}")
    print(f"  Encounters:       {len(gt['encounters'])}")
    print(f"  Immunizations:    {len(gt['immunizations'])}")
    print(f"  Vital Signs:      {len(gt['vital_signs'])}")
    print(f"  Procedures:       {len(gt['procedures'])}")
    print(f"  Diag. Results:    {len(gt['diagnostic_results'])}")
    print(f"  Care Plans:       {len(gt['care_plans'])}")
    print(f"  Functional Status:{len(gt['functional_status'])}")
    print(f"  Social History:   {len(gt['social_history'])}")
    print(f"  Allergies:        {len(gt['allergies'])}")
    print(f"  Injected Symptoms:{len(gt['injected_symptoms'])}")
    print(f"\n  Saved to: {out_path}")
    print(f"  ✓ Done!")


if __name__ == "__main__":
    main()
