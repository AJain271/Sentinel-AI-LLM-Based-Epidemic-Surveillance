"""
CCDA comparison viewer (v2-aware).

Features:
- Takes a v2 (or v1) ground-truth JSON and an optional explicit transcript path.
- Highlights CCDA items that DO / DO NOT appear in the transcript.
- Shows injected symptoms.
- If a v2 scenario file exists, shows present / negated / not_mentioned symptom lists.

Usage:
    python ccda_comparison_viewer_v2.py path/to/ground_truth_v2.json
    python ccda_comparison_viewer_v2.py path/to/ground_truth_v2.json path/to/transcript.txt
"""

from __future__ import annotations

import json
import os
import sys
import re
import webbrowser
from html import escape
from typing import Any, Dict, List


DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(DEFAULT_DIR, ".."))
TRANSCRIPT_DIR_V1 = os.path.join(ROOT_DIR, "Synthetic Transcript Generation", "Synthetic Transcripts")
TRANSCRIPT_DIR_V2 = os.path.join(
    ROOT_DIR, "Synthetic Transcript Generation", "Synthetic Transcripts", "Generated Transcripts v2"
)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transcript_by_patient(patient_name: str) -> str:
    """Fallback logic to find a transcript based on patient name."""
    safe_name = patient_name.replace(" ", "_")
    candidates = []

    if os.path.isdir(TRANSCRIPT_DIR_V2):
        for fname in os.listdir(TRANSCRIPT_DIR_V2):
            if safe_name in fname or patient_name.split()[0] in fname:
                candidates.append(os.path.join(TRANSCRIPT_DIR_V2, fname))

    if os.path.isdir(TRANSCRIPT_DIR_V1):
        for fname in os.listdir(TRANSCRIPT_DIR_V1):
            if safe_name in fname or patient_name.split()[0] in fname:
                candidates.append(os.path.join(TRANSCRIPT_DIR_V1, fname))

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return ""


def check_match(description: str, transcript_lower: str) -> bool:
    """Check if a ground-truth item is mentioned in the transcript."""
    desc_lower = description.lower()

    clean = re.sub(
        r"\s*\(disorder\)|\(finding\)|\(situation\)|\(procedure\)|\(regime/therapy\)|\(observable entity\)",
        "",
        desc_lower,
    ).strip()

    if clean and clean in transcript_lower:
        return True

    words = [w for w in clean.split() if len(w) > 3]
    if len(words) >= 2:
        matches = sum(1 for w in words if w in transcript_lower)
        if matches >= len(words) * 0.6:
            return True

    mapping = {
        "gingivitis": ["gingivitis", "gum disease", "gum"],
        "contact dermatitis": ["dermatitis", "rash", "skin"],
        "streptococcal sore throat": ["strep throat", "strep", "sore throat"],
        "viral sinusitis": ["sinusitis", "sinus"],
        "acute viral pharyngitis": ["pharyngitis", "sore throat"],
        "anemia": ["anemia", "anemic", "iron"],
        "concussion injury of brain": ["concussion", "head injury"],
        "stress": ["stress", "stressed"],
        "unhealthy alcohol drinking behavior": ["alcohol", "drinking", "drink"],
        "full-time employment": ["full-time", "full time", "work"],
        "part-time employment": ["part-time", "part time"],
        "normal pregnancy": ["pregnancy", "pregnant"],
        "limited social contact": ["social", "isolated"],
    }
    for key, synonyms in mapping.items():
        if key in clean:
            if any(s in transcript_lower for s in synonyms):
                return True

    return False


def build_html(gt: Dict[str, Any], transcript: str, scenario: Dict[str, Any] | None) -> str:
    patient_name = gt.get("demographics", {}).get("name", "Unknown")
    transcript_lower = transcript.lower()

    def _section_html(title: str, items: List[Dict[str, Any]], show_dates: bool = False) -> str:
        if not items:
            return f'<div class="section"><h3>{escape(title)}</h3><p class="empty">No data</p></div>'

        rows = []
        matched = 0
        for item in items:
            desc = item.get("description", "")
            if not desc:
                continue
            source = item.get("source", "ccda")
            is_match = check_match(desc, transcript_lower) if transcript else False
            if is_match:
                matched += 1

            priority = item.get("priority", "none")
            pri_badge = ""
            if priority != "none":
                pri_badge = f'<span class="badge priority-{priority}">{priority.upper()}</span>'

            badge = ""
            row_class = ""
            if source == "injected":
                badge = '<span class="badge injected">INJECTED</span>'
                row_class = "item-injected" if is_match else "item-miss-critical"
            elif is_match:
                badge = '<span class="badge match">✅ Found</span>'
                row_class = "item-match"
            elif priority == "must":
                badge = '<span class="badge miss-critical">❌ MISSING (REQUIRED)</span>'
                row_class = "item-miss-critical"
            elif priority in ["should", "may", "none"]:
                badge = '<span class="badge miss-minor">➖ Not mentioned</span>'
                row_class = "item-miss-minor"
            elif priority == "exclude":
                badge = '<span class="badge exclude">🚫 Excluded background</span>'
                row_class = "item-exclude"

            code = item.get("code", "")
            code_display = f'<span class="code">{escape(code)}</span>' if code else ""

            date_info = ""
            if show_dates:
                start = item.get("start", "")
                stop = item.get("stop", "")
                if start:
                    date_info = f'<span class="date">{escape(str(start)[:10])}'
                    if stop and stop != start:
                        date_info += f' → {escape(str(stop)[:10])}'
                    date_info += "</span>"

            extra = ""
            if "value" in item:
                extra = f'<span class="value">{escape(str(item["value"]))} {escape(item.get("unit", ""))}</span>'

            rows.append(
                f"""
                <div class="item {row_class}">
                    <div class="item-main">
                        {pri_badge}
                        {badge}
                        <span class="desc">{escape(desc)}</span>
                        {code_display}
                    </div>
                    <div class="item-meta">{date_info} {extra}</div>
                </div>
                """
            )

        coverage = f"{matched}/{len(items)}" if transcript else "N/A"
        return f"""
            <div class="section">
                <h3>{escape(title)} <span class="count">({len(items)} items, {coverage} matched)</span></h3>
                {''.join(rows)}
            </div>
        """

    def _demographics_html(demo: Dict[str, Any]) -> str:
        fields = [
            ("Name", demo.get("name", "?")),
            ("Age", str(demo.get("age", "?"))),
            ("Gender", "Female" if demo.get("gender") == "F" else "Male"),
            ("Race", demo.get("race", "?")),
            ("Ethnicity", demo.get("ethnicity", "?")),
            ("City", demo.get("city", "?")),
            ("State", demo.get("state", "?")),
        ]
        rows = "".join(f'<tr><td class="field-label">{escape(k)}</td><td>{escape(v)}</td></tr>' for k, v in fields)
        return f"""
            <div class="section">
                <h3>Demographics</h3>
                <table class="demo-table">{rows}</table>
            </div>
        """

    def _scenario_html(scn: Dict[str, Any] | None) -> str:
        if not scn:
            return ""

        def labellist(objs: List[Dict[str, Any]], key: str = "label") -> str:
            labels = [str(o.get(key, o.get("id", ""))) for o in objs if (isinstance(o, dict) and (o.get(key) or o.get("id")))]
            if not labels:
                return "<em>None</em>"
            return ", ".join(escape(l) for l in labels)

        pres = scn.get("present_symptoms", [])
        neg = scn.get("negated_symptoms_expanded", [])
        nm = scn.get("not_mentioned_symptoms_expanded", [])

        present_labels = labellist(pres, "label")
        neg_labels = labellist(neg, "label")
        nm_labels = labellist(nm, "label")

        clinical_suspicion = scn.get("clinical_suspicion", "general_primary_care")
        is_covid_case = scn.get("is_covid_case", False)

        return f"""
            <div class="section scenario-section">
                <h3>Scenario Summary (v2)</h3>
                <p class="scenario-meta">
                    <strong>Clinical suspicion:</strong> {escape(str(clinical_suspicion))}<br/>
                    <strong>is_covid_case flag:</strong> {escape(str(is_covid_case))}
                </p>
                <p><strong>Present symptoms (must appear):</strong> {present_labels}</p>
                <p><strong>Negated symptoms (asked &amp; denied):</strong> {neg_labels}</p>
                <p><strong>Not-mentioned symptoms (forbidden):</strong> {nm_labels}</p>
            </div>
        """

    def _transcript_html(text: str) -> str:
        if not text:
            return '<p class="empty">No transcript loaded.</p>'
        lines = text.strip().split("\n")
        html_lines: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("D:"):
                html_lines.append(
                    f'<div class="line doctor"><span class="speaker doc">D:</span>{escape(line[2:].lstrip())}</div>'
                )
            elif line.startswith("P:"):
                html_lines.append(
                    f'<div class="line patient"><span class="speaker pat">P:</span>{escape(line[2:].lstrip())}</div>'
                )
            else:
                html_lines.append(f'<div class="line other">{escape(line)}</div>')
        return "\n".join(html_lines)

    injected = gt.get("injected_symptoms", [])
    injected_summary = ""
    if injected:
        inj_rows = "".join(
            f'<div class="inj-item"><span class="badge injected">{escape(i.get("category", ""))}</span> '
            f'{escape(i.get("description", ""))} <span class="code">{escape(i.get("code", ""))}</span></div>'
            for i in injected
        )
        injected_summary = f"""
            <div class="section injected-section">
                <h3>🧪 Injected Symptoms <span class="count">({len(injected)})</span></h3>
                {inj_rows}
            </div>
        """

    diag_items = [
        {"description": r.get("panel", ""), "code": "", "code_system": "", "source": r.get("source", "ccda")}
        for r in gt.get("diagnostic_results", [])
    ]
    diag_section = _section_html("Diagnostic Results", diag_items)

    case_meta = gt.get("case_metadata", {}) or {}
    meta_line = ""
    if case_meta:
        cs = str(case_meta.get("clinical_suspicion", "") or "")
        covid_flag = str(case_meta.get("is_covid_case", "") or "")
        meta_line = f" &bull; Suspicion: {escape(cs)} &bull; is_covid_case: {escape(covid_flag)}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CCDA Ground Truth vs Synthetic Conversation — {escape(patient_name)}</title>
<style>
  :root {{
    --bg: #0f1117;
    --panel: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4eb;
    --text-muted: #8b8fa3;
    --accent: #6c63ff;
    --green: #22c55e;
    --red: #ef4444;
    --purple: #a855f7;
    --doctor: #3b82f6;
    --patient: #f59e0b;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  .header {{
    background: linear-gradient(135deg, #1a1d27 0%, #2a1d3a 100%);
    padding: 24px 32px;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 1.5rem;
    font-weight: 600;
    background: linear-gradient(135deg, #6c63ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .header .subtitle {{ color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }}
  .container {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    height: calc(100vh - 80px);
  }}
  .panel {{
    overflow-y: auto;
    padding: 20px;
    border-right: 1px solid var(--border);
  }}
  .panel:last-child {{ border-right: none; }}
  .panel-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--accent);
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 10;
    padding-top: 4px;
  }}
  .section {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 16px;
  }}
  .section h3 {{
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: var(--text);
  }}
  .scenario-section h3 {{ color: var(--purple); }}
  .scenario-meta {{ font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px; }}
  .count {{ color: var(--text-muted); font-weight: 400; font-size: 0.8rem; }}
  .item {{
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 0.85rem;
    border-left: 3px solid transparent;
  }}
  .item-match {{ background: rgba(34,197,94,0.08); border-left-color: var(--green); }}
  .item-miss-critical {{ background: rgba(239,68,68,0.15); border-left-color: var(--red); }}
  .item-miss-minor {{ background: rgba(139,143,163,0.05); border-left-color: var(--text-muted); opacity: 0.8; }}
  .item-exclude {{ background: transparent; border-left-color: transparent; opacity: 0.5; }}
  .item-injected {{ background: rgba(168,85,247,0.1); border-left-color: var(--purple); }}
  .item-main {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
  .item-meta {{ margin-top: 4px; font-size: 0.75rem; color: var(--text-muted); }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    white-space: nowrap;
  }}
  .badge.match {{ background: rgba(34,197,94,0.2); color: var(--green); }}
  .badge.miss-critical {{ background: rgba(239,68,68,0.3); color: #fca5a5; }}
  .badge.miss-minor {{ background: rgba(139,143,163,0.2); color: var(--text-muted); font-weight: 400; }}
  .badge.exclude {{ background: transparent; color: var(--text-muted); font-weight: 400; border: 1px solid var(--border); }}
  .badge.injected {{ background: rgba(168,85,247,0.2); color: var(--purple); }}
  .badge.priority-must {{ background: #ef4444; color: white; border-radius: 2px; padding: 2px 6px; font-size: 0.65rem; }}
  .badge.priority-should {{ background: #f59e0b; color: #1e293b; border-radius: 2px; padding: 2px 6px; font-size: 0.65rem; }}
  .badge.priority-may {{ background: #3b82f6; color: white; border-radius: 2px; padding: 2px 6px; font-size: 0.65rem; }}
  .badge.priority-exclude {{ background: #475569; color: white; border-radius: 2px; padding: 2px 6px; font-size: 0.65rem; }}
  .desc {{ flex: 1; min-width: 200px; }}
  .code {{ color: var(--text-muted); font-family: monospace; font-size: 0.75rem; }}
  .date {{ color: var(--text-muted); }}
  .value {{ color: var(--accent); font-weight: 500; }}
  .demo-table {{ width: 100%; border-collapse: collapse; }}
  .demo-table td {{ padding: 6px 12px; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
  .field-label {{ color: var(--text-muted); width: 100px; }}
  .empty {{ color: var(--text-muted); font-style: italic; font-size: 0.85rem; }}
  .injected-section {{ border-color: var(--purple); }}
  .inj-item {{ padding: 6px 0; font-size: 0.85rem; }}
  .line {{ padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.85rem; }}
  .line.doctor {{ background: rgba(59,130,246,0.08); border-left: 3px solid var(--doctor); }}
  .line.patient {{ background: rgba(245,158,11,0.08); border-left: 3px solid var(--patient); }}
  .speaker {{ font-weight: 700; margin-right: 8px; }}
  .speaker.doc {{ color: var(--doctor); }}
  .speaker.pat {{ color: var(--patient); }}
  ::-webkit-scrollbar {{ width: 8px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #3a3d4a; }}
</style>
</head>
<body>
<div class="header">
    <h1>CCDA Ground Truth vs Synthetic Conversation</h1>
    <div class="subtitle">Patient: {escape(patient_name)} &bull; Source: {escape(gt.get('source_file', '?'))}{meta_line}</div>
</div>
<div class="container">
    <div class="panel">
        <div class="panel-title">📋 CCDA Ground Truth & Scenario</div>
        {_demographics_html(gt.get('demographics', {}))}
        {_scenario_html(scenario)}
        {injected_summary}
        {_section_html('Conditions / Problems', gt.get('conditions', []))}
        {_section_html('Medications', gt.get('medications', []), show_dates=True)}
        {_section_html('Immunizations', gt.get('immunizations', []), show_dates=True)}
        {_section_html('Encounters', gt.get('encounters', []), show_dates=True)}
        {_section_html('Vital Signs', gt.get('vital_signs', []))}
        {_section_html('Procedures', gt.get('procedures', []))}
        {diag_section}
        {_section_html('Care Plans', gt.get('care_plans', []))}
        {_section_html('Social History', gt.get('social_history', []))}
        {_section_html('Allergies', gt.get('allergies', []))}
        {_section_html('Functional Status', gt.get('functional_status', []))}
    </div>
    <div class="panel">
        <div class="panel-title">💬 Synthetic Conversation</div>
        {_transcript_html(transcript)}
    </div>
</div>
</body>
</html>"""
    return html


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ccda_comparison_viewer_v2.py path/to/ground_truth.json [path/to/transcript.txt]")
        sys.exit(1)

    json_path = os.path.abspath(sys.argv[1])
    if not os.path.exists(json_path):
        print(f"ERROR: JSON not found: {json_path}")
        sys.exit(1)

    print(f"Loading ground truth: {json_path}")
    gt = load_json(json_path)

    patient_name = gt.get("demographics", {}).get("name", "Unknown")
    print(f"Patient: {patient_name}")

    if len(sys.argv) > 2:
        transcript_path = os.path.abspath(sys.argv[2])
        if not os.path.exists(transcript_path):
            print(f"WARNING: Transcript path does not exist: {transcript_path}")
            transcript = ""
        else:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = f.read()
        print(f"Transcript (explicit): {'loaded' if transcript else 'NOT FOUND'}")
    else:
        transcript = load_transcript_by_patient(patient_name)
        print(f"Transcript (by name): {'loaded' if transcript else 'NOT FOUND'}")

    scenario = None
    base = os.path.basename(json_path)
    if base.endswith("_ground_truth_v2.json"):
        scenario_path = json_path.replace("_ground_truth_v2.json", "_ground_truth_v2_scenario_v2.json")
        if os.path.exists(scenario_path):
            try:
                scenario = load_json(scenario_path)
                print(f"Loaded scenario: {scenario_path}")
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Failed to load scenario file {scenario_path}: {e}")

    html = build_html(gt, transcript, scenario)

    out_path = os.path.join(DEFAULT_DIR, "comparison_viewer_v2.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")

    webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")
    print("✓ Opened in browser!")


if __name__ == "__main__":
    main()

