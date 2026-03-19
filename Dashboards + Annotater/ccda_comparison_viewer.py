"""
Generate a self-contained HTML file that shows the CCDA ground truth
side-by-side with the synthetic conversation transcript.

Highlights:
  - CCDA items that appear in the transcript (green ✅)
  - CCDA items that are absent from the transcript (red ❌)
  - Injected symptoms in a distinct purple badge
  - Doctor / Patient lines color-coded

Usage:
    python ccda_comparison_viewer.py                    # uses defaults
    python ccda_comparison_viewer.py ground_truth.json  # explicit JSON
"""

import json
import os
import sys
import re
import webbrowser
from html import escape

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPT_DIR = os.path.join(DEFAULT_DIR, "..", "Synthetic Transcript Generation", "Synthetic Transcripts")
SYNTHETIC_DIR = os.path.join(DEFAULT_DIR, "..", "Synthetic Transcript Generation", "synthetic generation")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transcript(patient_name: str) -> str:
    """Try to find the synthetic transcript for this patient."""
    safe_name = patient_name.replace(" ", "_")
    candidates = [
        os.path.join(TRANSCRIPT_DIR, f"COVID_SYNTHETIC_{safe_name}.txt"),
    ]
    # Also search for any file containing the patient name
    if os.path.isdir(TRANSCRIPT_DIR):
        for fname in os.listdir(TRANSCRIPT_DIR):
            if safe_name in fname or patient_name.split()[0] in fname:
                candidates.append(os.path.join(TRANSCRIPT_DIR, fname))

    for c in candidates:
        if os.path.exists(c):
            with open(c, "r", encoding="utf-8") as f:
                return f.read()
    return ""


def check_match(description: str, transcript_lower: str) -> bool:
    """Check if a ground-truth item is mentioned in the transcript."""
    desc_lower = description.lower()

    # Remove parenthetical type clarifiers for matching
    clean = re.sub(r"\s*\(disorder\)|\(finding\)|\(situation\)|\(procedure\)|\(regime/therapy\)|\(observable entity\)", "", desc_lower).strip()

    # Direct substring
    if clean in transcript_lower:
        return True

    # Check key words (for multi-word descriptions)
    words = [w for w in clean.split() if len(w) > 3]
    if len(words) >= 2:
        # Check if most key words appear
        matches = sum(1 for w in words if w in transcript_lower)
        if matches >= len(words) * 0.6:
            return True

    # Special-case common mappings
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


def build_html(gt: dict, transcript: str) -> str:
    """Build the full HTML comparison page."""
    patient_name = gt.get("demographics", {}).get("name", "Unknown")
    transcript_lower = transcript.lower()

    # ── Section builders ────────────────────────────────────────────────
    def _section_html(title: str, items: list, show_dates: bool = False) -> str:
        if not items:
            return f'<div class="section"><h3>{escape(title)}</h3><p class="empty">No data</p></div>'

        rows = []
        matched = 0
        for item in items:
            desc = item.get("description", "")
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
            code_sys = item.get("code_system", "")
            code_display = f'<span class="code">{escape(code)}</span>' if code else ""

            date_info = ""
            if show_dates:
                start = item.get("start", "")
                stop = item.get("stop", "")
                if start:
                    date_info = f'<span class="date">{escape(start[:10])}'
                    if stop and stop != start:
                        date_info += f' → {escape(stop[:10])}'
                    date_info += '</span>'

            extra = ""
            if "value" in item:
                extra = f'<span class="value">{escape(str(item["value"]))} {escape(item.get("unit", ""))}</span>'

            rows.append(f'''
                <div class="item {row_class}">
                    <div class="item-main">
                        {pri_badge}
                        {badge}
                        <span class="desc">{escape(desc)}</span>
                        {code_display}
                    </div>
                    <div class="item-meta">{date_info} {extra}</div>
                </div>
            ''')

        coverage = f"{matched}/{len(items)}" if transcript else "N/A"
        return f'''
            <div class="section">
                <h3>{escape(title)} <span class="count">({len(items)} items, {coverage} matched)</span></h3>
                {''.join(rows)}
            </div>
        '''

    def _demographics_html(demo: dict) -> str:
        fields = [
            ("Name", demo.get("name", "?")),
            ("Age", str(demo.get("age", "?"))),
            ("Gender", "Female" if demo.get("gender") == "F" else "Male"),
            ("Race", demo.get("race", "?")),
            ("Ethnicity", demo.get("ethnicity", "?")),
            ("City", demo.get("city", "?")),
            ("State", demo.get("state", "?")),
        ]
        rows = "".join(f'<tr><td class="field-label">{k}</td><td>{escape(v)}</td></tr>' for k, v in fields)
        return f'''
            <div class="section">
                <h3>Demographics</h3>
                <table class="demo-table">{rows}</table>
            </div>
        '''

    def _transcript_html(text: str) -> str:
        if not text:
            return '<p class="empty">No transcript loaded.</p>'
        lines = text.strip().split("\n")
        html_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("D:"):
                html_lines.append(f'<div class="line doctor"><span class="speaker doc">D:</span>{escape(line[2:])}</div>')
            elif line.startswith("P:"):
                html_lines.append(f'<div class="line patient"><span class="speaker pat">P:</span>{escape(line[2:])}</div>')
            else:
                html_lines.append(f'<div class="line other">{escape(line)}</div>')
        return "\n".join(html_lines)

    # ── Injected symptoms summary ─────────────────────────────────────
    injected = gt.get("injected_symptoms", [])
    injected_summary = ""
    if injected:
        inj_rows = "".join(
            f'<div class="inj-item"><span class="badge injected">{escape(i.get("category", ""))}</span> '
            f'{escape(i["description"])} <span class="code">{escape(i.get("code", ""))}</span></div>'
            for i in injected
        )
        injected_summary = f'''
            <div class="section injected-section">
                <h3>🧪 Injected Symptoms <span class="count">({len(injected)})</span></h3>
                {inj_rows}
            </div>
        '''

    # ── Pre-compute diagnostic results for the f-string ──────────────
    diag_items = [
        {"description": r.get("panel", ""), "code": "", "code_system": "", "source": r.get("source", "ccda")}
        for r in gt.get('diagnostic_results', [])
    ]
    diag_section = _section_html('Diagnostic Results', diag_items)

    # ── Full HTML ─────────────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
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
  /* Transcript */
  .line {{ padding: 8px 12px; border-radius: 6px; margin-bottom: 6px; font-size: 0.85rem; }}
  .line.doctor {{ background: rgba(59,130,246,0.08); border-left: 3px solid var(--doctor); }}
  .line.patient {{ background: rgba(245,158,11,0.08); border-left: 3px solid var(--patient); }}
  .speaker {{ font-weight: 700; margin-right: 8px; }}
  .speaker.doc {{ color: var(--doctor); }}
  .speaker.pat {{ color: var(--patient); }}
  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 8px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: #3a3d4a; }}
</style>
</head>
<body>
<div class="header">
    <h1>CCDA Ground Truth vs Synthetic Conversation</h1>
    <div class="subtitle">Patient: {escape(patient_name)} &bull; Source: {escape(gt.get('source_file', '?'))}</div>
</div>
<div class="container">
    <div class="panel">
        <div class="panel-title">📋 CCDA Ground Truth</div>
        {_demographics_html(gt.get('demographics', {}))}
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
</html>'''
    return html



def main():
    # Determine which JSON to load
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        # Prefer modified if it exists, fallback to base
        modified = os.path.join(SYNTHETIC_DIR, "modified_ground_truth.json")
        base = os.path.join(SYNTHETIC_DIR, "ground_truth.json")
        json_path = modified if os.path.exists(modified) else base

    if not os.path.exists(json_path):
        print(f"ERROR: JSON not found: {json_path}")
        print("  Run ccda_to_ground_truth.py first.")
        sys.exit(1)

    print(f"Loading: {json_path}")
    gt = load_json(json_path)

    patient_name = gt.get("demographics", {}).get("name", "Unknown")
    print(f"Patient: {patient_name}")

    transcript = load_transcript(patient_name)
    print(f"Transcript: {'loaded' if transcript else 'NOT FOUND'} "
          f"({len(transcript)} chars)")

    html = build_html(gt, transcript)

    out_path = os.path.join(DEFAULT_DIR, "comparison_viewer.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")

    # Open in browser
    webbrowser.open(f"file:///{out_path.replace(os.sep, '/')}")
    print("✓ Opened in browser!")


if __name__ == "__main__":
    main()
