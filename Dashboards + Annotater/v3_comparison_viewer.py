"""
v3 Comparison Viewer — generates a self-contained HTML report showing:

  Panel 1: Patient profile (demographics, present symptoms, negated
           symptoms, CCDA noise)
  Panel 2: The exact prompt sent to GPT-4o (system + user)
  Panel 3: The generated transcript (D:/P: color-coded)

Usage:
    python v3_comparison_viewer.py --scenario scenario_v3.json --transcript COVID_SYNTHETIC_v3_....txt
    python v3_comparison_viewer.py --scenario scenario_v3.json   # auto-finds transcript

If run from the v3/ directory, it will auto-detect the latest scenario+transcript.
"""

from __future__ import annotations

import json
import os
import sys
import webbrowser
from html import escape

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.join(
    SCRIPT_DIR, "..", "Synthetic Transcript Generation",
    "synthetic generation", "v3",
)
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "..", "Synthetic Transcript Generation",
    "Synthetic Transcripts", "Generated Transcripts v3",
)


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_prompt_text(scenario: dict) -> tuple[str, str]:
    """Re-generate the prompt from the scenario so the viewer can display it."""
    # Import the v3 prompt builder
    v3_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "Synthetic Transcript Generation", "synthetic generation", "v3",
    )
    if v3_dir not in sys.path:
        sys.path.insert(0, v3_dir)
    from generate_transcript_v3 import build_prompt
    return build_prompt(scenario)


def _transcript_html(text: str) -> str:
    if not text.strip():
        return '<p class="empty">No transcript loaded.</p>'
    lines = text.strip().split("\n")
    html_parts: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("D:"):
            html_parts.append(
                f'<div class="line doctor"><span class="speaker doc">D:</span>'
                f'{escape(line[2:].strip())}</div>'
            )
        elif line.startswith("P:"):
            html_parts.append(
                f'<div class="line patient"><span class="speaker pat">P:</span>'
                f'{escape(line[2:].strip())}</div>'
            )
        else:
            html_parts.append(f'<div class="line other">{escape(line)}</div>')
    return "\n".join(html_parts)


def _symptom_check(symptom: str, transcript_lower: str) -> bool:
    """Check whether a present symptom appears in the transcript."""
    # Use the parenthetical description if present
    desc = symptom.lower()
    if "(" in desc:
        inner = desc.split("(", 1)[1].rstrip(")")
        keywords = [w.strip() for w in inner.replace("/", " ").split() if len(w.strip()) > 2]
        if keywords and sum(1 for k in keywords if k in transcript_lower) >= max(1, len(keywords) // 2):
            return True
    # Also check the main term
    main = desc.split("(")[0].strip()
    main_words = [w for w in main.split() if len(w) > 3]
    if main_words and all(w in transcript_lower for w in main_words):
        return True
    # Direct substring
    if main in transcript_lower:
        return True
    return False


def build_html(scenario: dict, transcript: str) -> str:
    demo = scenario.get("demographics", {})
    present = scenario.get("present_symptoms", [])
    negated = scenario.get("negated_symptoms", [])
    chief = scenario.get("chief_complaint", "")
    noise = scenario.get("ccda_noise", {})
    seed = scenario.get("seed", "N/A")
    location = scenario.get("location", {})

    patient_name = demo.get("name", "Unknown")
    transcript_lower = transcript.lower()

    # Build prompt text
    try:
        sys_prompt, usr_prompt = _build_prompt_text(scenario)
    except Exception as e:
        sys_prompt = f"(Could not regenerate prompt: {e})"
        usr_prompt = ""

    # ── Profile panel ────────────────────────────────────────────────
    demo_rows = "".join(
        f'<tr><td class="lbl">{k}</td><td>{escape(str(v))}</td></tr>'
        for k, v in [
            ("Name", demo.get("name", "?")),
            ("Age", demo.get("age", "?")),
            ("Gender", "Female" if demo.get("gender") == "F" else "Male"),
            ("Race", demo.get("race", "?")),
            ("Ethnicity", demo.get("ethnicity", "?")),
            ("Location", f"{demo.get('city', '?')}, {demo.get('state', '?')}"),
            ("Seed", seed),
        ]
    )

    # Present symptoms with match checking
    present_items = []
    for s in present:
        found = _symptom_check(s, transcript_lower)
        icon = "✅" if found else "❌"
        cls = "sym-match" if found else "sym-miss"
        present_items.append(f'<div class="sym-item {cls}">{icon} {escape(s)}</div>')
    present_html = "\n".join(present_items)
    present_matched = sum(1 for s in present if _symptom_check(s, transcript_lower))

    # Negated symptoms
    negated_items = []
    for s in negated:
        # For negated, check if it was mentioned at all (ideally denied)
        mentioned = _symptom_check(s, transcript_lower)
        icon = "✅" if mentioned else "⚠️"
        cls = "sym-neg-ok" if mentioned else "sym-neg-miss"
        negated_items.append(f'<div class="sym-item {cls}">{icon} {escape(s)}</div>')
    negated_html = "\n".join(negated_items)

    # CCDA noise
    noise_parts: list[str] = []
    for label, key in [
        ("Conditions", "conditions"),
        ("Medications", "medications"),
        ("Allergies", "allergies"),
        ("Social History", "social_history"),
        ("Immunizations", "immunizations"),
        ("Injected Symptoms", "injected_symptoms"),
    ]:
        items = noise.get(key, [])
        if not items:
            continue
        rows = "".join(f"<li>{escape(str(i))}</li>" for i in items)
        noise_parts.append(f'<div class="noise-group"><h4>{label} ({len(items)})</h4><ul>{rows}</ul></div>')
    noise_html = "\n".join(noise_parts) if noise_parts else '<p class="empty">No CCDA noise</p>'

    # ── Transcript panel ─────────────────────────────────────────────
    t_html = _transcript_html(transcript)
    t_lines = [l for l in transcript.strip().split("\n") if l.strip()]
    d_count = sum(1 for l in t_lines if l.startswith("D:"))
    p_count = sum(1 for l in t_lines if l.startswith("P:"))

    # ── Full HTML ────────────────────────────────────────────────────
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>v3 Pipeline Viewer — {escape(patient_name)}</title>
<style>
:root {{
  --bg: #0f1117; --panel: #1a1d27; --border: #2a2d3a;
  --text: #e1e4eb; --muted: #8b8fa3; --accent: #6c63ff;
  --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
  --purple: #a855f7; --blue: #3b82f6; --orange: #f97316;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: 'Inter','Segoe UI',system-ui,sans-serif;
  background: var(--bg); color: var(--text); line-height:1.5;
}}
.header {{
  background: linear-gradient(135deg,#1a1d27,#2a1d3a);
  padding: 16px 24px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 16px;
}}
.header h1 {{
  font-size:1.3rem; font-weight:700;
  background: linear-gradient(135deg,#6c63ff,#a855f7);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.header .meta {{ color:var(--muted); font-size:0.85rem; }}

/* Tab navigation */
.tabs {{
  display:flex; background:var(--panel);
  border-bottom:1px solid var(--border);
}}
.tab {{
  padding:10px 20px; cursor:pointer; font-size:0.9rem;
  color:var(--muted); border-bottom:2px solid transparent;
  transition: all 0.2s;
}}
.tab:hover {{ color:var(--text); }}
.tab.active {{
  color:var(--accent); border-bottom-color:var(--accent);
  font-weight:600;
}}

/* Panels */
.tab-content {{ display:none; height:calc(100vh - 100px); overflow-y:auto; padding:20px; }}
.tab-content.active {{ display:block; }}

/* Profile */
.profile-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
@media(max-width:900px) {{ .profile-grid {{ grid-template-columns:1fr; }} }}
.card {{
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; padding:16px;
}}
.card h3 {{
  font-size:0.95rem; color:var(--accent); margin-bottom:10px;
  padding-bottom:6px; border-bottom:1px solid var(--border);
}}
.card table {{ width:100%; }}
.card td {{ padding:3px 8px; font-size:0.88rem; }}
.card .lbl {{ color:var(--muted); width:100px; }}

.sym-item {{
  padding:4px 8px; margin:2px 0; border-radius:4px; font-size:0.88rem;
}}
.sym-match {{ background:rgba(34,197,94,0.1); border-left:3px solid var(--green); }}
.sym-miss {{ background:rgba(239,68,68,0.1); border-left:3px solid var(--red); }}
.sym-neg-ok {{ background:rgba(34,197,94,0.05); border-left:3px solid var(--green); }}
.sym-neg-miss {{ background:rgba(249,158,11,0.1); border-left:3px solid var(--yellow); }}

.noise-group h4 {{ font-size:0.85rem; color:var(--muted); margin:8px 0 4px; }}
.noise-group ul {{ list-style:none; padding-left:0; }}
.noise-group li {{
  font-size:0.85rem; padding:2px 0; color:var(--text);
  border-left:2px solid var(--border); padding-left:10px; margin:2px 0;
}}

/* Prompt */
.prompt-block {{
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; margin-bottom:16px; overflow:hidden;
}}
.prompt-label {{
  font-size:0.8rem; font-weight:600; padding:8px 12px;
  background:rgba(108,99,255,0.15); color:var(--accent);
  border-bottom:1px solid var(--border);
}}
.prompt-text {{
  padding:12px; white-space:pre-wrap; font-size:0.82rem;
  font-family:'Cascadia Code','Fira Code','Consolas',monospace;
  line-height:1.6; color:var(--text); max-height:600px; overflow-y:auto;
}}

/* Transcript */
.transcript-stats {{
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; padding:12px 16px; margin-bottom:16px;
  display:flex; gap:20px; font-size:0.85rem;
}}
.stat {{ display:flex; gap:6px; align-items:center; }}
.stat .num {{ font-weight:700; color:var(--accent); }}
.transcript-body {{
  background:var(--panel); border:1px solid var(--border);
  border-radius:8px; padding:16px;
}}
.line {{ padding:6px 10px; margin:2px 0; border-radius:4px; font-size:0.9rem; }}
.doctor {{ background:rgba(59,130,246,0.08); }}
.patient {{ background:rgba(249,158,11,0.08); }}
.speaker {{
  font-weight:700; margin-right:8px; display:inline-block; min-width:24px;
}}
.doc {{ color:var(--blue); }}
.pat {{ color:var(--yellow); }}
.other {{ color:var(--muted); font-style:italic; }}
.empty {{ color:var(--muted); font-style:italic; }}

.coverage-bar {{
  display:flex; align-items:center; gap:8px; margin:8px 0;
  font-size:0.85rem;
}}
.bar-track {{
  flex:1; height:8px; background:var(--border); border-radius:4px; overflow:hidden;
}}
.bar-fill {{ height:100%; border-radius:4px; transition:width 0.3s; }}
.bar-green {{ background:var(--green); }}
</style>
</head>
<body>
<div class="header">
  <h1>v3 Pipeline Viewer</h1>
  <div class="meta">{escape(patient_name)} &nbsp;|&nbsp; Seed: {escape(str(seed))}</div>
</div>

<div class="tabs">
  <div class="tab active" onclick="switchTab('profile')">📋 Profile</div>
  <div class="tab" onclick="switchTab('map')">🗺️ Map</div>
  <div class="tab" onclick="switchTab('prompt')">💬 Prompt</div>
  <div class="tab" onclick="switchTab('transcript')">📝 Transcript</div>
</div>

<!-- ═══ PROFILE TAB ═══ -->
<div class="tab-content active" id="tab-profile">
  <div class="profile-grid">
    <div class="card">
      <h3>Demographics</h3>
      <table>{demo_rows}</table>
    </div>

    <div class="card">
      <h3>🟢 Present Symptoms ({present_matched}/{len(present)} found in transcript)</h3>
      <div class="coverage-bar">
        <div class="bar-track"><div class="bar-fill bar-green"
          style="width:{int(present_matched/max(len(present),1)*100)}%"></div></div>
        <span>{int(present_matched/max(len(present),1)*100)}%</span>
      </div>
      {present_html}
    </div>

    <div class="card">
      <h3>🔴 Negated Symptoms (patient should deny these)</h3>
      {negated_html}
    </div>

    <div class="card">
      <h3>🏥 CCDA Background Noise</h3>
      {noise_html}
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <h3>Chief Complaint</h3>
    <p style="font-size:1.05rem; color:var(--orange); font-weight:600;">
      {escape(chief)}
    </p>
  </div>
</div>

<!-- ═══ MAP TAB ═══ -->
<div class="tab-content" id="tab-map">
  <div class="card" style="margin-bottom:16px">
    <h3>🗺️ Patient Location — South Carolina</h3>
    <p style="font-size:0.88rem; color:var(--muted); margin-bottom:8px;">
      Hospital hub: <strong style="color:var(--accent)">{escape(location.get("hospital", "N/A"))}</strong>
      ({escape(location.get("city", "?"))}, {escape(location.get("state", "?"))})
      &nbsp;|&nbsp; Hub ID: {escape(location.get("hub_id", "?"))}
      &nbsp;|&nbsp; Patient coords: {location.get("lat", "?")}, {location.get("lng", "?")}
    </p>
    <div id="sc-map" style="width:100%; height:520px; border-radius:8px; border:1px solid var(--border);"></div>
  </div>
</div>

<!-- Leaflet CSS + JS -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
(function() {{
  // SC hospital hubs
  var hubs = [
    {{id:"H1", name:"AnMed Health Medical Center",    city:"Anderson",     lat:34.526, lng:-82.646}},
    {{id:"H2", name:"Prisma Health Greenville",       city:"Greenville",   lat:34.819, lng:-82.416}},
    {{id:"H3", name:"Prisma Health Richland",         city:"Columbia",     lat:34.032, lng:-81.033}},
    {{id:"H4", name:"Aiken Regional Medical Center",  city:"Aiken",        lat:33.565, lng:-81.763}},
    {{id:"H5", name:"MUSC Health",                    city:"Charleston",   lat:32.785, lng:-79.947}},
    {{id:"H6", name:"McLeod Regional Medical Center", city:"Florence",     lat:34.192, lng:-79.765}},
    {{id:"H7", name:"Grand Strand Medical Center",    city:"Myrtle Beach", lat:33.748, lng:-78.847}}
  ];

  var patientLat = {location.get("lat", 33.8)};
  var patientLng = {location.get("lng", -81.0)};
  var patientHub = "{escape(location.get("hub_id", ""))}";

  // Wait for the map div to be visible before initializing
  var mapInitialized = false;
  var origSwitchTab = window.switchTab;

  function initMap() {{
    if (mapInitialized) return;
    mapInitialized = true;

    var map = L.map('sc-map', {{
      zoomControl: true,
      attributionControl: true
    }}).setView([33.85, -80.9], 7);

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; OpenStreetMap &copy; CARTO',
      maxZoom: 18
    }}).addTo(map);

    // Hospital hub markers (blue circles)
    hubs.forEach(function(h) {{
      var isActive = (h.id === patientHub);
      var color = isActive ? '#6c63ff' : '#3b82f6';
      var radius = isActive ? 10 : 7;
      var opacity = isActive ? 1.0 : 0.6;

      L.circleMarker([h.lat, h.lng], {{
        radius: radius,
        fillColor: color,
        color: '#fff',
        weight: isActive ? 2 : 1,
        fillOpacity: opacity
      }}).addTo(map).bindTooltip(
        '<strong>' + h.id + ': ' + h.name + '</strong><br>' + h.city + ', SC',
        {{ permanent: isActive, direction: 'top', className: 'hub-tooltip' }}
      );
    }});

    // Patient marker (orange/red pulsing dot)
    L.circleMarker([patientLat, patientLng], {{
      radius: 8,
      fillColor: '#f97316',
      color: '#fff',
      weight: 2,
      fillOpacity: 0.95
    }}).addTo(map).bindTooltip(
      '<strong>&#128205; Patient</strong><br>' +
      patientLat.toFixed(4) + ', ' + patientLng.toFixed(4),
      {{ permanent: true, direction: 'bottom', className: 'patient-tooltip' }}
    );

    // Draw a dashed line from patient to their hub
    hubs.forEach(function(h) {{
      if (h.id === patientHub) {{
        L.polyline([[h.lat, h.lng], [patientLat, patientLng]], {{
          color: '#6c63ff',
          weight: 1.5,
          dashArray: '6,4',
          opacity: 0.6
        }}).addTo(map);
      }}
    }});

    setTimeout(function() {{ map.invalidateSize(); }}, 200);
  }}

  window.switchTab = function(name) {{
    document.querySelectorAll('.tab-content').forEach(function(el) {{ el.classList.remove('active'); }});
    document.querySelectorAll('.tab').forEach(function(el) {{ el.classList.remove('active'); }});
    document.getElementById('tab-' + name).classList.add('active');
    var tabs = document.querySelectorAll('.tab');
    var idx = {{'profile':0,'map':1,'prompt':2,'transcript':3}}[name];
    if (idx !== undefined) tabs[idx].classList.add('active');
    if (name === 'map') initMap();
  }};
}})();
</script>
<style>
.hub-tooltip {{
  background: rgba(26,29,39,0.92) !important;
  color: #e1e4eb !important;
  border: 1px solid #6c63ff !important;
  border-radius: 6px !important;
  font-size: 0.82rem !important;
  padding: 4px 8px !important;
}}
.patient-tooltip {{
  background: rgba(249,115,22,0.92) !important;
  color: #fff !important;
  border: 1px solid #f97316 !important;
  border-radius: 6px !important;
  font-size: 0.82rem !important;
  padding: 4px 8px !important;
}}
.leaflet-tooltip-top:before {{ border-top-color: rgba(26,29,39,0.92) !important; }}
.leaflet-tooltip-bottom:before {{ border-bottom-color: rgba(249,115,22,0.92) !important; }}
</style>

<!-- ═══ PROMPT TAB ═══ -->
<div class="tab-content" id="tab-prompt">
  <div class="prompt-block">
    <div class="prompt-label">SYSTEM PROMPT ({len(sys_prompt)} chars)</div>
    <div class="prompt-text">{escape(sys_prompt)}</div>
  </div>
  <div class="prompt-block">
    <div class="prompt-label">USER PROMPT ({len(usr_prompt)} chars)</div>
    <div class="prompt-text">{escape(usr_prompt)}</div>
  </div>
</div>

<!-- ═══ TRANSCRIPT TAB ═══ -->
<div class="tab-content" id="tab-transcript">
  <div class="transcript-stats">
    <div class="stat"><span class="num">{d_count}</span> Doctor lines</div>
    <div class="stat"><span class="num">{p_count}</span> Patient lines</div>
    <div class="stat"><span class="num">{len(t_lines)}</span> Total lines</div>
    <div class="stat">Present coverage: <span class="num">{present_matched}/{len(present)}</span></div>
  </div>
  <div class="transcript-body">
    {t_html}
  </div>
</div>

</body>
</html>'''
    return html


def _find_latest_pair(output_dir: str) -> tuple[str, str]:
    """Find the most recent scenario + transcript pair in the output dir."""
    scenarios = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("scenario_v3") and f.endswith(".json")],
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
        reverse=True,
    )
    for sc_file in scenarios:
        sc_path = os.path.join(output_dir, sc_file)
        # Try matching transcript
        base = sc_file.replace("scenario_v3_", "").replace(".json", "")
        candidates = [
            f for f in os.listdir(output_dir)
            if f.endswith(".txt") and base in f
        ]
        if candidates:
            return sc_path, os.path.join(output_dir, candidates[0])
    return "", ""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="v3 Pipeline Comparison Viewer")
    parser.add_argument("--scenario", default="", help="Path to scenario_v3.json")
    parser.add_argument("--transcript", default="", help="Path to generated transcript .txt")
    parser.add_argument("--out", default="", help="Output HTML path (default: opens in browser)")
    args = parser.parse_args()

    # Auto-detect if not provided
    scenario_path = args.scenario
    transcript_path = args.transcript

    if not scenario_path:
        if os.path.isdir(OUTPUT_DIR):
            scenario_path, auto_transcript = _find_latest_pair(OUTPUT_DIR)
            if not transcript_path and auto_transcript:
                transcript_path = auto_transcript

    if not scenario_path:
        print("ERROR: No scenario file found. Use --scenario path/to/scenario_v3.json")
        sys.exit(1)

    print(f"Scenario:   {os.path.basename(scenario_path)}")
    scenario = _load_json(scenario_path)

    transcript = ""
    if transcript_path and os.path.exists(transcript_path):
        print(f"Transcript: {os.path.basename(transcript_path)}")
        transcript = _load_text(transcript_path)
    else:
        print("WARNING: No transcript file found — showing profile + prompt only.")

    html = build_html(scenario, transcript)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "v3_comparison_viewer.html",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved:      {out_path}")

    webbrowser.open(f"file:///{os.path.abspath(out_path).replace(os.sep, '/')}")
    print("✓ Opened in browser.")


if __name__ == "__main__":
    main()
