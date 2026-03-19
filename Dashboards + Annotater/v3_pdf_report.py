"""
v3 PDF Report Generator — builds a clean, professional PDF that mirrors
the v3 Comparison Viewer (Profile, Prompt, Transcript) for physician review.

Usage:
    python v3_pdf_report.py
    python v3_pdf_report.py --scenario path/to/scenario_v3.json --transcript path/to/transcript.txt
    python v3_pdf_report.py --out custom_report.pdf

Auto-detects the latest scenario + transcript if not specified.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from datetime import datetime

from fpdf import FPDF

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "..", "Synthetic Transcript Generation",
    "Synthetic Transcripts", "Generated Transcripts v3",
)

# ─── Colours (RGB tuples) ───────────────────────────────────────────
COL_HEADER_BG   = (44, 62, 80)     # dark blue-grey
COL_HEADER_FG   = (255, 255, 255)
COL_SECTION_BG  = (52, 73, 94)     # slightly lighter
COL_SECTION_FG  = (255, 255, 255)
COL_LABEL       = (100, 100, 100)
COL_VALUE       = (30, 30, 30)
COL_GREEN       = (34, 197, 94)
COL_RED         = (239, 68, 68)
COL_YELLOW      = (245, 158, 11)
COL_DOCTOR_BG   = (235, 245, 235)
COL_PATIENT_BG  = (245, 245, 245)
COL_DOCTOR_TXT  = (20, 110, 50)
COL_PATIENT_TXT = (60, 60, 60)
COL_MUTED       = (140, 140, 140)

_CASE_TYPE_LABELS = {
    "novel_virus":  "Novel Virus",
    "flu_like":     "Flu-Like",
    "differential": "Varied Case",
    "healthy":      "Healthy",
}


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _symptom_check(symptom: str, transcript_lower: str) -> bool:
    """Check whether a present symptom appears in the transcript."""
    desc = symptom.lower()
    if "(" in desc:
        inner = desc.split("(", 1)[1].rstrip(")")
        keywords = [w.strip() for w in inner.replace("/", " ").split() if len(w.strip()) > 2]
        if keywords and sum(1 for k in keywords if k in transcript_lower) >= max(1, len(keywords) // 2):
            return True
    main = desc.split("(")[0].strip()
    main_words = [w for w in main.split() if len(w) > 3]
    if main_words and all(w in transcript_lower for w in main_words):
        return True
    if main in transcript_lower:
        return True
    return False


def _safe(text: str) -> str:
    """Strip characters that Helvetica (latin-1) can't render."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


class PDFReport(FPDF):
    """Custom PDF with header/footer."""

    def __init__(self, patient_name: str, seed):
        super().__init__(orientation="P", unit="mm", format="Letter")
        self.patient_name = patient_name
        self.seed = seed
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_fill_color(*COL_HEADER_BG)
        self.rect(0, 0, self.w, 14, "F")
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*COL_HEADER_FG)
        self.set_xy(8, 3)
        self.cell(0, 8, f"v3 Synthetic Transcript Report  |  {self.patient_name}  |  Seed: {self.seed}")
        self.ln(12)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*COL_MUTED)
        self.cell(0, 10, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_fill_color(*COL_SECTION_BG)
        self.set_text_color(*COL_SECTION_FG)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, f"  {title}", fill=True)
        self.ln(10)

    def label_value(self, label: str, value: str):
        y = self.get_y()
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*COL_LABEL)
        self.set_x(self.l_margin)
        self.cell(35, 5, label + ":", align="R")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*COL_VALUE)
        self.set_xy(self.l_margin + 37, y)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 37, 5, _safe(value))

    def bullet(self, text: str, colour: tuple = COL_VALUE, indent: float = 12):
        self.set_x(self.l_margin + indent)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*colour)
        available = self.w - self.l_margin - self.r_margin - indent - 2
        self.multi_cell(available, 4.5, _safe(f"  -  {text}"))


def build_pdf(scenario: dict, transcript: str) -> bytes:
    """Build the PDF and return raw bytes."""
    demo = scenario.get("demographics", {})
    present = scenario.get("present_symptoms", [])
    negated = scenario.get("negated_symptoms", [])
    chief = scenario.get("chief_complaint", "")
    noise = scenario.get("ccda_noise", {})
    seed = scenario.get("seed", "N/A")
    case_type = scenario.get("case_type", "")
    diff_system = scenario.get("differential_system") or ""
    case_label = _CASE_TYPE_LABELS.get(case_type, case_type.replace("_", " ").title())
    if case_type == "differential" and diff_system:
        case_label += f"  —  {diff_system.title()}"
    patient_name = demo.get("name", "Unknown")
    transcript_lower = transcript.lower()

    pdf = PDFReport(patient_name, seed)
    pdf.alias_nb_pages()
    pdf.add_page()

    # ═══ DEMOGRAPHICS ═══════════════════════════════════════════════
    pdf.section_title("Patient Demographics")
    for label, val in [
        ("Name",      demo.get("name", "?")),
        ("Age",       str(demo.get("age", "?"))),
        ("Gender",    "Female" if demo.get("gender") == "F" else "Male"),
        ("Race",      demo.get("race", "?")),
        ("Ethnicity", demo.get("ethnicity", "?")),
        ("Location",  f"{demo.get('city', '?')}, {demo.get('state', '?')}"),
        ("Case Type", case_label),
    ]:
        pdf.label_value(label, val)
    pdf.ln(3)

    # ═══ CHIEF COMPLAINT ════════════════════════════════════════════
    pdf.section_title("Chief Complaint")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(230, 120, 0)
    pdf.set_x(pdf.l_margin + 8)
    pdf.multi_cell(0, 5, _safe(chief))
    pdf.ln(3)

    # ═══ PRESENT SYMPTOMS ═══════════════════════════════════════════
    if case_type != "healthy" and present:
        pdf.section_title(f"Present Symptoms  ({len(present)})")
        for s in present:
            pdf.bullet(s)
        pdf.ln(3)

    # ═══ NEGATED SYMPTOMS ═══════════════════════════════════════════
    if negated:
        pdf.section_title(f"Negated Symptoms  ({len(negated)})")
        for s in negated:
            pdf.bullet(s)
        pdf.ln(3)

    # ═══ CCDA BACKGROUND NOISE ══════════════════════════════════════
    noise_items = []
    for label, key in [
        ("Conditions",     "conditions"),
        ("Medications",    "medications"),
        ("Allergies",      "allergies"),
        ("Social History", "social_history"),
        ("Immunizations",  "immunizations"),
    ]:
        items = noise.get(key, [])
        if items:
            noise_items.append((label, items))
    if case_type != "healthy":
        items = noise.get("injected_symptoms", [])
        if items:
            noise_items.append(("Injected Symptoms", items))

    if noise_items:
        pdf.section_title("CCDA Background Noise")
        for label, items in noise_items:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*COL_LABEL)
            pdf.set_x(pdf.l_margin + 6)
            pdf.cell(0, 5, f"{label} ({len(items)}):")
            pdf.ln(5)
            for item in items:
                pdf.bullet(str(item), indent=14)
        pdf.ln(3)

    # ═══ TRANSCRIPT ═════════════════════════════════════════════════
    pdf.add_page()
    t_lines = [l for l in transcript.strip().split("\n") if l.strip()]
    d_count = sum(1 for l in t_lines if l.startswith("D:"))
    p_count = sum(1 for l in t_lines if l.startswith("P:"))

    pdf.section_title(f"Transcript  ({len(t_lines)} lines  |  D: {d_count}  P: {p_count})")

    for line in transcript.strip().split("\n"):
        stripped = line.strip()
        if not stripped:
            pdf.ln(2)
            continue

        if stripped.startswith("D:"):
            pdf.set_fill_color(*COL_DOCTOR_BG)
            pdf.set_text_color(*COL_DOCTOR_TXT)
            pdf.set_font("Helvetica", "B", 8.5)
            x = pdf.l_margin + 4
            pdf.set_x(x)
            w = pdf.w - pdf.l_margin - pdf.r_margin - 8
            # Write the "D:" label then the rest
            text = stripped[2:].strip()
            pdf.multi_cell(w, 4.5, _safe(f"D:  {text}"), fill=True)

        elif stripped.startswith("P:"):
            pdf.set_fill_color(*COL_PATIENT_BG)
            pdf.set_text_color(*COL_PATIENT_TXT)
            pdf.set_font("Helvetica", "", 8.5)
            x = pdf.l_margin + 4
            pdf.set_x(x)
            w = pdf.w - pdf.l_margin - pdf.r_margin - 8
            text = stripped[2:].strip()
            pdf.multi_cell(w, 4.5, _safe(f"P:  {text}"), fill=True)

        else:
            pdf.set_text_color(*COL_MUTED)
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(0, 4, _safe(stripped))

    return pdf.output()


# ─── File discovery ──────────────────────────────────────────────────

def _find_latest_pair(output_dir: str) -> tuple[str, str]:
    scenarios = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("scenario_v3") and f.endswith(".json")],
        key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
        reverse=True,
    )
    for sc_file in scenarios:
        sc_path = os.path.join(output_dir, sc_file)
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

    parser = argparse.ArgumentParser(description="v3 PDF Report Generator")
    parser.add_argument("--scenario", default="", help="Path to scenario_v3.json")
    parser.add_argument("--transcript", default="", help="Path to generated transcript .txt")
    parser.add_argument("--out", default="", help="Output PDF path")
    args = parser.parse_args()

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
        print("WARNING: No transcript file found — PDF will show profile only.")

    pdf_bytes = build_pdf(scenario, transcript)

    patient_name = scenario.get("demographics", {}).get("name", "Unknown").replace(" ", "_")
    seed = scenario.get("seed", "0")
    default_name = f"v3_report_{patient_name}_{seed}.pdf"

    out_path = args.out or os.path.join(OUTPUT_DIR, default_name)
    with open(out_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"Saved PDF:  {out_path}")

    # Open in default PDF viewer
    import subprocess
    subprocess.Popen(["start", "", out_path], shell=True)
    print("Opened in default viewer.")


if __name__ == "__main__":
    main()
