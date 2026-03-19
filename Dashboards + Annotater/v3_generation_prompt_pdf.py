"""
v3 Generation Prompt PDF
========================
Renders the transcript generation system + user prompts for a given
v3 scenario into a clean, readable PDF — same visual style as the
extraction _prompts_to_pdf_v3.py.

Usage:
    python v3_generation_prompt_pdf.py --scenario path/to/scenario_v3.json
    python v3_generation_prompt_pdf.py --scenario path/to/scenario_v3.json --out my_report.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from fpdf import FPDF

# ── Allow importing generate_transcript_v3 from the v3 generation directory ──
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
GEN_V3_DIR   = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "Synthetic Transcript Generation",
                 "synthetic generation", "v3")
)
if GEN_V3_DIR not in sys.path:
    sys.path.insert(0, GEN_V3_DIR)

from generate_transcript_v3 import build_prompt  # noqa: E402


# ─── Layout constants ────────────────────────────────────────────────────────
L_MARGIN      = 20
R_MARGIN      = 20
T_MARGIN      = 20
BODY_LINE_H   = 5.2
SECTION_LINE_H = 5.0

# ─── Colour palette (matches _prompts_to_pdf_v3 style) ──────────────────────
COL_TITLE_BAR   = (34,  51,  97)   # dark navy  – top-level banners
COL_SUB_BAR     = (70, 100, 160)   # medium blue – sub-section banners
COL_RULE        = (140, 150, 180)  # soft grey-blue rule
COL_BODY        = (25,  25,  25)   # near-black body text
COL_HEADER_TEXT = (255, 255, 255)
COL_MUTED       = (140, 140, 140)


# ─── Unicode sanitiser ───────────────────────────────────────────────────────
def _sanitise(text: str) -> str:
    text = re.sub(r'[═]{4,}', "<<<DIVIDER>>>", text)
    text = re.sub(r'[\u2500-\u257F]{4,}', "<<<DIVIDER>>>", text)
    text = text.replace('\u2550', '=')
    text = text.replace('\u2022', '-')
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2014', '--').replace('\u2013', '-')
    text = text.replace('\U0001f7e2', '[GREEN]')   # emoji → plain text
    text = text.replace('\U0001f534', '[RED]')
    text = text.replace('\u26a0', '[!]')
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


# ─── PDF class ───────────────────────────────────────────────────────────────
class GenPromptPDF(FPDF):
    def __init__(self, doc_title: str):
        super().__init__('P', 'mm', 'A4')
        self.doc_title = doc_title
        self.set_margins(L_MARGIN, T_MARGIN, R_MARGIN)
        self.set_auto_page_break(auto=True, margin=18)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*COL_MUTED)
        self.cell(0, 5, self.doc_title, align="R")
        self.ln(4)
        self.set_text_color(*COL_BODY)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*COL_MUTED)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")
        self.set_text_color(*COL_BODY)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _rule(self, thickness: float = 0.3):
        self.set_draw_color(*COL_RULE)
        self.set_line_width(thickness)
        y = self.get_y()
        self.line(L_MARGIN, y, self.w - R_MARGIN, y)
        self.ln(3)

    def _banner(self, label: str, color: tuple = COL_TITLE_BAR):
        self.ln(2)
        self.set_fill_color(*color)
        self.set_text_color(*COL_HEADER_TEXT)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, f"  {label}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*COL_BODY)
        self.ln(2)

    def _sub_banner(self, label: str):
        self._banner(label, color=COL_SUB_BAR)

    def _body(self, text: str, mono: bool = False, font_size: float = 9.5):
        font = "Courier" if mono else "Helvetica"
        self.set_font(font, "", font_size)
        self.set_text_color(*COL_BODY)
        self.multi_cell(0, BODY_LINE_H, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    # ── Prompt renderer ──────────────────────────────────────────────────────
    def _render_message(self, msg: str, use_sub_banners: bool = True):
        """
        Render a prompt string, turning ═══ dividers into visual rules and
        the text immediately after each divider into a sub-section banner.
        """
        clean = _sanitise(msg)
        parts = clean.split("<<<DIVIDER>>>")

        for i, part in enumerate(parts):
            part = part.strip("\n")
            if not part:
                continue

            if i > 0 and use_sub_banners:
                lines = part.split("\n", 1)
                title_line = lines[0].strip()
                rest = lines[1].lstrip("\n") if len(lines) > 1 else ""
                if title_line:
                    self._sub_banner(title_line)
                if rest.strip():
                    self._body(rest.strip())
            else:
                if part.strip():
                    self._body(part.strip())

    def render(self, sys_msg: str, user_msg: str):
        """Render system message then user message, each starting on a page."""

        # ── Page 1: title + system message ───────────────────────────────
        self.add_page()
        self.set_fill_color(*COL_TITLE_BAR)
        self.set_text_color(*COL_HEADER_TEXT)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 12, f"  {self.doc_title}", fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*COL_BODY)
        self.ln(5)

        self._banner("SYSTEM MESSAGE")
        self._render_message(sys_msg)
        self._rule()

        # ── User message (new page) ────────────────────────────────────
        self.add_page()
        self._banner("USER MESSAGE")
        self._render_message(user_msg, use_sub_banners=False)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _load_scenario(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

_CASE_LABELS = {
    "novel_virus":  "Novel Virus",
    "flu_like":     "Flu-Like",
    "differential": "Varied Case",
    "healthy":      "Healthy",
}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="v3 Generation Prompt PDF")
    parser.add_argument("--scenario", required=True, help="Path to scenario_v3.json")
    parser.add_argument("--out", default="", help="Output PDF path (default: next to scenario)")
    args = parser.parse_args()

    scenario_path = os.path.abspath(args.scenario)
    if not os.path.exists(scenario_path):
        print(f"ERROR: scenario not found: {scenario_path}")
        sys.exit(1)

    scenario = _load_scenario(scenario_path)
    patient  = scenario.get("demographics", {}).get("name", "Unknown")
    case_type = scenario.get("case_type", "novel_virus")
    diff_sys  = scenario.get("differential_system") or ""

    case_label = _CASE_LABELS.get(case_type, case_type.replace("_", " ").title())
    if case_type == "differential" and diff_sys:
        case_label += f" -- {diff_sys.title()}"

    doc_title = f"Generation Prompt  |  {patient}  |  {case_label}"

    print(f"Scenario : {os.path.basename(scenario_path)}")
    print(f"Patient  : {patient}  ({case_label})")
    print("Building prompts...")

    sys_msg, user_msg = build_prompt(scenario)

    print("Rendering PDF...")
    pdf = GenPromptPDF(doc_title)
    pdf.alias_nb_pages()
    pdf.render(sys_msg, user_msg)

    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(scenario_path)[0]
        out_path = base + "_generation_prompt.pdf"

    pdf.output(out_path)
    print(f"Saved   : {out_path}")

    import subprocess
    subprocess.Popen(["start", "", out_path], shell=True)


if __name__ == "__main__":
    main()
