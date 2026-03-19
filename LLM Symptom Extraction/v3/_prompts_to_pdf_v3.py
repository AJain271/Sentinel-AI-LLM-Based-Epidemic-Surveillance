"""
Generate clean, readable PDFs for the Zero-Shot and Few-Shot prompts.
Uses fpdf2 for proper text wrapping, automatic page breaks, and clear typography.
"""
import re
from fpdf import FPDF
from extract_zeroshot import build_prompt as zs_prompt
from extract_fewshot import build_prompt as fs_prompt

DUMMY = "(transcript would go here)"

L_MARGIN = 20
R_MARGIN = 20
T_MARGIN = 20
BODY_LINE_H = 5.2          # line height for body text (mm)
SECTION_LINE_H = 5.0       # line height inside sub-section bodies

# ─── Colour palette ──────────────────────────────────────────────────────────
COL_TITLE_BAR   = (34,  51,  97)   # dark navy  – top-level section banners
COL_SUB_BAR     = (70, 100, 160)   # medium blue – sub-section (PART 1, etc.)
COL_RULE        = (140, 150, 180)  # soft grey-blue rule
COL_BODY        = (25,  25,  25)   # near-black body text
COL_HEADER_TEXT = (255, 255, 255)  # white text on banners


# ─── Unicode clean-up ────────────────────────────────────────────────────────
# Built-in PDF fonts are Latin-1; strip/replace chars they can't handle.

def _sanitise(text: str) -> str:
    # Box-drawing dividers → sentinel we handle separately
    text = re.sub(r'[═]{4,}', "<<<DIVIDER>>>", text)
    text = re.sub(r'[\u2500-\u257F]{4,}', "<<<DIVIDER>>>", text)  # ─ etc.
    text = text.replace('\u2550', '=')
    text = text.replace('\u2022', '-')   # bullet
    text = text.replace('\u2019', "'")   # right single quote
    text = text.replace('\u2018', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = text.replace('\u2014', '--')  # em dash
    text = text.replace('\u2013', '-')   # en dash
    # drop anything still outside Latin-1
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text


# ─── PDF class ───────────────────────────────────────────────────────────────

class PromptPDF(FPDF):
    def __init__(self, doc_title: str):
        super().__init__('P', 'mm', 'A4')
        self.doc_title = doc_title
        self.set_margins(L_MARGIN, T_MARGIN, R_MARGIN)
        self.set_auto_page_break(auto=True, margin=18)

    # Running header / footer ---------------------------------------------------
    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 5, self.doc_title, align="R")
        self.ln(4)
        self.set_text_color(*COL_BODY)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")
        self.set_text_color(*COL_BODY)

    # Layout helpers ------------------------------------------------------------
    def _rule(self, thickness=0.3, color=COL_RULE):
        self.set_draw_color(*color)
        self.set_line_width(thickness)
        y = self.get_y()
        self.line(L_MARGIN, y, self.w - R_MARGIN, y)
        self.ln(3)

    def _banner(self, label: str, color=COL_TITLE_BAR):
        """Full-width filled banner with white text."""
        self.ln(2)
        self.set_fill_color(*color)
        self.set_text_color(*COL_HEADER_TEXT)
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 7, f"  {label}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*COL_BODY)
        self.ln(2)

    def _sub_banner(self, label: str):
        self._banner(label, color=COL_SUB_BAR)

    def _body(self, text: str, mono=False, font_size=9.5):
        font = "Courier" if mono else "Helvetica"
        self.set_font(font, "", font_size)
        self.set_text_color(*COL_BODY)
        self.multi_cell(0, BODY_LINE_H, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    # Public render method ------------------------------------------------------
    def render_prompt(self, sys_msg: str, user_msg: str):
        """Render system message then user message onto pages."""

        # ── Title bar (full width) ─────────────────────────────────────────
        self.add_page()
        self.set_fill_color(*COL_TITLE_BAR)
        self.set_text_color(*COL_HEADER_TEXT)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 12, f"  {self.doc_title}", fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*COL_BODY)
        self.ln(5)

        # ── System message ─────────────────────────────────────────────────
        self._banner("SYSTEM MESSAGE")
        self._body(_sanitise(sys_msg))
        self._rule()

        # ── User message ───────────────────────────────────────────────────
        self._banner("USER MESSAGE")
        self._render_user_msg(user_msg)

    def _render_user_msg(self, user_msg: str):
        """
        Render the user message, turning ════ dividers into visual rules and
        the text immediately after each divider into a sub-section banner.
        """
        clean = _sanitise(user_msg)
        # Split on the divider sentinel
        parts = clean.split("<<<DIVIDER>>>")

        for i, part in enumerate(parts):
            part = part.strip("\n")
            if not part:
                continue

            # The first non-empty line after a divider is a sub-section title
            if i > 0:
                lines = part.split("\n", 1)
                title_line = lines[0].strip()
                rest = lines[1].lstrip("\n") if len(lines) > 1 else ""
                if title_line:
                    self._sub_banner(title_line)
                if rest.strip():
                    self._body(rest.strip())
            else:
                # Text before the first divider (the opening paragraph)
                if part.strip():
                    self._body(part.strip())


# ─── Entry point ─────────────────────────────────────────────────────────────

def make_pdf(title: str, sys_msg: str, user_msg: str, out_path: str):
    pdf = PromptPDF(title)
    pdf.render_prompt(sys_msg, user_msg)
    pdf.output(out_path)


# Zero-Shot
sys_msg, user_msg = zs_prompt(DUMMY)
make_pdf("Zero-Shot Prompt", sys_msg, user_msg, "zero_shot_prompt.pdf")

# Few-Shot
sys_msg2, user_msg2 = fs_prompt(DUMMY)
make_pdf("Few-Shot Prompt", sys_msg2, user_msg2, "few_shot_prompt.pdf")

print("PDFs saved: zero_shot_prompt.pdf, few_shot_prompt.pdf")
