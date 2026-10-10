"""Render the spherical two-view evidence manuscript as a publication PDF."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "MANUSCRIPT.md"
EVIDENCE = ROOT / "evidence.json"
OUTPUT = ROOT.parents[2] / "output" / "pdf" / "panorai_spherical_two_view_rt_evidence.pdf"

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2C6E9F")
TEAL = colors.HexColor("#2A9D8F")
GOLD = colors.HexColor("#E9A23B")
RED = colors.HexColor("#C84C4C")
INK = colors.HexColor("#17232D")
GRAY = colors.HexColor("#687783")
LIGHT = colors.HexColor("#EAF0F4")
VERY_LIGHT = colors.HexColor("#F7F9FA")


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        ),
        (
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica Oblique.ttf",
        ),
    ]
    for regular, bold, italic in candidates:
        if all(Path(path).exists() for path in (regular, bold, italic)):
            pdfmetrics.registerFont(TTFont("PaperSans", regular))
            pdfmetrics.registerFont(TTFont("PaperSans-Bold", bold))
            pdfmetrics.registerFont(TTFont("PaperSans-Italic", italic))
            return "PaperSans", "PaperSans-Bold", "PaperSans-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


class PaperDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id="paper", frames=[frame], onPage=self.draw_page))

    def draw_page(self, canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(colors.HexColor("#DCE3E8"))
            canvas.setLineWidth(0.6)
            canvas.line(self.leftMargin, A4[1] - 14 * mm, A4[0] - self.rightMargin, A4[1] - 14 * mm)
            canvas.setFont(FONT, 7.2)
            canvas.setFillColor(GRAY)
            canvas.drawString(self.leftMargin, A4[1] - 11.5 * mm, "PanorAi spherical two-view R,t evidence")
            canvas.drawRightString(A4[0] - self.rightMargin, A4[1] - 11.5 * mm, "Retrospective - prospective confirmation pending")
        canvas.setFont(FONT, 7.4)
        canvas.setFillColor(GRAY)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, str(doc.page))
        canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName=FONT_BOLD, fontSize=25, leading=29, textColor=NAVY, alignment=TA_LEFT, spaceAfter=10),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"], fontName=FONT, fontSize=10.5, leading=14, textColor=GRAY, spaceAfter=4),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName=FONT_BOLD, fontSize=15, leading=18, textColor=NAVY, spaceBefore=13, spaceAfter=6, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName=FONT_BOLD, fontSize=11.5, leading=14, textColor=BLUE, spaceBefore=10, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName=FONT, fontSize=9.0, leading=12.1, textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6.5),
        "abstract": ParagraphStyle("abstract", parent=base["BodyText"], fontName=FONT, fontSize=9.0, leading=12.2, textColor=INK, alignment=TA_JUSTIFY, leftIndent=8 * mm, rightIndent=8 * mm, spaceAfter=8),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontName=FONT_ITALIC, fontSize=7.7, leading=9.5, textColor=GRAY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=8),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName=FONT, fontSize=8.9, leading=11.8, textColor=INK, leftIndent=8 * mm, firstLineIndent=-4 * mm, spaceAfter=3.5),
        "formula": ParagraphStyle("formula", parent=base["Code"], fontName="Courier", fontSize=8.5, leading=11, textColor=NAVY, backColor=VERY_LIGHT, borderColor=colors.HexColor("#D7E0E6"), borderWidth=0.6, borderPadding=7, spaceBefore=4, spaceAfter=8),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontName=FONT, fontSize=7.6, leading=9.4, textColor=GRAY, alignment=TA_LEFT),
        "callout": ParagraphStyle("callout", parent=base["BodyText"], fontName=FONT_BOLD, fontSize=10, leading=13.5, textColor=NAVY, backColor=colors.HexColor("#F0F6FA"), borderColor=BLUE, borderWidth=0.8, borderPadding=10, spaceBefore=8, spaceAfter=10),
    }


def inline_markup(text: str) -> str:
    escaped = escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    return escaped


def make_table(rows: list[list[str]], doc_width: float, style_map: dict) -> Table:
    ncols = len(rows[0])
    first = min(42 * mm, doc_width * 0.28)
    if ncols == 2:
        widths = [first, doc_width - first]
    else:
        widths = [first] + [(doc_width - first) / (ncols - 1)] * (ncols - 1)
    rendered = []
    for row_index, row in enumerate(rows):
        row_style = ParagraphStyle(
            f"table-{row_index}",
            parent=style_map["small"],
            fontName=FONT_BOLD if row_index == 0 else FONT,
            textColor=colors.white if row_index == 0 else INK,
            alignment=TA_LEFT if row_index == 0 else TA_CENTER,
        )
        rendered.append([Paragraph(inline_markup(cell), row_style) for cell in row])
    table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D1D8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, VERY_LIGHT]),
            ]
        )
    )
    return table


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if index == start + 1 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            index += 1
            continue
        rows.append(cells)
        index += 1
    return rows, index


def image_flowable(path: Path, caption: str, doc_width: float, style_map: dict):
    with PILImage.open(path) as image:
        width_px, height_px = image.size
    aspect = height_px / width_px
    width = doc_width
    height = width * aspect
    max_height = 128 * mm
    if height > max_height:
        height = max_height
        width = height / aspect
    return KeepTogether([Image(str(path), width=width, height=height), Paragraph(inline_markup(caption), style_map["caption"])])


def manuscript_story(text: str, doc_width: float, style_map: dict):
    lines = text.splitlines()
    story = []
    index = 0
    paragraph_buffer: list[str] = []
    abstract_mode = False

    def flush_paragraph():
        nonlocal paragraph_buffer
        if not paragraph_buffer:
            return
        raw = " ".join(part.strip() for part in paragraph_buffer).strip()
        paragraph_buffer = []
        if not raw:
            return
        if raw.startswith("`") and raw.endswith("`") and raw.count("`") == 2:
            story.append(Paragraph(escape(raw[1:-1]), style_map["formula"]))
        else:
            story.append(Paragraph(inline_markup(raw), style_map["abstract"] if abstract_mode else style_map["body"]))

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if index < 4:
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            heading = stripped[3:]
            abstract_mode = heading == "Abstract"
            if heading in {"5. Results", "8. Release boundary and prospective experiment", "References"}:
                story.append(PageBreak())
            story.append(Paragraph(inline_markup(heading), style_map["h1"]))
            index += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            abstract_mode = False
            story.append(Paragraph(inline_markup(stripped[4:]), style_map["h2"]))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[(.+?)\]\((.+?)\)", stripped)
        if image_match:
            flush_paragraph()
            caption, relative = image_match.groups()
            story.append(image_flowable(ROOT / relative, caption, doc_width, style_map))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|"):
            flush_paragraph()
            rows, index = parse_table(lines, index)
            story.extend([Spacer(1, 2), make_table(rows, doc_width, style_map), Spacer(1, 8)])
            continue
        if re.match(r"^\d+\. ", stripped):
            flush_paragraph()
            number, body = stripped.split(". ", 1)
            story.append(Paragraph(f"<b>{number}.</b> {inline_markup(body)}", style_map["bullet"]))
            index += 1
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            story.append(Paragraph(f"<font color='#2A9D8F'>-</font> {inline_markup(stripped[2:])}", style_map["bullet"]))
            index += 1
            continue
        paragraph_buffer.append(stripped)
        index += 1
    flush_paragraph()
    return story


def cover_story(data: dict, style_map: dict, doc_width: float):
    population = data["population"]
    pooled = data["post_policy"]["pooled"]
    return [
        Spacer(1, 14 * mm),
        Paragraph("PANORAI TECHNICAL EVIDENCE PAPER", style_map["subtitle"]),
        HRFlowable(width="100%", thickness=2.2, color=TEAL, spaceBefore=2, spaceAfter=14),
        Paragraph(inline_markup(data["study"]["title"]), style_map["title"]),
        Spacer(1, 4 * mm),
        Paragraph("Robinson Luiz Souza Garcia", ParagraphStyle("author", parent=style_map["subtitle"], fontName=FONT_BOLD, fontSize=12, textColor=INK)),
        Paragraph("9 October 2026 | PanorAi 3.5.0 | spherical two-view geometry", style_map["subtitle"]),
        Spacer(1, 10 * mm),
        Table(
            [
                [Paragraph("4,017", ParagraphStyle("metric", parent=style_map["h1"], alignment=TA_CENTER, textColor=NAVY)), Paragraph("2,385", ParagraphStyle("metric2", parent=style_map["h1"], alignment=TA_CENTER, textColor=NAVY)), Paragraph("98.1%", ParagraphStyle("metric3", parent=style_map["h1"], alignment=TA_CENTER, textColor=TEAL)), Paragraph("94.2%", ParagraphStyle("metric4", parent=style_map["h1"], alignment=TA_CENTER, textColor=TEAL))],
                [Paragraph("unique panoramas", style_map["small"]), Paragraph("two-view pairs", style_map["small"]), Paragraph("selected precision", style_map["small"]), Paragraph("one-sided 95% lower", style_map["small"])],
            ],
            colWidths=[doc_width / 4] * 4,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), VERY_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#D7E0E6")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7E0E6")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        ),
        Spacer(1, 12 * mm),
        Paragraph(
            "RETROSPECTIVE EVIDENCE - PROSPECTIVE CONFIRMATION PENDING",
            ParagraphStyle("status", parent=style_map["callout"], alignment=TA_CENTER, textColor=RED, backColor=colors.HexColor("#FBECEC"), borderColor=RED),
        ),
        Spacer(1, 6 * mm),
        Paragraph(
            f"The held-out rule selects {pooled['selected']} of {pooled['pairs']} pairs and returns {pooled['true_positive']} precise poses. The pooled result clears the target, but two domains do not yet have a 90% lower confidence bound. The paper therefore defines an operating boundary and a prospective test, not a release guarantee.",
            style_map["callout"],
        ),
        Spacer(1, 8 * mm),
        Paragraph(
            f"Evidence base: {population['total_independence_groups']} independent groups across three anonymized domains. Scope: exactly two central equirectangular panoramas. Translation direction only; no dense stereo or multiview optimization.",
            style_map["subtitle"],
        ),
        PageBreak(),
    ]


def main() -> None:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    source = MANUSCRIPT.read_text(encoding="utf-8")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    style_map = styles()
    doc = PaperDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=20 * mm,
        bottomMargin=17 * mm,
        title=data["study"]["title"],
        author="Robinson Luiz Souza Garcia",
        subject="Spherical two-view R,t estimation evidence",
        creator="PanorAi reproducible paper renderer",
    )
    story = cover_story(data, style_map, doc.width)
    story.extend(manuscript_story(source, doc.width, style_map))
    doc.build(story)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"Wrote {OUTPUT}")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()
