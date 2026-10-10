"""Render the R,t evidence manuscript in the January 2024 ISPRS format."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / "MANUSCRIPT_ISPRS.md"
OUTPUT = ROOT.parents[3] / "output" / "pdf" / "panorai_spherical_two_view_rt_isprs.pdf"

TITLE = "Probabilistic Reliability Boundaries for Spherical Two-View Relative Pose Estimation in PanorAi"
AUTHOR = "Robinson Luiz Souza Garcia"
AFFILIATION = "Independent Researcher, Brazil - rlsgarcia@icloud.com"
KEYWORDS = [
    "spherical vision",
    "equirectangular panorama",
    "relative pose",
    "overlap",
    "selective prediction",
    "calibration",
]
ABSTRACT = (
    "This paper studies the reliability boundary of spherical two-view relative-pose estimation from two central "
    "equirectangular panoramas. The evaluated PanorAi 3.5.0 route combines native spherical Difference-of-Gaussians "
    "detection, tangent-plane RootSIFT description, robust matching, essential-matrix estimation, public geometric "
    "quality checks, and a post-estimation probability of pose precision. Registered depth clouds are used only offline "
    "to define spatial overlap. The frozen retrospective census contains 4,017 unique panoramas, 2,385 unordered pairs, "
    "and 69 independence components across three anonymized domains. Precision requires rotation error at most 1 degree "
    "and oriented translation-direction error at most 5 degrees. In the largest domain, usable rate rises from 8.3% "
    "below 10% overlap to 92.0% at 50-70% and 94.1% above 70%. On 435 held-out pairs, the rule requiring public quality "
    "acceptance and post probability at least 0.90 selects 107 pairs, of which 105 are precise: precision 98.1%, recall "
    "74.5%, coverage 24.6%, and an exact one-sided 95% precision lower bound of 94.2%. Two smaller domains do not yet "
    "reach a 90% lower bound. A separate RGB overlap proxy obtains 81.1% precision but only 37.0% recall for detecting "
    "overlap of at least 50%. Controlled complete-pair medians are 8.82 s, 8.68 s, and 13.23 s by domain at 1024 x 2048 "
    "pixels. The evidence supports selective release and prospective confirmation rather than a release-reliability claim."
)

INK = colors.black
GRID = colors.HexColor("#A7A7A7")
LIGHT = colors.HexColor("#F2F2F2")


def register_times() -> None:
    font_dir = Path("/System/Library/Fonts/Supplemental")
    mapping = {
        "ISPRS-Times": font_dir / "Times New Roman.ttf",
        "ISPRS-Times-Bold": font_dir / "Times New Roman Bold.ttf",
        "ISPRS-Times-Italic": font_dir / "Times New Roman Italic.ttf",
        "ISPRS-Times-BoldItalic": font_dir / "Times New Roman Bold Italic.ttf",
    }
    if all(path.exists() for path in mapping.values()):
        for name, path in mapping.items():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        pdfmetrics.registerFontFamily(
            "ISPRS-Times",
            normal="ISPRS-Times",
            bold="ISPRS-Times-Bold",
            italic="ISPRS-Times-Italic",
            boldItalic="ISPRS-Times-BoldItalic",
        )
    else:
        raise RuntimeError("Times New Roman fonts required by ISPRS were not found")


register_times()


class ISPRSDocument(BaseDocTemplate):
    """A4 document with the exact ISPRS margins and 82 mm columns."""

    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        page_width, page_height = A4
        left = 20 * mm
        bottom = 25 * mm
        column_width = 82 * mm
        gap = 6 * mm
        usable_height = page_height - 50 * mm

        first_top_height = 91 * mm
        first_body_height = usable_height - first_top_height - 4 * mm
        first_body_y = bottom
        first_top_y = bottom + first_body_height + 4 * mm
        first_frames = [
            Frame(left, first_top_y, 170 * mm, first_top_height, id="first-title", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(left, first_body_y, column_width, first_body_height, id="first-left", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(left + column_width + gap, first_body_y, column_width, first_body_height, id="first-right", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        ]

        normal_frames = [
            Frame(left, bottom, column_width, usable_height, id="left", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(left + column_width + gap, bottom, column_width, usable_height, id="right", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        ]

        wide_top_height = 120 * mm
        wide_body_height = usable_height - wide_top_height - 4 * mm
        wide_top_y = bottom + wide_body_height + 4 * mm
        wide_frames = [
            Frame(left, wide_top_y, 170 * mm, wide_top_height, id="wide-top", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(left, bottom, column_width, wide_body_height, id="wide-left", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            Frame(left + column_width + gap, bottom, column_width, wide_body_height, id="wide-right", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        ]

        self.addPageTemplates(
            [
                PageTemplate(id="First", frames=first_frames, autoNextPageTemplate="Later"),
                PageTemplate(id="Later", frames=normal_frames),
                PageTemplate(id="WideTop", frames=wide_frames, autoNextPageTemplate="Later"),
            ]
        )


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="ISPRS-Times-Bold", fontSize=12, leading=14, alignment=TA_CENTER, textColor=INK, spaceAfter=8),
        "author": ParagraphStyle("author", parent=base["Normal"], fontName="ISPRS-Times", fontSize=9, leading=10.5, alignment=TA_CENTER, textColor=INK, spaceAfter=2),
        "affiliation": ParagraphStyle("affiliation", parent=base["Normal"], fontName="ISPRS-Times-Italic", fontSize=9, leading=10.5, alignment=TA_CENTER, textColor=INK, spaceAfter=8),
        "keywords": ParagraphStyle("keywords", parent=base["Normal"], fontName="ISPRS-Times", fontSize=9, leading=10.5, alignment=TA_LEFT, textColor=INK, spaceAfter=10),
        "abstract-heading": ParagraphStyle("abstract-heading", parent=base["Normal"], fontName="ISPRS-Times-Bold", fontSize=9, leading=10.5, alignment=TA_LEFT, textColor=INK, spaceAfter=5),
        "abstract": ParagraphStyle("abstract", parent=base["Normal"], fontName="ISPRS-Times", fontSize=9, leading=10.5, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=0),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="ISPRS-Times-Bold", fontSize=9.5, leading=11, alignment=TA_CENTER, textColor=INK, spaceBefore=7, spaceAfter=5, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="ISPRS-Times-Bold", fontSize=9, leading=10.5, alignment=TA_LEFT, textColor=INK, spaceBefore=6, spaceAfter=4, keepWithNext=True),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="ISPRS-Times", fontSize=9, leading=10.5, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5),
        "reference": ParagraphStyle("reference", parent=base["BodyText"], fontName="ISPRS-Times", fontSize=8.5, leading=9.7, alignment=TA_LEFT, textColor=INK, spaceAfter=4),
        "list": ParagraphStyle("list", parent=base["BodyText"], fontName="ISPRS-Times", fontSize=9, leading=10.5, alignment=TA_JUSTIFY, textColor=INK, leftIndent=5 * mm, firstLineIndent=-4 * mm, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontName="ISPRS-Times", fontSize=8, leading=9, alignment=TA_CENTER, textColor=INK, spaceBefore=2, spaceAfter=6),
        "table": ParagraphStyle("table", parent=base["BodyText"], fontName="ISPRS-Times", fontSize=7.5, leading=8.5, alignment=TA_CENTER, textColor=INK),
        "table-head": ParagraphStyle("table-head", parent=base["BodyText"], fontName="ISPRS-Times-Bold", fontSize=7.5, leading=8.5, alignment=TA_CENTER, textColor=INK),
        "equation": ParagraphStyle("equation", parent=base["BodyText"], fontName="ISPRS-Times-Italic", fontSize=9, leading=11, alignment=TA_CENTER, textColor=INK),
    }


def markup(text: str) -> str:
    value = escape(text)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r"<i>\1</i>", value)
    return value


def title_block(styles: dict[str, ParagraphStyle]):
    if not 100 <= len(ABSTRACT.split()) <= 250:
        raise ValueError("ISPRS abstract must contain 100-250 words")
    if len(KEYWORDS) > 6:
        raise ValueError("ISPRS permits at most six keywords")
    return [
        Paragraph(TITLE, styles["title"]),
        Paragraph(AUTHOR, styles["author"]),
        Paragraph(AFFILIATION, styles["affiliation"]),
        Paragraph(f"<b>Keywords:</b> {escape('; '.join(KEYWORDS))}", styles["keywords"]),
        Paragraph("Abstract", styles["abstract-heading"]),
        Paragraph(escape(ABSTRACT), styles["abstract"]),
        FrameBreak(),
    ]


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if index == start + 1 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            index += 1
            continue
        rows.append(cells)
        index += 1
    return rows, index


def make_table(rows: list[list[str]], width: float, styles: dict[str, ParagraphStyle]) -> Table:
    ncols = len(rows[0])
    first = width * (0.22 if ncols >= 5 else 0.30)
    widths = [first] + [(width - first) / (ncols - 1)] * (ncols - 1)
    rendered = []
    for row_index, row in enumerate(rows):
        paragraph_style = styles["table-head"] if row_index == 0 else styles["table"]
        rendered.append([Paragraph(markup(cell), paragraph_style) for cell in row])
    table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="CENTER", splitByRow=0)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def figure(path: Path, caption: str, width: float, styles: dict[str, ParagraphStyle]):
    with PILImage.open(path) as image:
        pixel_width, pixel_height = image.size
    height = width * pixel_height / pixel_width
    return KeepTogether([Image(str(path), width=width, height=height), Paragraph(markup(caption), styles["caption"])])


def equation(text: str, number: str, width: float, styles: dict[str, ParagraphStyle]) -> Table:
    table = Table(
        [[Paragraph(markup(text), styles["equation"]), Paragraph(escape(number), styles["body"])]],
        colWidths=[width - 12 * mm, 12 * mm],
    )
    table.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("ALIGN", (1, 0), (1, 0), "RIGHT"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return table


def body_story(styles: dict[str, ParagraphStyle]):
    lines = MANUSCRIPT.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    buffer: list[str] = []
    wide = False
    in_references = False

    def current_width() -> float:
        return 170 * mm if wide else 82 * mm

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = " ".join(part.strip() for part in buffer).strip()
        buffer = []
        if text:
            story.append(Paragraph(markup(text), styles["reference"] if in_references else styles["body"]))

    while index < len(lines):
        stripped = lines[index].strip()
        if index == 0 and stripped.startswith("# "):
            index += 1
            continue
        if not stripped:
            flush()
            index += 1
            continue
        if stripped == ":::widepage":
            flush()
            story.extend([NextPageTemplate("WideTop"), PageBreak()])
            wide = True
            index += 1
            continue
        if stripped == ":::newpage":
            flush()
            story.extend([NextPageTemplate("Later"), PageBreak()])
            wide = False
            index += 1
            continue
        if stripped == ":::columns":
            flush()
            story.append(FrameBreak())
            wide = False
            index += 1
            continue
        if stripped.startswith("## "):
            flush()
            heading = stripped[3:]
            in_references = heading == "References"
            story.append(Paragraph(markup(heading), styles["h1"]))
            index += 1
            continue
        if stripped.startswith("### "):
            flush()
            story.append(Paragraph(markup(stripped[4:]), styles["h2"]))
            index += 1
            continue
        match = re.fullmatch(r"!\[(.+?)\]\((.+?)\)", stripped)
        if match:
            flush()
            caption, relative = match.groups()
            story.append(figure(ROOT / relative, caption, current_width(), styles))
            index += 1
            continue
        if stripped.startswith("EQUATION: "):
            flush()
            content = stripped[len("EQUATION: ") :]
            expression, number = [part.strip() for part in content.rsplit("|", 1)]
            story.append(equation(expression, number, current_width(), styles))
            index += 1
            continue
        if stripped.startswith("TABLECAPTION: "):
            flush()
            story.append(Paragraph(markup(stripped[len("TABLECAPTION: ") :]), styles["caption"]))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and lines[index + 1].strip().startswith("|"):
            flush()
            rows, index = parse_table(lines, index)
            story.extend([make_table(rows, current_width(), styles), Spacer(1, 5)])
            continue
        if re.match(r"^\d+\. ", stripped):
            flush()
            number, body = stripped.split(". ", 1)
            story.append(Paragraph(f"{number}. {markup(body)}", styles["list"]))
            index += 1
            continue
        buffer.append(stripped)
        index += 1
    flush()
    return story


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = ISPRSDocument(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
        title=TITLE,
        author=AUTHOR,
        subject="Spherical two-view R,t estimation evidence in ISPRS format",
        creator="PanorAi ISPRS paper renderer",
    )
    story = title_block(styles)
    story.extend(body_story(styles))
    doc.build(story)
    pages = len(PdfReader(str(OUTPUT)).pages)
    if not 6 <= pages <= 8:
        raise RuntimeError(f"ISPRS full paper must be approximately 6-8 pages; rendered {pages}")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"Wrote {OUTPUT}")
    print(f"Pages {pages}")
    print(f"Abstract words {len(ABSTRACT.split())}")
    print(f"Keywords {len(KEYWORDS)}")
    print(f"SHA-256 {digest}")


if __name__ == "__main__":
    main()
