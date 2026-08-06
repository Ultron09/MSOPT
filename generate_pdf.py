"""
Publication-Grade PDF Generator for Department Research Proposal
================================================================
Generates an executive, highly polished, graphical academic PDF for the proposal.
Eliminates all raw text artifacts and replaces code diagrams with clean ReportLab Flowables.
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(PROJECT_ROOT, "RESEARCH_PROPOSAL.md")
PDF_PATH = os.path.join(PROJECT_ROOT, "RESEARCH_PROPOSAL.pdf")

# Custom Canvas for Header/Footer & Page Numbering
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#4A5568"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(43, 755, "UNIVERSAL AI UNIVERSITY — DEPARTMENT OF FUTURE TECH")
            self.drawRightString(569, 755, "Suryaansh Prithvijit Singh | MSOPT Proposal")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.75)
            self.line(43, 748, 569, 748)
            
        # Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.75)
        self.line(43, 42, 569, 42)
        
        self.drawString(43, 28, "CONFIDENTIAL — FOR DEPARTMENTAL REVIEW & FACULTY APPROVAL ONLY")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(569, 28, page_text)
        self.restoreState()


def build_graphical_pipeline(styles):
    """Build a graphical 6-stage architecture pipeline flowable."""
    stages = [
        ("STAGE 1: RAW FINANCIAL MULTIVARIATE SERIES", "Log Returns, Parkinson Volatility, Relative Volume", "#1A365D", "#EBF8FF"),
        ("STAGE 2: DENSE MULTI-SCALE DILATED RECEPTIVE FIELDS", "Window w in {4, 8, 16, 32}, Dilation d in {1, 2, 4}, Stride s = 1", "#2B6CB0", "#EBF8FF"),
        ("STAGE 3: THRESHOLDED 1D-SAX SYMBOLIC DISCRETIZATION", "Segment Mean Quantization a_mu + Segment Slope Quantization a_beta", "#2C7A7B", "#E6FFFA"),
        ("STAGE 4: 2D SCALE-TIME SPATIAL TENSOR MAPPING", "Y-axis = Receptive Field Scale (w,d), X-axis = Time Index t", "#6B46C1", "#FAF5FF"),
        ("STAGE 5: 2D SPATIAL CONVOLUTION & TRANSFORMER ENCODER", "Position + Scale + Volatility Multi-Dimensional Embeddings", "#2C5282", "#EBF8FF"),
        ("STAGE 6: DIRECTIONAL THRESHOLD & REGIME CLASSIFICATION", "High-SNR Target Formulation (Directional Move +-0.5 Vol & Vol Regimes)", "#22543D", "#F0FFF4"),
    ]
    
    flowables = []
    
    for i, (title, desc, header_color, bg_color) in enumerate(stages):
        title_p = Paragraph(f"<b>{title}</b>", ParagraphStyle(f'ST_{i}', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.white))
        desc_p = Paragraph(desc, ParagraphStyle(f'SD_{i}', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor("#2D3748")))
        
        card_table = Table([[title_p], [desc_p]], colWidths=[510])
        card_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), colors.HexColor(header_color)),
            ('BACKGROUND', (0,1), (0,1), colors.HexColor(bg_color)),
            ('PADDING', (0,0), (-1,-1), 4),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor(header_color)),
        ]))
        
        flowables.append(card_table)
        
        # Down Arrow Connector (between stages)
        if i < len(stages) - 1:
            arrow_p = Paragraph("<b>│<br/>▼</b>", ParagraphStyle(f'Arr_{i}', fontName='Helvetica-Bold', fontSize=9, leading=9, textColor=colors.HexColor("#4A5568"), alignment=TA_CENTER))
            flowables.append(Spacer(1, 1))
            flowables.append(arrow_p)
            flowables.append(Spacer(1, 1))
            
    return KeepTogether(flowables)


def build_graphical_timeline(styles):
    """Build a graphical 4-phase roadmap table."""
    title_style = ParagraphStyle('THead', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.white)
    badge_style = ParagraphStyle('TBadge', fontName='Helvetica-Bold', fontSize=8, leading=10, textColor=colors.white, alignment=TA_CENTER)
    cell_style = ParagraphStyle('TCell', fontName='Helvetica', fontSize=8, leading=11, textColor=colors.HexColor("#2D3748"))
    cell_bold = ParagraphStyle('TCellB', fontName='Helvetica-Bold', fontSize=8.5, leading=11, textColor=colors.HexColor("#1A365D"))

    data = [
        [
            Paragraph("<b>Phase & Horizon</b>", title_style),
            Paragraph("<b>Milestones & Key Deliverables</b>", title_style),
            Paragraph("<b>Status</b>", title_style)
        ],
        [
            Paragraph("<b>PHASE 1</b><br/><font size=7 color='#718096'>Literature & Blueprint</font>", cell_bold),
            Paragraph("Deep-dive literature knowledge base across 8 core domains (`research_papers/`), formal Department Research Proposal compilation, and workspace agent memory system setup.", cell_style),
            Paragraph("<font color='#22543D'><b>COMPLETED</b></font>", ParagraphStyle('B1', fontName='Helvetica-Bold', fontSize=8, leading=10, alignment=TA_CENTER))
        ],
        [
            Paragraph("<b>PHASE 2</b><br/><font size=7 color='#718096'>Weeks 1 – 2</font>", cell_bold),
            Paragraph("Formal mathematical specification of 1D-SAX codebook & 2D scale-time spatial grid. Setup of baseline walk-forward backtest protocol environment.", cell_style),
            Paragraph("<font color='#2B6CB0'><b>PLANNED</b></font>", ParagraphStyle('B2', fontName='Helvetica-Bold', fontSize=8, leading=10, alignment=TA_CENTER))
        ],
        [
            Paragraph("<b>PHASE 3</b><br/><font size=7 color='#718096'>Weeks 3 – 5</font>", cell_bold),
            Paragraph("Implementation of MSOPT Tokenizer & 2D Scale-Time Spatial Embedder (`src/tokenizer/`). Controlled ablation studies (MSOPT vs PatchTST vs TS-BPE vs TimesNet vs BORF).", cell_style),
            Paragraph("<font color='#6B46C1'><b>PLANNED</b></font>", ParagraphStyle('B3', fontName='Helvetica-Bold', fontSize=8, leading=10, alignment=TA_CENTER))
        ],
        [
            Paragraph("<b>PHASE 4</b><br/><font size=7 color='#718096'>Weeks 6 – 8</font>", cell_bold),
            Paragraph("Walk-forward expanding window cross-asset backtesting (SPY, AAPL, QQQ, TLT) with explicit 5 bps transaction costs. Drafting academic manuscript for top-tier submission.", cell_style),
            Paragraph("<font color='#1A365D'><b>PLANNED</b></font>", ParagraphStyle('B4', fontName='Helvetica-Bold', fontSize=8, leading=10, alignment=TA_CENTER))
        ]
    ]

    roadmap_table = Table(data, colWidths=[110, 316, 100])
    roadmap_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1A365D")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 6),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#1A365D")),
    ]))
    
    return KeepTogether(roadmap_table)


def build_pdf():
    print(f"[PDF] Reading {MD_PATH}...")
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=43,
        rightMargin=43,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    PRIMARY = colors.HexColor("#1A365D")   # Deep Navy
    SECONDARY = colors.HexColor("#2B6CB0") # Slate Blue
    ACCENT_RED = colors.HexColor("#9B2C2C")# Dark Crimson
    TEXT_DARK = colors.HexColor("#2D3748") # Dark Slate
    BG_LIGHT = colors.HexColor("#F7FAFC")  # Light Gray
    BG_CALLOUT = colors.HexColor("#FFF5F5")# Soft Red Tint

    # Custom Typography Styles
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=5,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=SECONDARY,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
        spaceAfter=5,
        alignment=TA_LEFT
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
        leftIndent=12,
        spaceAfter=3
    )

    callout_title_style = ParagraphStyle(
        'CalloutTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=ACCENT_RED,
        spaceAfter=4
    )

    callout_body_style = ParagraphStyle(
        'CalloutBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#742A2A"),
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=TA_LEFT
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=TEXT_DARK
    )

    story = []

    # Parse Header Metadata
    story.append(Paragraph("UNIVERSAL AI UNIVERSITY", ParagraphStyle('InstHeader', fontName='Helvetica-Bold', fontSize=10, textColor=SECONDARY, leading=12, spaceAfter=2)))
    story.append(Paragraph("DEPARTMENT OF FUTURE TECH — RESEARCH PROPOSAL", ParagraphStyle('DeptSub', fontName='Helvetica-Bold', fontSize=12, textColor=PRIMARY, leading=15, spaceAfter=6)))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceBefore=0, spaceAfter=10))

    # Metadata Table Block
    meta_data = [
        [Paragraph("Project Title:", ParagraphStyle('ML', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=PRIMARY)), Paragraph("Multi-Scale Overlapping Pattern Tokenization (MSOPT) for Non-Stationary Financial Time Series", ParagraphStyle('MV', fontName='Helvetica', fontSize=9, leading=12, textColor=TEXT_DARK))],
        [Paragraph("Lead Researcher:", ParagraphStyle('ML', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=PRIMARY)), Paragraph("<b>Suryaansh Prithvijit Singh</b> (Universal AI University)", ParagraphStyle('MV', fontName='Helvetica', fontSize=9, leading=12, textColor=TEXT_DARK))],
        [Paragraph("Faculty Guide:", ParagraphStyle('ML', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=PRIMARY)), Paragraph("<b>Prof. Shivaji Pawar</b>, Department of Future Tech, Universal AI University", ParagraphStyle('MV', fontName='Helvetica', fontSize=9, leading=12, textColor=TEXT_DARK))],
        [Paragraph("Date & Status:", ParagraphStyle('ML', fontName='Helvetica-Bold', fontSize=9, leading=12, textColor=PRIMARY)), Paragraph("August 2026 | Proposal Pending Formal Department Approval", ParagraphStyle('MV', fontName='Helvetica', fontSize=9, leading=12, textColor=TEXT_DARK))],
    ]
    meta_table = Table(meta_data, colWidths=[100, 426])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('PADDING', (0,0), (-1,-1), 4),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Core Problem Statement Box
    problem_box_data = [
        [Paragraph("🔴 CORE PROBLEM STATEMENT", callout_title_style)],
        [Paragraph("<b>EXISTING TIME SERIES DEEP LEARNING ARCHITECTURES SUFFER FROM A FUNDAMENTAL PARADIGM FAILURE WHEN APPLIED TO NON-STATIONARY FINANCIAL MARKETS:</b>", callout_body_style)],
        [Paragraph("<b>1. THE 1D SERIAL POINTWISE BLINDSPOT:</b> Serial models (LSTMs, vanilla Transformers) evaluate scalar price points <i>x_t</i> sequentially. This destroys local visual shape context (inflections, wedges, double bottoms) and causes quadratic computational explosion <i>O(T²)</i>, preventing long historical context processing.", callout_body_style)],
        [Paragraph("<b>2. THE RIGID UNIFORM PATCHING BOTTLENECK:</b> Modern patch transformers (PatchTST) force rigid uniform sequence slicing (<i>P=16</i>). In non-stationary markets, fixed boundaries clip pattern inflections arbitrarily, enforce single-scale rigidity, and lack translation invariance—rendering them fragile during market regime shifts.", callout_body_style)],
        [Paragraph("<b>3. THE QUANTITATIVE VISUAL GAP:</b> Human quantitative traders analyze market dynamics visually as <b>multi-scale 2D spatial chart primitives</b> (micro-spikes, daily consolidations, macro regimes). Existing machine learning paradigms force 1D serial vectors or rigid 1D uniform grids, failing to extract localized multi-scale visual tokens.", callout_body_style)],
    ]
    problem_table = Table(problem_box_data, colWidths=[526])
    problem_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_CALLOUT),
        ('PADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 1.25, ACCENT_RED),
        ('LINEBELOW', (0,0), (-1,0), 0.5, ACCENT_RED),
    ]))
    story.append(problem_table)
    story.append(Spacer(1, 10))

    # Clean markdown helper
    def clean_text(t):
        t = t.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
        t = re.sub(r'\*(.*?)\*', r'<i>\1</i>', t)
        t = re.sub(r'`(.*?)`', r'<font face="Courier">\1</font>', t)
        t = re.sub(r'\$(.*?)\$', r'<i>\1</i>', t)
        return t

    # Read markdown sections
    lines = md_text.split('\n')
    in_code_block = False
    code_lines = []
    in_table = False
    table_rows = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # Skip main title, metadata, problem statement box, and empty divider lines
        if i < 8 or "CORE PROBLEM STATEMENT" in line or line.startswith(">") or line.strip() == "---":
            i += 1
            continue

        # Intercept code blocks for Section 3 (Architecture Pipeline) & Section 7 (Timeline)
        if line.startswith("```"):
            if in_code_block:
                # End of code block: check if it's pipeline or timeline
                full_code = "\n".join(code_lines)
                if "RAW FINANCIAL MULTIVARIATE SERIES" in full_code:
                    story.append(build_graphical_pipeline(styles))
                    story.append(Spacer(1, 10))
                elif "PHASE 1" in full_code:
                    story.append(build_graphical_timeline(styles))
                    story.append(Spacer(1, 10))
                else:
                    # Fallback simple code box
                    code_text = "<br/>".join([clean_text(l) for l in code_lines])
                    p_code = Paragraph(code_text, ParagraphStyle('CodeBlock', parent=styles['Normal'], fontName='Courier', fontSize=7.5, leading=10, backColor=colors.HexColor("#1A202C"), textColor=colors.HexColor("#E2E8F0"), borderPadding=6, spaceAfter=8))
                    story.append(p_code)
                    
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Tables
        if line.startswith("|") and "|" in line[1:]:
            in_table = True
            table_rows.append([cell.strip() for cell in line.split("|")[1:-1]])
            i += 1
            continue

        if in_table and not line.startswith("|"):
            # Process table
            if len(table_rows) > 1:
                headers = [Paragraph(clean_text(c), table_header_style) for c in table_rows[0]]
                # Skip separator line
                data_rows = table_rows[1:]
                if data_rows and any("---" in c for c in data_rows[0]):
                    data_rows = data_rows[1:]
                
                body_rows = []
                for r in data_rows:
                    body_rows.append([Paragraph(clean_text(c), table_cell_style) for c in r])
                
                table_data = [headers] + body_rows
                
                # Calculate column widths
                n_cols = len(headers)
                col_w = 526 / n_cols
                
                t_obj = Table(table_data, colWidths=[col_w]*n_cols)
                t_obj.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), PRIMARY),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('PADDING', (0,0), (-1,-1), 4),
                    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
                ]))
                story.append(t_obj)
                story.append(Spacer(1, 8))

            table_rows = []
            in_table = False

        # Headings
        if line.startswith("## "):
            story.append(Paragraph(clean_text(line[3:]), h1_style))
            story.append(HRFlowable(width="100%", thickness=0.8, color=SECONDARY, spaceBefore=1, spaceAfter=5))
            i += 1
            continue
        elif line.startswith("### "):
            story.append(Paragraph(clean_text(line[4:]), h2_style))
            i += 1
            continue

        # Bullet points
        if line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph(f"• {clean_text(line[2:])}", bullet_style))
            i += 1
            continue
        elif re.match(r'^\d+\.\s', line):
            content = re.sub(r'^\d+\.\s', '', line)
            story.append(Paragraph(f"• {clean_text(content)}", bullet_style))
            i += 1
            continue

        # Regular Paragraph
        if line.strip():
            story.append(Paragraph(clean_text(line), body_style))

        i += 1

    # Add Sign-Off Table at the end
    story.append(Spacer(1, 10))
    story.append(Paragraph("9. Departmental Approval & Sign-Off Request", h1_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=SECONDARY, spaceBefore=1, spaceAfter=6))
    story.append(Paragraph("We respectfully request departmental review and approval to proceed with the formal research program outlined above.", body_style))
    story.append(Spacer(1, 6))

    signoff_data = [
        [Paragraph("<b>Role</b>", table_header_style), Paragraph("<b>Name & Institution</b>", table_header_style), Paragraph("<b>Signature</b>", table_header_style), Paragraph("<b>Date</b>", table_header_style)],
        [Paragraph("Lead Researcher / Author", table_cell_style), Paragraph("<b>Suryaansh Prithvijit Singh</b><br/>Universal AI University", table_cell_style), Paragraph("<br/><br/>_______________________", table_cell_style), Paragraph("<br/><br/>___ / ___ / 2026", table_cell_style)],
        [Paragraph("Faculty Guide / Supervisor", table_cell_style), Paragraph("<b>Prof. Shivaji Pawar</b><br/>Faculty Guide, Dept. of Future Tech<br/>Universal AI University", table_cell_style), Paragraph("<br/><br/>_______________________", table_cell_style), Paragraph("<br/><br/>___ / ___ / 2026", table_cell_style)],
    ]
    signoff_table = Table(signoff_data, colWidths=[120, 206, 120, 80])
    signoff_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    
    story.append(KeepTogether(signoff_table))

    # Build PDF
    print(f"[PDF] Generating {PDF_PATH}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"  → PDF successfully generated: {PDF_PATH}")
    return PDF_PATH


if __name__ == "__main__":
    build_pdf()
