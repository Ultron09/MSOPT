"""
Generates MSOPT_Conference_Paper.pdf using ReportLab (IEEE 2-Column Academic Style)
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(PROJECT_ROOT, "paper", "MSOPT_Conference_Paper.pdf")
os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)

class AcademicNumberedCanvas(canvas.Canvas):
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
        
        if self._pageNumber > 1:
            self.drawString(43, 755, "IEEE TRANSACTIONS ON PATTERN ANALYSIS & MACHINE INTELLIGENCE (PREPRINT 2026)")
            self.drawRightString(569, 755, "Singh & Pawar: MSOPT for Non-Stationary Financial Time Series")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.75)
            self.line(43, 748, 569, 748)
            
        self.setFont("Helvetica", 8)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.75)
        self.line(43, 42, 569, 42)
        
        self.drawString(43, 28, "CONFIDENTIAL & PROPRIETARY — ACADEMIC CONFERENCE MANUSCRIPT DRAFT")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(569, 28, page_text)
        self.restoreState()

def build_conference_pdf():
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=43,
        rightMargin=43,
        topMargin=54,
        bottomMargin=54
    )
    
    PRIMARY = colors.HexColor("#1A365D")
    SECONDARY = colors.HexColor("#2B6CB0")
    TEXT_DARK = colors.HexColor("#2D3748")
    BG_LIGHT = colors.HexColor("#F7FAFC")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=16, leading=20, textColor=PRIMARY, alignment=TA_CENTER, spaceAfter=8)
    author_style = ParagraphStyle('Author', fontName='Helvetica', fontSize=10, leading=14, textColor=SECONDARY, alignment=TA_CENTER, spaceAfter=14)
    
    h1_style = ParagraphStyle('H1', fontName='Helvetica-Bold', fontSize=11, leading=14, textColor=PRIMARY, spaceBefore=12, spaceAfter=4, keepWithNext=True)
    h2_style = ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=SECONDARY, spaceBefore=8, spaceAfter=3, keepWithNext=True)
    body_style = ParagraphStyle('Body', fontName='Helvetica', fontSize=8.5, leading=12, textColor=TEXT_DARK, spaceAfter=5, alignment=TA_JUSTIFY)
    
    abstract_title = ParagraphStyle('AbsT', fontName='Helvetica-Bold', fontSize=9.5, leading=12, textColor=PRIMARY, spaceAfter=3)
    abstract_body = ParagraphStyle('AbsB', fontName='Helvetica-Oblique', fontSize=8.5, leading=12, textColor=TEXT_DARK, spaceAfter=10)

    story = []

    # Title & Authors
    story.append(Paragraph("Multi-Scale Overlapping Pattern Tokenization (MSOPT) for Non-Stationary Financial Time Series", title_style))
    story.append(Paragraph("<b>Suryaansh Prithvijit Singh</b> (Lead Researcher) &amp; <b>Prof. Shivaji Pawar</b> (Faculty Supervisor)<br/><i>Department of Future Tech, Universal AI University</i>", author_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceBefore=0, spaceAfter=10))

    # Abstract Box
    abs_text = "Deep learning architectures for time series modeling overwhelmingly rely on 1D serial pointwise vectors or uniform non-overlapping patch partitions. When applied to non-stationary financial markets, these paradigms suffer from severe limitations: single-scale rigidity, token boundary clipping distortion, and vulnerability to low signal-to-noise ratio (SNR) regime shifts. To resolve these challenges, we introduce Multi-Scale Overlapping Pattern Tokenization (MSOPT)—a novel framework that reformulates scalar price series into a multi-scale 2D Scale-Time Spatial Tensor. MSOPT combines dense dilated receptive field extraction (w in {4,8,16,32}, d in {1,2,4}) with dense translation-invariant stride (s=1) and thresholded 1D-SAX symbolic discretization (segment mean and slope quantization) to discover human-legible visual pattern primitives ('words'). Over a rigorous 10-year walk-forward expanding window evaluation (2016–2026) across S&amp;P 500 (SPY), Nasdaq (QQQ), Apple (AAPL), and Treasury Bonds (TLT) post 5 bps transaction costs, MSOPT achieves a +71.1% Out-of-Sample Sharpe Ratio lift over standard fixed-window baselines (0.8668 vs 0.5065 on SPY) while reducing Maximum Drawdown by 50% (from -41.91% to -22.09%)."
    
    abs_data = [[Paragraph("ABSTRACT", abstract_title)], [Paragraph(abs_text, abstract_body)]]
    abs_table = Table(abs_data, colWidths=[526])
    abs_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOX', (0,0), (-1,-1), 1, SECONDARY),
    ]))
    story.append(abs_table)
    story.append(Spacer(1, 10))

    # 1. Introduction
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph("Financial time series analysis remains one of the most challenging frontiers in quantitative machine learning due to extreme non-stationarity, low signal-to-noise ratio (SNR), and non-linear regime shifts. Traditional deep sequence architectures—including Recurrent Neural Networks (RNNs), Long Short-Term Memory networks (LSTMs), and vanilla Transformers—process temporal data as a 1D sequence of continuous scalar points x_t in R. This serial approach exhibits two fundamental drawbacks: first, it incurs quadratic computational complexity O(T^2) over long temporal horizons; second, it fails to capture localized multi-scale visual shapes (e.g., micro-spikes, daily consolidations, macro breakouts) that human quantitative traders natively recognize.", body_style))
    story.append(Paragraph("Recent innovations in time series foundation models, such as PatchTST, attempt to mitigate sequence length by aggregating raw scalar values into contiguous uniform patches (P=16). However, in financial markets, uniform non-overlapping patching forces rigid partition boundaries that clip pattern inflections arbitrarily. Furthermore, non-overlapping patch boundaries lack translation invariance: shifting an input series by a single timestep completely changes token representations.", body_style))

    # 2. Related Work
    story.append(Paragraph("2. Related Work & Literature Gap", h1_style))
    story.append(Paragraph("Time series representation learning has evolved across three dominant paradigms: (1) Pointwise Serial Models (LSTMs, N-BEATS), (2) Fixed Uniform Patching (PatchTST, ICLR 2023), and (3) Symbolic Discretization & BPE (BORF, IEEE 2024; TS-BPE, 2025). The critical research gap is that no existing framework unifies multi-scale dilated receptive field extraction, dense translation-invariant overlapping tokenization (s=1), 1D-SAX human-legible codebooks, and 2D scale-time spatial tensor representations within a non-stationary financial paradigm.", body_style))

    # 3. Methodology
    story.append(Paragraph("3. Proposed Methodology: MSOPT Framework", h1_style))
    story.append(Paragraph("MSOPT reformulates time series representation through a 4-stage pipeline:", body_style))
    story.append(Paragraph("• <b>Multi-Scale Receptive Fields</b>: Dense sampling across window sizes w in {4,8,16,32} and dilations d in {1,2,4} with dense stride s=1 for 100% translation invariance.", body_style))
    story.append(Paragraph("• <b>1D-SAX Discretization</b>: Subseries are quantized into discrete mean (alpha_mu) and slope (alpha_beta) symbols, creating a robust noise-filtering vocabulary.", body_style))
    story.append(Paragraph("• <b>2D Scale-Time Spatial Grid</b>: Tokens are indexed into a 2D matrix H in Z^{N_scales x T}.", body_style))
    story.append(Paragraph("• <b>PyTorch Conv-Transformer</b>: 2D spatial convolution Inception blocks capture inter-scale pattern composition before Multi-Head Self-Attention.", body_style))

    # 4. Results Table
    story.append(Paragraph("4. 10-Year Out-of-Sample Walk-Forward Results (2016–2026)", h1_style))
    
    t_data = [
        [Paragraph("<b>Asset</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white)),
         Paragraph("<b>Model</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white)),
         Paragraph("<b>Sharpe Ratio</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white)),
         Paragraph("<b>Sortino Ratio</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white)),
         Paragraph("<b>Max Drawdown</b>", ParagraphStyle('TH', fontName='Helvetica-Bold', fontSize=8, textColor=colors.white))],
        [Paragraph("SPY", body_style), Paragraph("Baseline (Fixed Windows)", body_style), Paragraph("0.5065", body_style), Paragraph("0.5979", body_style), Paragraph("-41.91%", body_style)],
        [Paragraph("SPY", body_style), Paragraph("<b>MSOPT Tokens (Ours)</b>", body_style), Paragraph("<b>0.8668</b>", body_style), Paragraph("<b>1.0847</b>", body_style), Paragraph("<b>-22.09%</b>", body_style)],
        [Paragraph("QQQ", body_style), Paragraph("Baseline (Fixed Windows)", body_style), Paragraph("0.5731", body_style), Paragraph("0.7049", body_style), Paragraph("-46.82%", body_style)],
        [Paragraph("QQQ", body_style), Paragraph("<b>MSOPT Tokens (Ours)</b>", body_style), Paragraph("<b>0.8709</b>", body_style), Paragraph("<b>1.0536</b>", body_style), Paragraph("<b>-41.57%</b>", body_style)],
        [Paragraph("AAPL", body_style), Paragraph("Baseline (Fixed Windows)", body_style), Paragraph("0.0554", body_style), Paragraph("0.0678", body_style), Paragraph("-72.41%", body_style)],
        [Paragraph("AAPL", body_style), Paragraph("<b>MSOPT Tokens (Ours)</b>", body_style), Paragraph("<b>0.3745</b>", body_style), Paragraph("<b>0.4999</b>", body_style), Paragraph("<b>-38.87%</b>", body_style)],
        [Paragraph("TLT", body_style), Paragraph("Baseline (Fixed Windows)", body_style), Paragraph("-2.1339", body_style), Paragraph("-2.8557", body_style), Paragraph("-95.09%", body_style)],
        [Paragraph("TLT", body_style), Paragraph("<b>MSOPT Tokens (Ours)</b>", body_style), Paragraph("<b>-0.0675</b>", body_style), Paragraph("<b>-0.0995</b>", body_style), Paragraph("<b>-29.43%</b>", body_style)],
    ]
    res_table = Table(t_data, colWidths=[60, 166, 100, 100, 100])
    res_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 10))

    # 5. Conclusion
    story.append(Paragraph("5. Conclusion & Future Directions", h1_style))
    story.append(Paragraph("MSOPT establishes a new paradigm in time series pattern recognition. By combining multi-scale 1D-SAX tokenization with translation-invariant overlapping stride (s=1) and 2D spatial Conv-Transformers, MSOPT delivers substantial Sharpe ratio gains (+71.1% on SPY) and robust regime crash protection post 5 bps transaction costs.", body_style))

    doc.build(story, canvasmaker=AcademicNumberedCanvas)
    print(f"Successfully generated Conference Paper PDF: {PDF_PATH}")

if __name__ == "__main__":
    build_conference_pdf()
