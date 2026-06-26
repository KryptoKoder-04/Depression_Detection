import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from backend.config import REPORTS_DIR

def generate_pdf_report(patient_id, session_date, filename, prediction, confidence, prob_depressed, duration):
    """
    Generates a premium clinical report in PDF format for depression screening.
    Saves it to REPORTS_DIR and returns the absolute file path.
    """
    report_filename = f"report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, report_filename)
    
    # 1. Setup Document Layout
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom harmonious styles (Navy blue, Slate gray, and Slate blue colors)
    PRIMARY_COLOR = colors.HexColor("#1e293b")   # Slate 800
    SECONDARY_COLOR = colors.HexColor("#0f172a") # Slate 900
    ACCENT_COLOR = colors.HexColor("#3b82f6")    # Blue 500
    LIGHT_GRAY = colors.HexColor("#f8fafc")      # Slate 50
    TEXT_COLOR = colors.HexColor("#334155")      # Slate 700
    
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=PRIMARY_COLOR,
        spaceAfter=15
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=PRIMARY_COLOR,
        spaceBefore=15,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=15,
        textColor=TEXT_COLOR
    )
    
    body_bold_style = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        textColor=colors.HexColor("#64748b")
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#94a3b8")
    )
    
    # 2. Header Title Section
    story.append(Paragraph("🎓 BTP Depression Detection System", title_style))
    story.append(Paragraph("AI-Assisted Behavioral Screening Report", ParagraphStyle('Sub', parent=body_style, fontSize=12, textColor=ACCENT_COLOR)))
    story.append(Spacer(1, 15))
    
    # 3. Patient Metadata Table
    meta_data = [
        [Paragraph("Patient ID:", body_bold_style), Paragraph(str(patient_id), body_style),
         Paragraph("Date of Session:", body_bold_style), Paragraph(str(session_date), body_style)],
        [Paragraph("Video Filename:", body_bold_style), Paragraph(str(filename), body_style),
         Paragraph("Processing Duration:", body_bold_style), Paragraph(f"{duration:.1f} seconds", body_style)],
        [Paragraph("Report Generated:", body_bold_style), Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), body_style),
         Paragraph("Analysis Pipeline:", body_bold_style), Paragraph("TSFFM-BiLSTM (MLP projection)", body_style)]
    ]
    
    meta_table = Table(meta_data, colWidths=[110, 150, 110, 150])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    
    story.append(meta_table)
    story.append(Spacer(1, 20))
    
    # 4. Screening Predictions Section
    story.append(Paragraph("📊 Screening Analysis Results", section_title_style))
    
    # Set prediction result color
    pred_display = prediction.upper().replace('_', ' ')
    pred_color = "#ef4444" if prediction == "depressed" else "#22c55e" # Red for depressed, Green for not depressed
    
    pred_style = ParagraphStyle(
        'PredValue',
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor(pred_color)
    )
    
    results_data = [
        [Paragraph("Model Classification:", body_bold_style), Paragraph(pred_display, pred_style)],
        [Paragraph("Inference Confidence:", body_bold_style), Paragraph(f"{confidence * 100:.2f}% probability of class", body_style)],
        [Paragraph("Raw Class Probability:", body_bold_style), Paragraph(f"Not Depressed: {(1 - prob_depressed) * 100:.1f}% | Depressed: {prob_depressed * 100:.1f}%", body_style)]
    ]
    
    results_table = Table(results_data, colWidths=[150, 370])
    results_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 25))
    
    # 5. Technical Details Section
    story.append(Paragraph("🧠 Technical Processing Details", section_title_style))
    tech_text = (
        "The system processes the video clip by downsampling to 5 frames per second and extracting facial and body coordinates via MediaPipe. "
        "The Face Stream processes 68 landmarks mapped to a 128-dimensional latent space. The Body Stream processes joint movements mapped to a 32-dimensional latent space. "
        "The streams are combined (160-dim feature vector) and processed through a Bidirectional LSTM (128 hidden units) using temporal average-pooling for final binary classification."
    )
    story.append(Paragraph(tech_text, body_style))
    story.append(Spacer(1, 20))
    
    # 6. Disclaimer Section (Boxed warning)
    story.append(Paragraph("⚠️ Medical Disclaimer", ParagraphStyle('DisTitle', parent=section_title_style, fontSize=11, textColor=colors.HexColor("#94a3b8"))))
    disclaimer_text = (
        "<b>Important Notice:</b> This automated analysis report is generated by an artificial intelligence screening prototype. "
        "It is <b>NOT</b> a clinical diagnostic tool and does not constitute a medical diagnosis. "
        "Results should only be used as a preliminary screening reference. Always consult a qualified, licensed mental health professional for formal psychiatric evaluation, screening, and treatment."
    )
    
    disclaimer_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[520])
    disclaimer_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BORDER', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(disclaimer_table)
    
    # 7. Build Document
    doc.build(story)
    
    return pdf_path, report_filename
