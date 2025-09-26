# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

"""
Utility for generating comprehensive PDF reports using ReportLab.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
import textwrap
import logging

logger = logging.getLogger(__name__)

# Define styles
try:
    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    styleH1 = styles['Heading1']
    styleH2 = styles['Heading2']
    styleH3 = styles['Heading3']
    # Define a specific style for errors if available, otherwise use Normal
    styleError = styles.get('Error', styleN)

    # Custom style for metadata table
    metadataStyle = ParagraphStyle(
        'metadata',
        parent=styleN,
        fontSize=10,
        leading=14,
    )

    # Custom style for the main report content (handling preformatted text better)
    reportContentStyle = ParagraphStyle(
        'reportContent',
        parent=styleN,
        fontSize=10,
        leading=14,
        # Using a monospaced font might help if the reports contain ASCII tables or aligned text
        # fontName='Courier', 
    )

except Exception as e:
    logger.error(f"Error initializing ReportLab styles: {e}")
    # Handle potential initialization issues gracefully if styles cannot be loaded
    styles = None


def generate_pdf_report(log_data: dict) -> io.BytesIO:
    """Generates a comprehensive PDF report from the analysis log data."""
    
    if styles is None:
        raise RuntimeError("ReportLab styles could not be initialized.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    Story = []
    
    # --- Title ---
    title = f"O'QT Assistant - Comprehensive Analysis Report"
    Story.append(Paragraph(title, styleH1))
    Story.append(Spacer(1, 0.2*inch))

    # --- Metadata Section ---
    Story.append(Paragraph("1. Analysis Metadata", styleH2))
    Story.append(Spacer(1, 0.1*inch))

    metadata = log_data.get("metadata", {})
    configuration = log_data.get("configuration", {})
    inputs = log_data.get("inputs", {})
    
    # Prepare data for the metadata table
    data = [
        ['Category', 'Detail'],
        ['Tool Name', metadata.get("tool_name", "N/A")],
        ['Tool Version', metadata.get("version", "N/A")],
        ['Timestamp', metadata.get("timestamp", "N/A")],
        ['Target Chemical', inputs.get("identifier", "N/A")],
        ['Search Type', inputs.get("search_type", "N/A")],
        ['Analysis Context', inputs.get("context", "N/A")]
    ]
    
    # LLM Configuration
    llm_config = configuration.get("llm_configuration", {})
    data.append(['LLM Provider', llm_config.get("provider", "N/A")])
    data.append(['LLM Model (Display Name)', llm_config.get("model_name", "N/A")])
    data.append(['LLM Model (ID)', llm_config.get("model_id", "N/A")])
    
    # QSAR Configuration
    qsar_config = configuration.get("qsar_toolbox_configuration", {})
    data.append(['QSAR Toolbox API URL', qsar_config.get("api_url", "N/A")])

    # Metabolism Simulators
    simulator_guids = inputs.get("simulator_guids", [])
    if simulator_guids:
        data.append(['Metabolism Simulators (GUIDs)', ", ".join(simulator_guids)])
    else:
        data.append(['Metabolism Simulators', "None selected (Skipped)"])

    # Create the table
    # Wrap text in the 'Detail' column if it's too long
    col_widths = [1.5*inch, 5.0*inch]
    
    # We need to convert data entries into Paragraphs to allow text wrapping within cells
    wrapped_data = []
    for row in data:
        # Ensure inputs to Paragraph are strings
        wrapped_row = [Paragraph(str(row[0]), metadataStyle), Paragraph(str(row[1]), metadataStyle)]
        wrapped_data.append(wrapped_row)

    t = Table(wrapped_data, colWidths=col_widths)
    
    # Style the table
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    
    Story.append(t)
    Story.append(PageBreak())

    # --- Synthesized Report Section ---
    Story.append(Paragraph("2. Synthesized Analysis Report", styleH2))
    Story.append(Spacer(1, 0.2*inch))
    
    synthesized_report = log_data.get("analysis", {}).get("synthesized_report", "Report content not available.")
    
    # Handle potential formatting issues by splitting the report into paragraphs
    # This handles Markdown-like formatting slightly better than one large block
    for part in synthesized_report.split('\n\n'):
        if part.strip():
            # Basic handling of headings (very rudimentary Markdown interpretation)
            if part.startswith("### "):
                style = styleH3
                text = part[4:].strip()
            elif part.startswith("## "):
                style = styleH2
                text = part[3:].strip()
            elif part.startswith("# "):
                 # Avoid clash with main title, use H2
                style = styleH2
                text = part[2:].strip()
            else:
                style = reportContentStyle
                text = part.strip()
            
            # Replace potential problematic characters if necessary (e.g., non-UTF8)
            try:
                p = Paragraph(text, style)
                Story.append(p)
                Story.append(Spacer(1, 0.1*inch))
            except Exception as e:
                # Handle reportlab errors during paragraph creation (e.g. bad characters or XML entities)
                logger.warning(f"Error rendering synthesized report block: {e}")
                p = Paragraph(f"[Error rendering content block: {e}]", styleError)
                Story.append(p)
                # Try adding sanitized text (escaping potential XML/HTML tags)
                from reportlab.platypus.paragraph import cleanBlock
                sanitized_text = cleanBlock(text.encode('utf-8', errors='replace').decode('utf-8'))
                try:
                    p = Paragraph(sanitized_text, style)
                    Story.append(p)
                except Exception:
                    p = Paragraph("[Sanitized content also failed rendering]", styleError)
                    Story.append(p)
                Story.append(Spacer(1, 0.1*inch))


    Story.append(PageBreak())

    # --- Specialist Agent Reports Section ---
    Story.append(Paragraph("3. Specialist Agent Reports", styleH2))
    Story.append(Spacer(1, 0.2*inch))

    specialist_reports = log_data.get("analysis", {}).get("specialist_agent_outputs", {})
    
    # Define the order of reports
    report_order = [
        "Chemical_Context",
        "Physical_Properties",
        "Environmental_Fate",
        "Profiling_Reactivity",
        "Experimental_Data",
        "Metabolism",
        "Read_Across"
    ]

    for i, key in enumerate(report_order):
        report_content = specialist_reports.get(key, "Content not available.")
        title = f"3.{i+1} {key.replace('_', ' ')}"
        
        Story.append(Paragraph(title, styleH3))
        Story.append(Spacer(1, 0.1*inch))
        
        # Split specialist reports into paragraphs as well
        for part in report_content.split('\n\n'):
            if part.strip():
                try:
                    p = Paragraph(part.strip(), reportContentStyle)
                    Story.append(p)
                    Story.append(Spacer(1, 0.1*inch))
                except Exception as e:
                     # Handle reportlab errors during paragraph creation
                    logger.warning(f"Error rendering specialist report block ({key}): {e}")
                    p = Paragraph(f"[Error rendering content block: {e}]", styleError)
                    Story.append(p)
                    # Try adding sanitized text
                    from reportlab.platypus.paragraph import cleanBlock
                    sanitized_text = cleanBlock(part.strip().encode('utf-8', errors='replace').decode('utf-8'))
                    try:
                        p = Paragraph(sanitized_text, reportContentStyle)
                        Story.append(p)
                    except Exception:
                        p = Paragraph("[Sanitized content also failed rendering]", styleError)
                        Story.append(p)
                    Story.append(Spacer(1, 0.1*inch))

        # Add a larger spacer between specialist reports
        Story.append(Spacer(1, 0.3*inch))

    # Build the PDF
    try:
        doc.build(Story)
    except Exception as e:
        logger.error(f"Failed to build PDF document: {e}", exc_info=True)
        # Attempt to build an error-only PDF if the main build fails
        buffer.seek(0)
        buffer.truncate()
        error_doc = SimpleDocTemplate(buffer, pagesize=A4)
        error_story = [Paragraph("Error Generating PDF Report", styleH1),
                       Paragraph(f"An unrecoverable error occurred during PDF generation: {e}", styleN)]
        try:
            error_doc.build(error_story)
        except Exception:
            # If even the error PDF fails, raise the original exception
            raise RuntimeError(f"Failed to generate PDF report: {e}")

    buffer.seek(0)
    return buffer