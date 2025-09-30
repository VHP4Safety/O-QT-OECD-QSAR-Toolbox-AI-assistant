# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

"""
Utility for generating comprehensive PDF reports using ReportLab.
UPDATED: Includes Markdown table parsing, list rendering, and improved structuring.
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
import re # Added for Markdown parsing
from xml.sax.saxutils import escape # Added for robust text handling

logger = logging.getLogger(__name__)

# Define styles globally
styles = None
styleN = None
styleH1 = None
styleH2 = None
styleH3 = None
styleError = None
metadataStyle = None
reportContentStyle = None
tableBodyStyle = None
tableHeaderStyle = None
listItemStyle = None # NEW: Style for list items

def initialize_styles():
    """Initializes ReportLab styles globally."""
    global styles, styleN, styleH1, styleH2, styleH3, styleError, metadataStyle, reportContentStyle, tableBodyStyle, tableHeaderStyle, listItemStyle
    try:
        styles = getSampleStyleSheet()
        styleN = styles['Normal']
        styleH1 = styles['Heading1']
        styleH2 = styles['Heading2']
        styleH3 = styles['Heading3']
        # Define a specific style for errors
        styleError = styles.get('Error', ParagraphStyle('Error', parent=styleN, textColor=colors.red))

        # Custom style for metadata table
        metadataStyle = ParagraphStyle(
            'metadata',
            parent=styleN,
            fontSize=10,
            leading=14,
        )

        # Custom style for the main report content
        reportContentStyle = ParagraphStyle(
            'reportContent',
            parent=styleN,
            fontSize=10,
            leading=14,
        )

        # NEW: Custom style for table content
        tableBodyStyle = ParagraphStyle(
            'tableBody',
            parent=styleN,
            fontSize=9,
            leading=12,
        )

        # NEW: Custom style for table headers
        tableHeaderStyle = ParagraphStyle(
            'tableHeader',
            parent=styleN,
            fontSize=9,
            leading=12,
            fontName='Helvetica-Bold',
        )

        # NEW: Custom style for list items (indented)
        listItemStyle = ParagraphStyle(
            'listItem',
            parent=reportContentStyle,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=5,
        )

    except Exception as e:
        logger.error(f"Error initializing ReportLab styles: {e}")
        # styles will remain None if initialization fails

# Initialize styles upon module import
initialize_styles()


# --- HELPER FUNCTIONS FOR TEXT PROCESSING ---

def format_and_escape_text(text):
    """Escapes XML entities and applies basic Markdown formatting for ReportLab."""
    # 1. Escape XML entities (e.g., <, >, &)
    # This is crucial as ReportLab Paragraphs interpret content as XML.
    escaped_text = escape(str(text))

    # 2. Handle markdown bold/italic (simple replacement to ReportLab tags)
    # Bold: **text** or __text__ -> <b>text</b>
    # Use non-greedy matching (.*?)
    escaped_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped_text)
    escaped_text = re.sub(r'__(.*?)__', r'<b>\1</b>', escaped_text)

    # Italic: *text* or _text_ -> <i>text</i>
    # Use lookarounds to handle single asterisks correctly without interfering with bold
    # This regex ensures we only match single asterisks that are not part of a double asterisk
    escaped_text = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', escaped_text)
    # Simple regex for underscore italics (less ambiguous than asterisks)
    escaped_text = re.sub(r'_(.*?)_', r'<i>\1</i>', escaped_text)

    # Handle line breaks within the text (e.g. for table cells or preformatted text)
    # Replace newlines with ReportLab's break tag.
    escaped_text = escaped_text.replace('\n', '<br/>')

    return escaped_text

# --- HELPER FUNCTIONS FOR TABLE PARSING AND CREATION ---

def parse_markdown_table(markdown_text):
    """
    Attempt to parse a block of text as a GFM-style Markdown table.
    """
    lines = markdown_text.strip().split('\n')
    if len(lines) < 2:
        return None

    # Normalize lines and filter for lines containing pipes
    normalized_lines = []
    for line in lines:
        line = line.strip()
        if not line or '|' not in line:
            continue
        # Ensure outer pipes for consistent splitting
        if not line.startswith('|'): line = f"| {line}"
        if not line.endswith('|'): line = f"{line} |"
        normalized_lines.append(line)

    if len(normalized_lines) < 2:
        return None

    # Check for GFM separator line (usually the second line)
    separator_line = normalized_lines[1]
    # Regex for GFM separator: allows optional pipes at start/end, and colons for alignment.
    # We check the structure after removing spaces for robustness against formatting variations.
    is_separator = re.match(r'^\|?(:?-+:?\|)+:?-+:?\|?$', separator_line.replace(' ', ''))

    header_line = normalized_lines[0]
    # Split by pipe and take elements between the first and last pipe
    headers = [h.strip() for h in header_line.split('|')][1:-1]
    if not headers or all(not h for h in headers):
        return None
    num_cols = len(headers)

    data = [headers]
    start_index = 1

    # If a valid separator exists, start data rows from the line after it.
    if is_separator:
        start_index = 2
    # We proceed even without a strict separator if the structure looks like a table, common in LLM outputs.

    # Parse data rows
    for line in normalized_lines[start_index:]:
        # Skip subsequent separator-like lines if they appear
        if re.match(r'^\|?(:?-+:?\|)+:?-+:?\|?$', line.replace(' ', '')):
             continue

        row = [r.strip() for r in line.split('|')][1:-1]

        if len(row) == num_cols:
            data.append(row)
        elif len(row) > 0:
            # Handle mismatched column counts by padding or truncating (robustness)
            if len(row) < num_cols:
                row.extend([''] * (num_cols - len(row)))
            else:
                row = row[:num_cols]
            data.append(row)

    # If we only have headers (and maybe a separator), it's not really a useful table.
    if len(data) < 2:
        return None

    return data

def create_rl_table(data, doc_width):
    """Creates a styled ReportLab Table from parsed data."""
    if not data:
        return None

    # Convert data entries into Paragraphs for wrapping and styling
    wrapped_data = []

    # Process Headers (using header style)
    header_row = [Paragraph(format_and_escape_text(cell), tableHeaderStyle) for cell in data[0]]
    wrapped_data.append(header_row)

    # Process Body (using body style)
    for row in data[1:]:
        wrapped_row = [Paragraph(format_and_escape_text(cell), tableBodyStyle) for cell in row]
        wrapped_data.append(wrapped_row)

    # Calculate column widths
    num_cols = len(data[0])
    # Ensure num_cols is not zero
    if num_cols > 0:
        # Simple distribution: equal widths.
        col_widths = [doc_width / num_cols] * num_cols
    else:
        return None # Cannot create table with zero columns

    # Create the table
    t = Table(wrapped_data, colWidths=col_widths)

    # Style the table
    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#444444")), # Dark grey header
        ('TEXTCOLOR',(0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        # FONTNAME is handled by the ParagraphStyle (tableHeaderStyle)
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F9F9F9")), # Light grey body
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BOX', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'), # Align content to top
    ])

    # Alternate row colors
    for i in range(1, len(wrapped_data)):
        if i % 2 == 0:
            style.add('BACKGROUND', (0, i), (-1, i), colors.HexColor("#EFEFEF"))

    t.setStyle(style)
    return t


def process_report_content(report_content, Story, doc_width):
    """Processes report content, detecting tables, lists, and formatting paragraphs."""

    # Split the report into blocks (paragraphs, tables, or lists) based on double newlines
    for block in report_content.split('\n\n'):
        text = block.strip()
        if not text:
            continue

        # 1. Try to parse as Markdown table
        table_data = parse_markdown_table(text)
        if table_data:
            try:
                rl_table = create_rl_table(table_data, doc_width)
                if rl_table:
                    Story.append(rl_table)
                    Story.append(Spacer(1, 0.15*inch))
                    continue # Successfully processed as table
            except Exception as e:
                logger.warning(f"Error creating ReportLab table from Markdown block: {e}. Falling back to text.")
                # Fallback to rendering as text if table creation fails

        # 2. If not a table, determine style based on headings
        if text.startswith("#### "):
            style = styles.get('Heading4', styleH3) # Use H4 if available, else H3
            content = text[5:].strip()
        elif text.startswith("### "):
            style = styleH3
            content = text[4:].strip()
        elif text.startswith("## "):
            style = styleH2
            content = text[3:].strip()
        elif text.startswith("# "):
            style = styleH2 # Use H2 for main sections within the report
            content = text[2:].strip()
        else:
            style = reportContentStyle
            content = text

        # 3. Handle Lists (Check if the block contains list items)
        is_list_block = False
        for line in content.split('\n'):
            line = line.strip()
            # Check for bulleted or numbered lists
            if line.startswith('- ') or line.startswith('* ') or re.match(r'^\d+\.\s+', line):
                is_list_block = True
                break

        if is_list_block:
             list_lines = content.split('\n')
             for line in list_lines:
                 line = line.strip()
                 if not line: continue

                 bullet_text = None
                 item_text = line

                 # Determine bullet type and extract text
                 if line.startswith('- ') or line.startswith('* '):
                     bullet_text = '•' # Standard bullet
                     item_text = line[2:].strip()
                 else:
                     # Check for numbered list
                     match = re.match(r'^(\d+)\.\s+(.*)', line)
                     if match:
                         bullet_text = f"{match.group(1)}."
                         item_text = match.group(2).strip()

                 # Render the item
                 try:
                    # Apply formatting/escaping to the list item text
                    formatted_item_text = format_and_escape_text(item_text)

                    if bullet_text:
                        # Use the dedicated listItemStyle with indentation
                        p = Paragraph(formatted_item_text, listItemStyle, bulletText=bullet_text)
                    else:
                        # Fallback for lines within a list block that don't match list syntax
                        p = Paragraph(formatted_item_text, style)

                    Story.append(p)
                 except Exception as e:
                     handle_paragraph_error(e, item_text, style, Story)

             Story.append(Spacer(1, 0.1*inch))
             continue # Move to the next block


        # 4. Handle as standard Paragraph
        try:
            formatted_content = format_and_escape_text(content)

            p = Paragraph(formatted_content, style)
            Story.append(p)
            Story.append(Spacer(1, 0.1*inch))

        except Exception as e:
            handle_paragraph_error(e, content, style, Story)


def handle_paragraph_error(e, content, style, Story):
    """Centralized error handling for paragraph creation."""
    # Handle reportlab errors during paragraph creation (e.g., invalid XML structure after substitution)
    logger.warning(f"Error rendering content block: {e}. Block content: {str(content)[:200]}...")
    Story.append(Paragraph(f"[Error rendering content block: {e}]", styleError))

    # Try adding sanitized text (fallback without markdown formatting)
    try:
        # Basic sanitation: escape XML entities only
        sanitized_text = escape(content.encode('utf-8', errors='replace').decode('utf-8'))
        Story.append(Paragraph(sanitized_text, style))
    except Exception:
        Story.append(Paragraph("[Sanitized content also failed rendering]", styleError))

    Story.append(Spacer(1, 0.1*inch))

# --- MAIN PDF GENERATION FUNCTION ---

def generate_pdf_report(log_data: dict) -> io.BytesIO:
    """Generates a comprehensive PDF report from the analysis log data."""

    if styles is None:
        initialize_styles() # Attempt re-initialization if failed on import
        if styles is None:
            raise RuntimeError("ReportLab styles could not be initialized.")

    buffer = io.BytesIO()
    # Standard margins (72 points = 1 inch)
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)

    # Calculate available width for tables
    page_width, _ = A4
    doc_width = page_width - doc.leftMargin - doc.rightMargin

    Story = []

    # --- Title ---
    title = f"O'QT Assistant - Comprehensive Analysis Report"
    Story.append(Paragraph(title, styleH1))
    Story.append(Spacer(1, 0.2*inch))

    # --- 1. Metadata Section (Kept as is, user likes it) ---
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

    # Create the metadata table
    # We need to convert data entries into Paragraphs for wrapping
    wrapped_data = []
    for row in data:
        # Ensure inputs to Paragraph are strings and escaped (minimal escaping for metadata)
        # We use escape() here just for safety, though metadata is usually clean.
        wrapped_row = [Paragraph(escape(str(row[0])), metadataStyle), Paragraph(escape(str(row[1])), metadataStyle)]
        wrapped_data.append(wrapped_row)

    # Define column widths for metadata table
    meta_col_widths = [1.5*inch, doc_width - 1.5*inch]

    t = Table(wrapped_data, colWidths=meta_col_widths)

    # Style the metadata table (Original style requested by the user)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))

    Story.append(t)

    # --- 2. Synthesized Report Section (UPDATED) ---
    Story.append(PageBreak())
    Story.append(Paragraph("2. Synthesized Analysis Report", styleH2))
    Story.append(Spacer(1, 0.2*inch))

    synthesized_report = log_data.get("analysis", {}).get("synthesized_report", "Report content not available.")

    # Process report content (handling tables, lists, and paragraphs)
    process_report_content(synthesized_report, Story, doc_width)


    # --- 3. Specialist Agent Reports Section (UPDATED) ---
    Story.append(PageBreak())
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

        # Process specialist report content
        process_report_content(report_content, Story, doc_width)

        # Add a larger spacer between specialist reports, and a page break if not the last one
        Story.append(Spacer(1, 0.3*inch))
        if i < len(report_order) - 1:
             Story.append(PageBreak())


    # Build the PDF
    try:
        doc.build(Story)
    except Exception as e:
        logger.error(f"Failed to build PDF document: {e}", exc_info=True)
        # Attempt to build an error-only PDF if the main build fails
        buffer.seek(0)
        buffer.truncate()
        # Re-initialize styles if needed for error doc
        if styles is None:
             initialize_styles()

        if styles:
            error_doc = SimpleDocTemplate(buffer, pagesize=A4)
            # Use available styles for the error message
            error_h1 = styleH1 if styleH1 else styles['Heading1']
            error_n = styleN if styleN else styles['Normal']

            error_story = [Paragraph("Error Generating PDF Report", error_h1),
                        Paragraph(f"An unrecoverable error occurred during PDF generation. This may be due to complex formatting or invalid characters in the report content. Error details: {e}", error_n)]
            try:
                error_doc.build(error_story)
            except Exception:
                # If even the error PDF fails, raise the original exception
                raise RuntimeError(f"Failed to generate PDF report and error fallback failed: {e}")
        else:
            raise RuntimeError(f"Failed to generate PDF report and styles are unavailable: {e}")

    buffer.seek(0)
    return buffer