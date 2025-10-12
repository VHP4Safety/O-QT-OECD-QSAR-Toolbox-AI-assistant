# src/oqt_assistant/utils/pdf_generator.py

# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""
Utility for generating summary PDF reports using ReportLab (Open Source Edition).
This version provides a structured layout with metadata, provenance highlights, and key AI summaries.
Advanced formatting (e.g., complex comparative tables, full specialist appendices) is reserved for O-QT Pro.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import Counter
from pathlib import Path
import logging
import re
from xml.sax.saxutils import escape
from urllib.parse import urlsplit
import os

logger = logging.getLogger(__name__)
LOGO_PATH = Path(__file__).resolve().parents[3] / "o-qt_logo.jpg"

_REPORTLAB_CACHE: Dict[str, Any] | None = None


def _ensure_reportlab() -> Dict[str, Any]:
    """Lazy-import ReportLab components to avoid hard dependency at import time."""
    global _REPORTLAB_CACHE
    if _REPORTLAB_CACHE is not None:
        return _REPORTLAB_CACHE

    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
        from reportlab.platypus import (  # type: ignore
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            PageBreak,
            Table,
            TableStyle,
            Preformatted,
            Image as RLImage,
        )
        from reportlab.lib.units import inch  # type: ignore
        from reportlab.lib import colors  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - triggered in environments without reportlab
        raise RuntimeError(
            "PDF generation requires the 'reportlab' package. Install it via `pip install reportlab`."
        ) from exc

    _REPORTLAB_CACHE = {
        "A4": A4,
        "getSampleStyleSheet": getSampleStyleSheet,
        "ParagraphStyle": ParagraphStyle,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Paragraph": Paragraph,
        "Spacer": Spacer,
        "PageBreak": PageBreak,
        "Table": Table,
        "TableStyle": TableStyle,
        "Preformatted": Preformatted,
        "RLImage": RLImage,
        "inch": inch,
        "colors": colors,
    }
    return _REPORTLAB_CACHE


def initialize_styles():
    """Initializes and customizes ReportLab styles."""
    try:
        rl = _ensure_reportlab()
        getSampleStyleSheet = rl["getSampleStyleSheet"]
        ParagraphStyle = rl["ParagraphStyle"]
        colors = rl["colors"]
        styles = getSampleStyleSheet()
        
        # Customize existing styles for a cleaner look using Helvetica
        styles['Title'].fontSize = 20
        styles['Title'].spaceAfter = 20
        styles['Title'].fontName = 'Helvetica-Bold'
        
        styles['Heading1'].fontSize = 16
        styles['Heading1'].spaceBefore = 18
        styles['Heading1'].spaceAfter = 8
        styles['Heading1'].fontName = 'Helvetica-Bold'
        styles['Heading1'].textColor = colors.darkblue
        
        styles['Heading2'].fontSize = 14
        styles['Heading2'].spaceBefore = 14
        styles['Heading2'].spaceAfter = 6
        styles['Heading2'].fontName = 'Helvetica-Bold'
        styles['Heading2'].textColor = colors.darkblue

        styles['Heading3'].fontSize = 12
        styles['Heading3'].spaceBefore = 12
        styles['Heading3'].spaceAfter = 4
        styles['Heading3'].fontName = 'Helvetica-Bold'
        styles['Heading3'].textColor = colors.black
        
        styles['Normal'].fontSize = 10
        styles['Normal'].leading = 14
        styles['Normal'].fontName = 'Helvetica'
        
        # Add styles for metadata tables
        if 'MetaKey' not in styles:
            styles.add(ParagraphStyle(name='MetaKey', fontSize=10, leading=14, alignment=0, fontName='Helvetica-Bold'))
        if 'MetaValue' not in styles:
            styles.add(ParagraphStyle(name='MetaValue', fontSize=10, leading=14, alignment=0, fontName='Helvetica'))
        
        # Add a style for list items (inherits from Normal)
        if 'ListItem' not in styles:
            styles.add(ParagraphStyle(name='ListItem', parent=styles['Normal'], leftIndent=12, spaceBefore=2))

        # Add a basic style for preformatted text fallback (if needed)
        if 'Code' not in styles:
            styles.add(ParagraphStyle(name='Code', fontName='Courier', fontSize=9, leading=12))

        if 'TableHeader' not in styles:
            styles.add(ParagraphStyle(name='TableHeader', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, leading=14))
        if 'TableCell' not in styles:
            styles.add(ParagraphStyle(name='TableCell', parent=styles['Normal'], fontSize=10, leading=14))

        return styles
    except Exception as e:
        logger.error(f"Error initializing ReportLab styles: {e}")
        return None


# Initialize styles lazily
STYLES: Dict[str, Any] | None = None


def apply_inline_formatting(content: str) -> str:
    """Applies basic inline Markdown formatting (bold/italic) using ReportLab XML tags."""
    # Escape XML entities first
    content = escape(content)
    
    # Bold and Italic (***text*** or ___text___)
    # Use non-greedy matching (.*?)
    content = re.sub(r'(\*\*\*|___)(.*?)\1', r'<b><i>\2</i></b>', content)
    # Bold (**text** or __text__)
    content = re.sub(r'(\*\*|__)(.*?)\1', r'<b>\2</b>', content)
    
    # Italic (*text* or _text_)
    # Use lookarounds to handle edge cases and avoid matching parts of already processed tags
    # This regex aims to match *text* but not * text * or parts of **text**
    content = re.sub(r'(?<!\*)\*(?!\s)(.*?)(?<!\s)\*(?!\*)', r'<i>\1</i>', content)
    content = re.sub(r'(?<!_)_(?!\s)(.*?)(?<!\s)_(?!_)', r'<i>\1</i>', content)
    
    return content


def parse_markdown_to_flowables(markdown_text: str) -> List:
    """
    Basic parser to convert LLM Markdown output into ReportLab Flowables.
    Handles headings, paragraphs, lists, horizontal rules, and simple Markdown tables.
    """
    rl = _ensure_reportlab()
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Preformatted = rl["Preformatted"]
    inch = rl["inch"]

    flowables: List = []
    paragraph_lines: List[str] = []
    list_items: List[str] = []
    table_lines: List[str] = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            # Join lines with spaces to keep natural flow
            content = " ".join(paragraph_lines).strip()
            if content:
                flowables.append(Paragraph(apply_inline_formatting(content), STYLES['Normal']))
            paragraph_lines = []

    def flush_list():
        nonlocal list_items
        if list_items:
            for item in list_items:
                flowables.append(Paragraph(apply_inline_formatting(item), STYLES['ListItem'], bulletText='•'))
            list_items = []

    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        table_data = parse_markdown_table(table_lines)
        if table_data:
            flowables.append(build_table_flowable(table_data))
            flowables.append(Spacer(1, 0.12 * inch))
        else:
            # Fallback to preformatted text if parsing fails
            flowables.append(Preformatted("\n".join(table_lines), STYLES['Code']))
            flowables.append(Spacer(1, 0.12 * inch))
        table_lines = []

    lines = markdown_text.splitlines()
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        # Blank line separates blocks
        if not stripped:
            flush_paragraph()
            flush_list()
            flush_table()
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            flush_paragraph()
            flush_list()
            flush_table()
            flowables.append(Spacer(1, 0.12 * inch))
            continue

        heading_match = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if heading_match:
            flush_paragraph()
            flush_list()
            flush_table()
            hashes, content = heading_match.groups()
            level = len(hashes)
            formatted = apply_inline_formatting(content.strip())
            if level == 1:
                flowables.append(Paragraph(formatted, STYLES['Heading1']))
            elif level == 2:
                flowables.append(Paragraph(formatted, STYLES['Heading2']))
            else:
                flowables.append(Paragraph(formatted, STYLES['Heading3']))
            continue

        bullet_match = re.match(r'^(\s*[\-\*]|\s*\d+\.)\s+(.*)', line)
        if bullet_match:
            flush_paragraph()
            flush_table()
            content = bullet_match.group(2).strip()
            if content:
                list_items.append(content)
            continue

        if "|" in stripped and stripped.startswith("|"):
            flush_paragraph()
            flush_list()
            table_lines.append(stripped)
            continue
        elif table_lines:
            flush_table()

        # Default: accumulate paragraph text
        paragraph_lines.append(stripped)

    # Flush any remaining content
    flush_paragraph()
    flush_list()
    flush_table()

    return flowables


def extract_read_across_summary(report_text: str, max_sections: int = 5, max_characters: int = 4000) -> str:
    """
    Compresses the read-across specialist report to the first few informative blocks.
    Falls back to the full text when short while appending a note when truncated.
    """
    if not report_text:
        return ""

    blocks = [block.strip() for block in re.split(r"\n\s*\n", report_text) if block.strip()]
    if not blocks:
        return report_text

    summary_blocks: List[str] = []
    running_length = 0
    for block in blocks:
        summary_blocks.append(block)
        running_length += len(block)
        if len(summary_blocks) >= max_sections or running_length >= max_characters:
            break

    summary_text = "\n\n".join(summary_blocks).strip()
    if len(summary_text) < len(report_text):
        summary_text += "\n\n> _(Additional read-across rationale is available in the specialist downloads.)_"
    return summary_text


def parse_markdown_table(lines: List[str]) -> Optional[List[List[str]]]:
    """
    Parses a simple Markdown table from a list of lines.
    Returns a 2D list (including header) or None if parsing fails.
    """
    normalized: List[str] = []
    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            continue
        if "|" not in trimmed:
            return None
        if not trimmed.startswith("|"):
            trimmed = "|" + trimmed
        if not trimmed.endswith("|"):
            trimmed = trimmed + "|"
        normalized.append(trimmed)

    if len(normalized) < 2:
        return None

    header_parts = [cell.strip() for cell in normalized[0].split("|")[1:-1]]
    if not header_parts:
        return None

    data: List[List[str]] = [header_parts]
    separator_pattern = re.compile(r'^\|?(?:\s*:?-+:?\s*\|)+\s*$')
    start_index = 1
    if len(normalized) > 1 and separator_pattern.match(normalized[1]):
        start_index = 2

    for row_line in normalized[start_index:]:
        if separator_pattern.match(row_line):
            continue
        row_cells = [cell.strip() for cell in row_line.split("|")[1:-1]]
        if len(row_cells) < len(header_parts):
            row_cells.extend([""] * (len(header_parts) - len(row_cells)))
        elif len(row_cells) > len(header_parts):
            row_cells = row_cells[:len(header_parts)]
        data.append(row_cells)

    if len(data) <= 1:
        return None
    return data


def build_table_flowable(data: List[List[str]]) -> Table:
    """Converts parsed Markdown table data into a styled ReportLab Table."""
    rl = _ensure_reportlab()
    Paragraph = rl["Paragraph"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    colors = rl["colors"]

    rows: List[List[Paragraph]] = []
    for row_index, row in enumerate(data):
        paragraph_row: List[Paragraph] = []
        for cell in row:
            style_name = 'TableHeader' if row_index == 0 else 'TableCell'
            paragraph_row.append(Paragraph(apply_inline_formatting(cell), STYLES[style_name]))
        rows.append(paragraph_row)

    table = Table(rows, hAlign='LEFT')
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e6eef8")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0b1f33")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.HexColor("#b5c5d6")),
    ])
    table.setStyle(table_style)
    return table


def _sanitize_api_url_for_report(url: Optional[str], *, redact: bool) -> str:
    """Return a safe, placeholder-form URL for the QSAR API when redaction is enabled.

    If `redact` is False or the URL is empty, return the original or 'N/A'.
    When redacting, always present as '<scheme>://<host>:<port>/<api...>'.
    If the original path starts with '/api', preserve it; otherwise use '/api'.
    """
    if not url:
        return "N/A"
    if not redact:
        return str(url)

    try:
        parsed = urlsplit(str(url))
        scheme = parsed.scheme or "http"
        # Preserve '/api' path if present; otherwise default to '/api'
        path = parsed.path if (parsed.path and parsed.path.startswith("/api")) else "/api"
        return f"{scheme}://<host>:<port>{path}"
    except Exception:
        # Fallback to generic placeholder when parsing fails
        return "http://<host>:<port>/api"


def generate_pdf_report(comprehensive_log: Dict[str, Any]) -> bytes:
    """
    Generates a clean PDF report from the comprehensive analysis log using basic Markdown parsing.
    """
    rl = _ensure_reportlab()
    SimpleDocTemplate = rl["SimpleDocTemplate"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    PageBreak = rl["PageBreak"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    Preformatted = rl["Preformatted"]
    RLImage = rl["RLImage"]
    inch = rl["inch"]
    A4 = rl["A4"]
    colors = rl["colors"]

    global STYLES
    if STYLES is None:
        STYLES = initialize_styles()
    if not STYLES:
        raise RuntimeError("ReportLab styles could not be initialized.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=40,
    )

    Story = []

    # --- Extract Data ---
    metadata = comprehensive_log.get("metadata", {})
    inputs = comprehensive_log.get("inputs", {})
    configuration = comprehensive_log.get("configuration", {})
    synthesized_report = comprehensive_log.get("analysis", {}).get("synthesized_report", "Report content not available.")
    
    identifier = inputs.get("identifier", "Unknown Chemical")
    timestamp = metadata.get("timestamp", datetime.now().isoformat())
    version = metadata.get("version", "N/A")
    
    llm_config = configuration.get("llm_configuration", {})
    qsar_config = configuration.get("qsar_toolbox_configuration", {})

    # --- Title Page ---
    Story.append(Paragraph("O-QT Assistant Summary Report", STYLES['Title']))
    if LOGO_PATH.exists():
        try:
            logo = RLImage(str(LOGO_PATH))
            logo.hAlign = 'CENTER'
            logo._restrictSize(1.8 * inch, 1.8 * inch)
            Story.append(logo)
            Story.append(Spacer(1, 0.2 * inch))
        except Exception as logo_error:
            logger.warning(f"Unable to embed logo image: {logo_error}")
    # Use escape for user-provided identifier
    Story.append(Paragraph(f"Target Chemical: {escape(str(identifier))}", STYLES['Heading1']))
    Story.append(Paragraph("Origin: Generated by the O-QT Assistant (Open Source Edition) using OECD QSAR Toolbox data.", STYLES['Normal']))
    Story.append(Spacer(1, 0.3*inch))

    # --- Metadata/Configuration Section (Using Tables for clean layout) ---
    
    # Define a common table style
    meta_table_style = TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LINEBELOW', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ])
    # Calculate column widths (approx 6 inches total content width)
    col_widths = [1.8*inch, 4.2*inch]

    # 1. Run Details
    Story.append(Paragraph("Run Details", STYLES['Heading2']))
    
    run_data = [
        [Paragraph("Generated On:", STYLES['MetaKey']), Paragraph(timestamp, STYLES['MetaValue'])],
        [Paragraph("Tool Version:", STYLES['MetaKey']), Paragraph(version, STYLES['MetaValue'])],
    ]
    
    t_run = Table(run_data, colWidths=col_widths)
    t_run.setStyle(meta_table_style)
    Story.append(t_run)
    Story.append(Spacer(1, 0.2*inch))

    # Provenance legend for clarity
    Story.append(Paragraph("Provenance Legend", STYLES['Heading2']))
    legend_rows = [
        [Paragraph("Experimental (Toolbox)", STYLES['MetaKey']),
         Paragraph("Value retrieved directly from the user's OECD QSAR Toolbox instance via WebAPI.", STYLES['MetaValue'])],
        [Paragraph("QSAR Estimate (Toolbox)", STYLES['MetaKey']),
         Paragraph("Calculated or profiled result generated by the Toolbox during the same session.", STYLES['MetaValue'])],
        [Paragraph("LLM Narrative", STYLES['MetaKey']),
         Paragraph("Interpretation created by the configured LLM; never substitutes the raw values above.", STYLES['MetaValue'])],
    ]
    legend_table = Table(legend_rows, colWidths=col_widths)
    legend_table.setStyle(meta_table_style)
    Story.append(legend_table)
    Story.append(Spacer(1, 0.2 * inch))

    # 2. Inputs
    Story.append(Paragraph("Inputs and Context", STYLES['Heading2']))
    
    input_data = [
        [Paragraph("Identifier:", STYLES['MetaKey']), Paragraph(escape(str(identifier)), STYLES['MetaValue'])],
        [Paragraph("Search Type:", STYLES['MetaKey']), Paragraph(escape(str(inputs.get('search_type', 'N/A'))), STYLES['MetaValue'])],
        # Context needs careful handling as it might be long and requires escaping
        [Paragraph("Context:", STYLES['MetaKey']), Paragraph(escape(str(inputs.get('context', 'N/A'))), STYLES['MetaValue'])],
    ]
    
    t_input = Table(input_data, colWidths=col_widths)
    t_input.setStyle(meta_table_style)
    Story.append(t_input)
    Story.append(Spacer(1, 0.2*inch))

    # 3. Configuration
    Story.append(Paragraph("Configuration", STYLES['Heading2']))
    
    # Determine whether to redact API URLs in PDF outputs (for public examples)
    redact_urls = os.getenv("OQT_REDACT_PDF_URLS", "0").strip().lower() in {"1", "true", "yes", "on"}
    api_url_display = _sanitize_api_url_for_report(qsar_config.get('api_url'), redact=redact_urls)

    config_data = [
        [Paragraph("LLM Provider:", STYLES['MetaKey']), Paragraph(escape(str(llm_config.get('provider', 'N/A'))), STYLES['MetaValue'])],
        [Paragraph("LLM Model:", STYLES['MetaKey']), Paragraph(escape(str(llm_config.get('model_name', 'N/A'))), STYLES['MetaValue'])],
        [Paragraph("QSAR Toolbox API:", STYLES['MetaKey']), Paragraph(escape(api_url_display), STYLES['MetaValue'])],
    ]
    
    t_config = Table(config_data, colWidths=col_widths)
    t_config.setStyle(meta_table_style)
    Story.append(t_config)
    Story.append(Spacer(1, 0.2*inch))

    processed_data = comprehensive_log.get("data_retrieval", {}).get("processed_qsar_toolbox_data", {})
    chemical_basic = {}
    if isinstance(processed_data, dict):
        chemical_basic = (processed_data.get("chemical_data") or {}).get("basic_info", {}) if isinstance(processed_data.get("chemical_data"), dict) else {}

    # --- Chemical Highlights ---
    chemical_rows = []
    if chemical_basic:
        highlight_fields = [
            ("Primary Name:", "Name"),
            ("CAS:", "Cas"),
            ("SMILES:", "Smiles"),
            ("Substance Type:", "SubstanceType"),
        ]
        for label, key in highlight_fields:
            value = chemical_basic.get(key)
            if value:
                chemical_rows.append([Paragraph(label, STYLES['MetaKey']), Paragraph(escape(str(value)), STYLES['MetaValue'])])

    if chemical_rows:
        Story.append(Paragraph("Chemical Highlights", STYLES['Heading2']))
        table = Table(chemical_rows, colWidths=col_widths)
        table.setStyle(meta_table_style)
        Story.append(table)
        Story.append(Spacer(1, 0.2 * inch))

    # --- Provenance Snapshot ---
    experimental_data = processed_data.get("experimental_data", []) if isinstance(processed_data, dict) else []
    num_experimental = len(experimental_data) if isinstance(experimental_data, list) else 0

    profiling_results = processed_data.get("profiling", {}) if isinstance(processed_data, dict) else {}
    profiler_count = len(profiling_results.get("results", {}) or {})

    qsar_models_section = processed_data.get("qsar_models", {}) if isinstance(processed_data, dict) else {}
    qsar_processed = qsar_models_section.get("processed", {}) if isinstance(qsar_models_section, dict) else {}
    in_domain_models = qsar_processed.get("in_domain", []) if isinstance(qsar_processed, dict) else []
    out_domain_models = qsar_processed.get("out_of_domain", []) if isinstance(qsar_processed, dict) else []

    snapshot_rows = [
        [Paragraph("Experimental records retrieved", STYLES['MetaKey']), Paragraph(str(num_experimental), STYLES['MetaValue'])],
        [Paragraph("Profilers executed", STYLES['MetaKey']), Paragraph(str(profiler_count), STYLES['MetaValue'])],
        [Paragraph("QSAR predictions in domain", STYLES['MetaKey']), Paragraph(str(len(in_domain_models) if isinstance(in_domain_models, list) else 0), STYLES['MetaValue'])],
    ]
    if isinstance(out_domain_models, list) and out_domain_models:
        snapshot_rows.append([Paragraph("QSAR predictions out of domain", STYLES['MetaKey']), Paragraph(str(len(out_domain_models)), STYLES['MetaValue'])])

    Story.append(Paragraph("Data Provenance Snapshot", STYLES['Heading2']))
    snapshot_table = Table(snapshot_rows, colWidths=col_widths)
    snapshot_table.setStyle(meta_table_style)
    Story.append(snapshot_table)

    # Top provenance sources for transparency
    source_counter: Counter[str] = Counter()
    experiment_records = experimental_data if isinstance(experimental_data, list) else []
    for record in experiment_records:
        if not isinstance(record, dict):
            continue
        prov = record.get("Provenance")
        if isinstance(prov, dict):
            caption = prov.get("SourceCaption") or prov.get("SourceName")
            if caption:
                source_counter[str(caption)] += 1

    if source_counter:
        Story.append(Spacer(1, 0.15 * inch))
        Story.append(Paragraph("Most Frequent Experimental Data Sources", STYLES['Heading3']))
        for source, count in source_counter.most_common(5):
            Story.append(Paragraph(escape(f"{source} - {count} records"), STYLES['ListItem'], bulletText='•'))
        Story.append(Spacer(1, 0.2 * inch))

    # --- Read-Across Strategy Summary ---
    read_across_text = comprehensive_log.get("analysis", {}).get("specialist_agent_outputs", {}).get("Read_Across")
    if read_across_text:
        Story.append(Paragraph("Read-Across Strategy Snapshot", STYLES['Heading2']))
        summary_text = extract_read_across_summary(read_across_text)
        try:
            summary_flowables = parse_markdown_to_flowables(summary_text)
            Story.extend(summary_flowables)
        except Exception as exc:  # Fallback to plain paragraph if parsing fails
            logger.error(f"Read-across summary parsing failed: {exc}", exc_info=True)
            Story.append(Paragraph(escape(summary_text), STYLES['Normal']))
        Story.append(Spacer(1, 0.3 * inch))

    Story.append(PageBreak())

    # --- Synthesized Report Section ---
    # Use the parser to convert Markdown to Flowables
    try:
        report_flowables = parse_markdown_to_flowables(synthesized_report)
        Story.extend(report_flowables)
    except Exception as e:
        logger.error(f"Error parsing Markdown report: {e}", exc_info=True)
        Story.append(Paragraph("Error Processing Report Content", STYLES['Heading1']))
        Story.append(Paragraph(f"The report could not be formatted correctly due to an error during parsing: {e}. Falling back to plain text.", STYLES['Normal']))
        Story.append(Spacer(1, 0.2*inch))
        # Fallback to Preformatted (Code style) if parsing fails
        Story.append(Preformatted(escape(synthesized_report), STYLES['Code']))

    specialist_outputs = comprehensive_log.get("analysis", {}).get("specialist_agent_outputs", {})
    if isinstance(specialist_outputs, dict) and specialist_outputs:
        Story.append(PageBreak())
        Story.append(Paragraph("Specialist Agent Appendices", STYLES['Heading2']))
        Story.append(Paragraph("Full text outputs from each specialist agent are included below for transparency and traceability.", STYLES['Normal']))
        Story.append(Spacer(1, 0.2 * inch))

        agent_order = [
            "Chemical_Context",
            "Physical_Properties",
            "Environmental_Fate",
            "Profiling_Reactivity",
            "Experimental_Data",
            "Metabolism",
            "QSAR_Predictions",
            "Read_Across",
        ]

        def append_agent_section(agent_key: str, agent_text: str):
            Story.append(Paragraph(agent_key.replace("_", " "), STYLES['Heading3']))
            Story.append(Spacer(1, 0.1 * inch))
            try:
                agent_flowables = parse_markdown_to_flowables(agent_text)
                Story.extend(agent_flowables)
            except Exception as agent_error:
                logger.error(f"Failed to parse specialist agent report ({agent_key}): {agent_error}", exc_info=True)
                Story.append(Paragraph("Unable to render this specialist report due to formatting issues.", STYLES['Normal']))
                Story.append(Preformatted(escape(agent_text), STYLES['Code']))
            Story.append(Spacer(1, 0.2 * inch))

        for key in agent_order:
            text = specialist_outputs.get(key)
            if isinstance(text, str) and text.strip():
                append_agent_section(key, text)

        for key, text in specialist_outputs.items():
            if key not in agent_order and isinstance(text, str) and text.strip():
                append_agent_section(key, text)

    else:
        logger.debug("No specialist agent outputs available to include in the PDF appendices.")

    # Build the PDF
    try:
        doc.build(Story)
    except Exception as e:
        logger.error(f"Error during PDF build process: {e}", exc_info=True)
        # Fallback if build fails completely (e.g., invalid XML generated by parser/LLM)
        buffer.seek(0)
        buffer.truncate()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        # Ensure a fallback style is available
        fallback_style = STYLES['Normal'] if STYLES else getSampleStyleSheet()['Normal']
        error_message = f"PDF generation failed during the final build phase. This may be due to complex or invalid formatting generated by the LLM. Error: {e}. Please download the TXT report instead."
        doc.build([Paragraph(error_message, fallback_style)])

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
