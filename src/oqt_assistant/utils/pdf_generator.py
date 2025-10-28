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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
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
    # Repeat header rows across pages and left-align the table
    t = Table(wrapped_data, colWidths=col_widths, repeatRows=1, hAlign='LEFT')

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

# --- EXPERIMENTAL DATA HELPERS (dynamic fields) ---

def _first_present(dct, keys, default="N/A"):
    """Return first non-empty value for any of the keys from dct (case-sensitive)."""
    if not isinstance(dct, dict):
        return default
    for k in keys:
        v = dct.get(k)
        if v not in (None, "", []):
            return v
    return default

def _derive_species_or_model(record):
    meta = record.get("MetaDict") or {}
    parsed = record.get("Parsed_Metadata") or {}
    species = record.get("Species") or _first_present(parsed, [
        "Test organisms (species)", "Organism", "Organism species", "Species"
    ], default=None) or _first_present(meta, [
        "test organisms (species)", "organism", "species"
    ], default=None)

    cell = _first_present(parsed, [
        "Cell short name", "Cell line", "Cell format", "Cell growth mode"
    ], default=None) or _first_present(meta, [
        "cell short name", "cell line", "cell format", "cell growth mode"
    ], default=None)

    tissue = _first_present(parsed, ["Tissue"], default=None) or _first_present(meta, ["tissue"], default=None)

    if species and (cell or tissue):
        return f"{species} ({cell or tissue})"
    return species or cell or tissue or "N/A"

def _derive_source_caption(record):
    prov = record.get("Provenance") if isinstance(record.get("Provenance"), dict) else {}
    parsed = record.get("Parsed_Metadata") or {}
    meta = record.get("MetaDict") or {}
    src = _first_present(parsed, ["Database", "Reference source"], default=None)
    if not src:
        src = _first_present(meta, ["database", "reference source"], default=None)
    return (prov or {}).get("SourceCaption") or src or "N/A"

def _format_value_unit(value, unit):
    try:
        if isinstance(value, (int, float)):
            v = f"{value:g}"
        else:
            s = str(value).strip()
            if re.match(r"^-?\d*\.\d+$", s):
                v = f"{float(s):g}"
            else:
                v = s
        return (v + (f" {unit}" if unit else "")).strip()
    except Exception:
        return (str(value) + (f" {unit}" if unit else "")).strip()

def _pick_metadata_pairs(record):
    """Return ordered key-value pairs of the most informative metadata present for a record.
    Limits to ~18 pairs to fit the page nicely.
    """
    parsed = record.get("Parsed_Metadata") or {}
    meta = record.get("MetaDict") or {}

    priority = [
        ("Assay", None), ("Assay description", None), ("Assay category", None),
        ("Assay technology", None), ("Assay mode", None), ("Assay design type", None),
        ("Cell short name", None), ("Cell format", None), ("Tissue", None),
        ("Test type", None), ("Type of method", None), ("Endpoint type", None),
        ("Metabolic activation", None), ("Applied transforms", None),
        ("Assay provider", None), ("Assay source name", None), ("Assay URL", None),
        ("Reference source", None), ("Database", None),
        ("Test organisms (species)", None), ("Media type", None), ("Duration", None),
        ("Timepoint", None), ("Gene symbol", None), ("Intended target family", None),
    ]

    pairs = []
    for key, _ in priority:
        if key in parsed and parsed[key] not in (None, ""):
            pairs.append((key, str(parsed[key])))
    for key, _ in priority:
        lk = key.lower()
        if any(k == key for k, _ in pairs):
            continue
        if lk in meta and meta[lk] not in (None, ""):
            pairs.append((key, str(meta[lk])))

    if len(pairs) < 8 and isinstance(parsed, dict):
        for k, v in parsed.items():
            if len(pairs) >= 18:
                break
            if v not in (None, "") and not any(k == ek for ek, _ in pairs):
                pairs.append((k, str(v)))

    return pairs[:18]

def _kv_table(pairs, doc_width):
    """Create a compact two-column key/value table with wrapping."""
    if not pairs:
        return None
    left = 1.7*inch
    data = [[Paragraph(f"<b>{escape(str(k))}</b>", metadataStyle), Paragraph(escape(str(v)), metadataStyle)] for k, v in pairs]
    t = Table(data, colWidths=[left, max(doc_width - left, 1*inch)], hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f0f3f6")),
        ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor("#ccd6e0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('FONTSIZE', (0,0), (-1,-1), 8.5),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return t

def _derive_notes(record):
    """Compose a short notes string from key metadata fields (assay/test context).
    Example: "in vitro; RT-CES; ESR1; Tissue: breast; Cell: T47D; +MA"
    """
    parsed = record.get("Parsed_Metadata") or {}
    meta = record.get("MetaDict") or {}

    def get_any(keys):
        v = _first_present(parsed, keys, default=None)
        if v is None:
            v = _first_present(meta, [k.lower() for k in keys], default=None)
        return v

    tokens = []
    # Method / category / endpoint type
    method = get_any(["Type of method", "Assay category", "Endpoint type"]) or get_any(["Test type"]) 
    if method:
        tokens.append(str(method))

    # Technology / mode
    tech = get_any(["Assay technology", "Detection technology", "Detection technology type"]) 
    if tech:
        tokens.append(str(tech))
    mode = get_any(["Assay mode"]) 
    if mode:
        tokens.append(str(mode))

    # Target gene
    gene = get_any(["Gene symbol", "Intended target gene symbol"]) 
    if gene:
        tokens.append(str(gene))

    # Tissue / cell line
    tissue = get_any(["Tissue"]) 
    if tissue:
        tokens.append(f"Tissue: {tissue}")
    cell = get_any(["Cell short name", "Cell line"]) 
    if cell:
        tokens.append(f"Cell: {cell}")

    # Metabolic activation flag
    ma = get_any(["Metabolic activation"]) 
    if isinstance(ma, str) and ma:
        mal = ma.lower()
        if any(x in mal for x in ["not specified", "no", "none", "absent"]):
            tokens.append("-MA")
        else:
            tokens.append("+MA")

    # Compose and trim
    if not tokens:
        return "N/A"
    s = "; ".join(tokens)
    if len(s) > 160:
        s = s[:157] + "…"
    return s

# --- HELPER FUNCTIONS FOR METABOLISM AND PROFILER TABLES ---

def _safe_na(value, na="N/A"):
    """Normalize empty/None/blank values to a standard NA string."""
    if value is None:
        return na
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return na
    return s

def _extract_year_from_record(record):
    """Best-effort year extraction for existing logs without full preprocessing."""
    yr = record.get("Publication_Year")
    if isinstance(yr, int) and 1800 <= yr <= 2200:
        return yr
    try:
        # Search inside parsed metadata text fields
        parsed = record.get("Parsed_Metadata") or {}
        candidates = []
        for k in (
            "Year", "Publication year", "PublicationYear", "StudyYear",
            "Ref Year", "RefYear", "Published date", "Created date", "Reference source"
        ):
            v = parsed.get(k)
            if v:
                candidates.append(str(v))
        # Also scan top-level Reference-like fields if present
        for k in ("Reference", "Notes", "Citation", "Source"):
            v = record.get(k)
            if v:
                candidates.append(str(v))
        import re as _re
        for text in candidates:
            m = _re.search(r"(19|20|21)\d{2}", text)
            if m:
                val = int(m.group(0))
                if 1850 <= val <= 2200:
                    return val
    except Exception:
        pass
    return None

def _render_metabolism_simulators_table(story, styles, log, doc_width):
    """Render table of metabolism simulators with Name, GUID, Status, and metabolite count"""
    try:
        metab = (
            log.get("data_retrieval", {})
               .get("processed_qsar_toolbox_data", {})
               .get("metabolism", {})
        )
        sims = metab.get("simulations", {}) or {}
        if not sims:
            return

        # Simulator name mapping based on common GUID prefixes
        simulator_names = {
            "981641a6": "Autoxidation simulator",
            "8b22d7ba": "Autoxidation simulator (alkaline medium)",
            "efbc6766": "Dissociation simulator",
            "cb0e8397": "Hydrolysis simulator (acidic)",
            "f81521d5": "Hydrolysis simulator (basic)",
            "27870237": "Hydrolysis simulator (neutral)",
            "bfd4dcfb": "in vivo Rat metabolism simulator",
            "740eedf3": "Microbial metabolism simulator",
            "4bb2fbcb": "Observed Mammalian metabolism",
            "b8abff65": "Observed Microbial metabolism",
            "80936612": "Observed Rat In vivo metabolism",
            "bf12b06c": "Observed rat liver metabolism with quantitative data",
            "15d7ae50": "Observed Rat Liver S9 metabolism",
            "cc97d91b": "Rat liver S9 metabolism simulator",
            "e456e127": "Skin metabolism simulator",
            "e34410c8": "Tautomerism"
        }

        story.append(Spacer(1, 6))
        story.append(Paragraph("Metabolism Overview", styles.get("Heading4", styleH3)))

        data = [["Simulator", "GUID", "Status", "Metabolites (#)"]]
        for guid, sim in sims.items():
            # Try to get simulator name from the mapping, fallback to sim data or GUID prefix
            guid_prefix = guid[:8] if len(guid) >= 8 else guid
            sim_name = simulator_names.get(guid_prefix, sim.get("simulator_name", guid_prefix))
            
            data.append([
                sim_name,
                guid,
                sim.get("status", "—"),
                len(sim.get("metabolites", []) or [])
            ])

        # Column widths scaled to page width to prevent overflow on A4
        col_widths = [
            doc_width * 0.42,  # Simulator
            doc_width * 0.34,  # GUID (truncated)
            doc_width * 0.12,  # Status
            doc_width * 0.12,  # Metabolites (#)
        ]

        # Wrap long text for better layout
        wrapped = [[
            Paragraph("<b>Simulator</b>", tableHeaderStyle),
            Paragraph("<b>GUID</b>", tableHeaderStyle),
            Paragraph("<b>Status</b>", tableHeaderStyle),
            Paragraph("<b>Metabolites (#)</b>", tableHeaderStyle),
        ]]
        for row in data[1:]:
            wrapped.append([
                Paragraph(format_and_escape_text(str(row[0])), tableBodyStyle),
                Paragraph(format_and_escape_text(str(row[1])), tableBodyStyle),
                Paragraph(format_and_escape_text(str(row[2])), tableBodyStyle),
                Paragraph(format_and_escape_text(str(row[3])), tableBodyStyle),
            ])

        t = Table(wrapped, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f3f6")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.HexColor("#0b1f33")),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,0), 9),
            ("GRID",       (0,0), (-1,-1), 0.25, colors.HexColor("#ccd6e0")),
            ("ALIGN",      (-1,1), (-1,-1), "CENTER"),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(t)

        # Optional: small note if truncation occurred
        note = metab.get("note")
        if note:
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<i>{note}</i>", styles["BodyText"]))
    except Exception as e:
        story.append(Paragraph(f"<font color='red'>[Metabolism table render failed: {e}]</font>", styles["BodyText"]))


def _render_profilers_table(story, styles, log, doc_width):
    """Render table of profilers with Name, GUID, and category count"""
    try:
        prof = (
            log.get("data_retrieval", {})
               .get("processed_qsar_toolbox_data", {})
               .get("profiling", {})
        )
        if not prof:
            return
        
        # Handle different profiling data structures
        results = prof.get("results", {})
        if not results:
            return
        
        story.append(Spacer(1, 6))
        story.append(Paragraph("Profilers Executed (Name • GUID • Categories)", styles.get("Heading4", styleH3)))
        
        data = [["Profiler", "GUID", "Categories (#)"]]
        for profiler_name, profiler_data in results.items():
            if isinstance(profiler_data, dict):
                guid = profiler_data.get("guid", "—")
                result = profiler_data.get("result", [])
                cat_count = len(result) if isinstance(result, list) else 0
                data.append([
                    profiler_name,
                    guid[:36] if guid and len(guid) > 36 else guid,  # Truncate if needed
                    cat_count
                ])
        
        if len(data) > 1:  # If we have data beyond headers
            # Scale column widths to available width to avoid overflow
            col_widths = [
                doc_width * 0.50,  # Profiler
                doc_width * 0.36,  # GUID (truncated)
                doc_width * 0.14,  # Categories (#)
            ]

            # Wrap long text in Paragraphs for proper line breaks
            wrapped = [[
                Paragraph("<b>Profiler</b>", tableHeaderStyle),
                Paragraph("<b>GUID</b>", tableHeaderStyle),
                Paragraph("<b>Categories (#)</b>", tableHeaderStyle),
            ]]
            for row in data[1:]:
                wrapped.append([
                    Paragraph(format_and_escape_text(str(row[0])), tableBodyStyle),
                    Paragraph(format_and_escape_text(str(row[1])), tableBodyStyle),
                    Paragraph(format_and_escape_text(str(row[2])), tableBodyStyle),
                ])

            t = Table(wrapped, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f3f6")),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.HexColor("#0b1f33")),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",   (0,0), (-1,0), 9),
                ("GRID",       (0,0), (-1,-1), 0.25, colors.HexColor("#ccd6e0")),
                ("ALIGN",      (-1,1), (-1,-1), "CENTER"),
                ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ]))
            story.append(t)
    except Exception as e:
        story.append(Paragraph(f"<font color='red'>[Profilers table render failed: {e}]</font>", styles["BodyText"]))


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

    # --- Logo Section ---
    try:
        import os
        from PIL import Image as PILImage
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        logo_candidates = [
            "o-qt_logo.png",
            "o-qt_logo.jpg",
            "O-QT-Pro.jpg",
            "logo.png",
        ]
        logo_path = None
        for name in logo_candidates:
            candidate = os.path.join(project_root, name)
            if os.path.exists(candidate):
                logo_path = candidate
                break

        if logo_path:
            pil_img = PILImage.open(logo_path)
            width, height = pil_img.size
            aspect_ratio = width / height if height else 1.0
            logo_width = 1.8 * inch
            logo_height = logo_width / aspect_ratio
            logo = RLImage(logo_path, width=logo_width, height=logo_height)
            Story.append(logo)
            Story.append(Spacer(1, 0.15*inch))
        else:
            logger.info("No logo file found for PDF header.")
    except Exception as e:
        logger.warning(f"Could not load O-QT logo: {e}")

    # --- Title ---
    title = f"O-QT Assistant Summary Report"
    Story.append(Paragraph(title, styleH1))
    
    # --- Subtitle with chemical name ---
    inputs = log_data.get("inputs", {})
    subtitle = f"Target Chemical: {inputs.get('identifier', 'N/A')}"
    Story.append(Paragraph(subtitle, ParagraphStyle('subtitle', parent=styleN, fontSize=12, textColor=colors.HexColor("#555555"))))
    Story.append(Spacer(1, 0.05*inch))
    
    # --- Origin Note ---
    origin_note = "Origin: Generated by the O-QT Assistant using OECD QSAR Toolbox data."
    Story.append(Paragraph(origin_note, ParagraphStyle('origin', parent=styleN, fontSize=9, textColor=colors.HexColor("#666666"), fontName='Helvetica-Oblique')))
    Story.append(Spacer(1, 0.2*inch))
    
    # --- Key Studies Coverage Badge (QC Summary) ---
    try:
        experimental_data = log_data.get("data_retrieval", {}).get("processed_qsar_toolbox_data", {}).get("experimental_data", [])
        if experimental_data:
            key_studies = [r for r in experimental_data if r.get("IsKeyStudy")]
            
            # Get unique families with key studies
            families_with_keys = set()
            for record in key_studies:
                family = record.get("Family")
                if family:
                    families_with_keys.add(family)
            
            # Get all unique families
            all_families = set()
            for record in experimental_data:
                family = record.get("Family")
                if family:
                    all_families.add(family)
            
            num_families_with_keys = len(families_with_keys)
            num_all_families = len(all_families)
            
            # Determine status
            if num_families_with_keys == 0:
                status = "FAIL"
                badge_color = colors.HexColor("#d32f2f")  # Red
                status_text = f"Key Studies Coverage: {num_families_with_keys}/{num_all_families} families (FAIL)"
            elif num_families_with_keys < num_all_families * 0.5:
                status = "PARTIAL"
                badge_color = colors.HexColor("#f57c00")  # Orange
                status_text = f"Key Studies Coverage: {num_families_with_keys}/{num_all_families} families (PARTIAL)"
            else:
                status = "PASS"
                badge_color = colors.HexColor("#388e3c")  # Green
                status_text = f"Key Studies Coverage: {num_families_with_keys}/{num_all_families} families (PASS)"
            
            # Create badge table
            badge_data = [[Paragraph(f"<b>{status_text}</b>", ParagraphStyle('badge', parent=styleN, fontSize=10, textColor=colors.whitesmoke))]]
            badge_table = Table(badge_data, colWidths=[doc_width])
            badge_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), badge_color),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ]))
            Story.append(badge_table)
    except Exception as e:
        logger.warning(f"Could not generate key studies badge: {e}")
    
    Story.append(Spacer(1, 0.2*inch))

    # --- Run Details Section ---
    Story.append(Paragraph("Run Details", styleH2))
    Story.append(Spacer(1, 0.1*inch))

    metadata = log_data.get("metadata", {})
    configuration = log_data.get("configuration", {})

    # Prepare data for Run Details table
    data = [
        ['Generated On', metadata.get("timestamp", "N/A")],
        ['Tool Version', metadata.get("version", "N/A")],
    ]

    # Create compact metadata table
    wrapped_data = []
    for row in data:
        wrapped_row = [
            Paragraph(f"<b>{escape(str(row[0]))}</b>", metadataStyle), 
            Paragraph(escape(str(row[1])), metadataStyle)
        ]
        wrapped_data.append(wrapped_row)

    # Narrower first column, shift table left
    meta_col_widths = [1.3*inch, doc_width - 1.3*inch]
    t = Table(wrapped_data, colWidths=meta_col_widths, hAlign='LEFT')

    # Improved styling
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f0f3f6")),
        ('TEXTCOLOR',(0,0), (-1,-1), colors.HexColor("#0b1f33")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ccd6e0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))

    Story.append(t)
    Story.append(Spacer(1, 0.15*inch))
    
    # --- Provenance Legend ---
    Story.append(Paragraph("Provenance Legend", styleH3))
    Story.append(Spacer(1, 0.05*inch))
    
    legend_data = [
        ['Experimental (Toolbox)', 'Value retrieved directly from the user\'s OECD QSAR Toolbox instance via WebAPI.'],
        ['QSAR Estimate (Toolbox)', 'Calculated or profiled result generated by the Toolbox during the same session.'],
        ['LLM Narrative', 'Interpretation created by the configured LLM; never substitutes the raw values above.']
    ]
    
    legend_wrapped = []
    for row in legend_data:
        legend_wrapped.append([
            Paragraph(f"<b>{escape(str(row[0]))}</b>", metadataStyle),
            Paragraph(escape(str(row[1])), metadataStyle)
        ])
    
    legend_table = Table(legend_wrapped, colWidths=[1.6*inch, doc_width - 1.6*inch], hAlign='LEFT')
    legend_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f0f3f6")),
        ('TEXTCOLOR',(0,0), (-1,-1), colors.HexColor("#0b1f33")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ccd6e0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    
    Story.append(legend_table)
    Story.append(Spacer(1, 0.15*inch))
    
    # --- Inputs and Context ---
    Story.append(Paragraph("Inputs and Context", styleH3))
    Story.append(Spacer(1, 0.05*inch))
    
    input_data = [
        ['Identifier', inputs.get("identifier", "N/A")],
        ['Search Type', inputs.get("search_type", "N/A")],
        ['Context', inputs.get("context", "N/A")]
    ]
    
    input_wrapped = []
    for row in input_data:
        input_wrapped.append([
            Paragraph(f"<b>{escape(str(row[0]))}</b>", metadataStyle),
            Paragraph(escape(str(row[1])), metadataStyle)
        ])
    
    input_table = Table(input_wrapped, colWidths=[1.1*inch, doc_width - 1.1*inch], hAlign='LEFT')
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f0f3f6")),
        ('TEXTCOLOR',(0,0), (-1,-1), colors.HexColor("#0b1f33")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ccd6e0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    
    Story.append(input_table)
    Story.append(Spacer(1, 0.15*inch))
    
    # --- Configuration ---
    Story.append(Paragraph("Configuration", styleH3))
    Story.append(Spacer(1, 0.05*inch))
    
    llm_config = configuration.get("llm_configuration", {})
    qsar_config = configuration.get("qsar_toolbox_configuration", {})
    
    config_data = [
        ['LLM Provider', llm_config.get("provider", "N/A")],
        ['LLM Model', llm_config.get("model_name", "N/A")],
        ['QSAR Toolbox API', qsar_config.get("api_url", "N/A")]
    ]
    
    config_wrapped = []
    for row in config_data:
        config_wrapped.append([
            Paragraph(f"<b>{escape(str(row[0]))}</b>", metadataStyle),
            Paragraph(escape(str(row[1])), metadataStyle)
        ])
    
    config_table = Table(config_wrapped, colWidths=[1.3*inch, doc_width - 1.3*inch], hAlign='LEFT')
    config_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f0f3f6")),
        ('TEXTCOLOR',(0,0), (-1,-1), colors.HexColor("#0b1f33")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ccd6e0")),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    
    Story.append(config_table)
    
    # Add metabolism simulators table (detailed view)
    _render_metabolism_simulators_table(Story, styles, log_data, doc_width)
    
    # Add profilers table (transparency)
    _render_profilers_table(Story, styles, log_data, doc_width)

    # --- Optional: Chemical structure snapshot (3D snapshot if possible, else 2D) ---
    try:
        # Retrieve SMILES from the processed log data - handle both old and new structure
        data_retrieval = log_data.get("data_retrieval", {})
        processed_data = data_retrieval.get("processed_qsar_toolbox_data", {})
        
        # Try to get basic_info from nested structure
        basic = {}
        if isinstance(processed_data, dict):
            chemical_data = processed_data.get("chemical_data", {})
            if isinstance(chemical_data, dict):
                basic = chemical_data.get("basic_info", {})
        
        smiles = basic.get("Smiles") if isinstance(basic, dict) else None
        if smiles:
            # Try RDKit 2D as a robust fallback; py3Dmol screenshot is unreliable headless.
            from rdkit import Chem
            from rdkit.Chem import Draw
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(500, 360))
                png_buf = io.BytesIO()
                img.save(png_buf, format="PNG")
                png_buf.seek(0)
                Story.append(Spacer(1, 0.15*inch))
                Story.append(Paragraph("Structure depiction (snapshot)", styleH2))
                Story.append(Spacer(1, 0.1*inch))
                Story.append(RLImage(png_buf, width=5.5*inch, height=4.0*inch))
                Story.append(Spacer(1, 0.2*inch))
    except Exception as e:
        logger.warning(f"Could not embed structure image: {e}")

    # --- 2. Synthesized Report Section (UPDATED) ---
    Story.append(PageBreak())
    Story.append(Paragraph("2. Synthesized Analysis Report", styleH2))
    Story.append(Spacer(1, 0.2*inch))

    synthesized_report = log_data.get("analysis", {}).get("synthesized_report", "Report content not available.")

    # Process report content (handling tables, lists, and paragraphs)
    process_report_content(synthesized_report, Story, doc_width)

    # --- NEW: Key Studies Section (Top-ranked by Klimisch/Adequacy) ---
    Story.append(PageBreak())
    Story.append(Paragraph("3. Key Studies (Klimisch Reliability)", styleH2))
    Story.append(Spacer(1, 0.2*inch))

    experimental_data = log_data.get("data_retrieval", {}).get("processed_qsar_toolbox_data", {}).get("experimental_data", [])

    if experimental_data:
        # Coverage summary by family for transparency
        try:
            fam_totals = {}
            fam_keys = {}
            for r in experimental_data:
                fam = r.get("Family") or "Unspecified"
                fam_totals[fam] = fam_totals.get(fam, 0) + 1
                if r.get("IsKeyStudy"):
                    fam_keys[fam] = fam_keys.get(fam, 0) + 1

            cov_table = [["Family", "Total Records", "Key Studies"]]
            for fam in sorted(fam_totals.keys()):
                cov_table.append([
                    fam,
                    str(fam_totals.get(fam, 0)),
                    str(fam_keys.get(fam, 0)),
                ])
            Story.append(Paragraph("Coverage by Family", styleH3))
            Story.append(Spacer(1, 0.05*inch))
            Story.append(create_rl_table(cov_table, doc_width))
            Story.append(Spacer(1, 0.15*inch))
        except Exception as e:
            logger.warning(f"Family coverage summary failed: {e}")

        # Key studies list
        # Filter for key studies (flagged by IsKeyStudy)
        key_studies = [r for r in experimental_data if r.get("IsKeyStudy")]
        
        if key_studies:
            Story.append(Paragraph(f"Found {len(key_studies)} key studies (Klimisch 1 or flagged as 'Key study')", styleN))
            Story.append(Spacer(1, 0.1*inch))
            
            key_data = []
            headers = ["Family/Endpoint", "Value ± Unit", "Reliability", "Adequacy", "Year", "Species/Model", "DB Caption", "DataId"]
            key_data.append(headers)
            
            for record in key_studies[:30]:  # Cap at 30 for readability
                family = record.get("Family", "")
                endpoint = record.get("Endpoint", "")
                value_unit = _format_value_unit(record.get("Value", ""), record.get("Unit", ""))
                
                row = [
                    _safe_na(f"{family} / {endpoint}" if family else endpoint),
                    _safe_na(value_unit),
                    _safe_na(record.get("Reliability")),
                    _safe_na(record.get("AdequacyOfStudy")),
                    _safe_na(_extract_year_from_record(record)),
                    _safe_na(_derive_species_or_model(record)),
                    _safe_na(_derive_source_caption(record)),
                    _safe_na(record.get("DataId"))
                ]
                key_data.append(row)
            
            table = create_rl_table(key_data, doc_width)
            Story.append(table)
            Story.append(Spacer(1, 0.2*inch))
        else:
            Story.append(Paragraph("No records flagged as key studies (Klimisch 1 or explicit 'Key study' markers) were detected.", styleN))
            Story.append(Spacer(1, 0.1*inch))
        
        # All Experimental Data (Provenance)
        Story.append(Paragraph("3.1 All Experimental Data (Provenance)", styleH3))
        Story.append(Spacer(1, 0.1*inch))
        
        provenance_data = []
        # Keep columns compact: drop DataId here and add a Notes summary column
        headers = ["Endpoint", "Value/Unit", "Year", "Species/Model", "DB Caption", "Notes"]
        provenance_data.append(headers)

        # Sort by Publication_Year (desc) when available to surface recent studies
        exp_sorted = sorted(experimental_data, key=lambda r: (r.get('Publication_Year') is None, r.get('Publication_Year')), reverse=True)
        for record in exp_sorted[:100]:  # Cap at 100
            year = _extract_year_from_record(record)
            row = [
                _safe_na(record.get("Endpoint")),
                _safe_na(_format_value_unit(record.get("Value", ""), record.get("Unit", ""))),
                _safe_na(year if year is not None else None),
                _safe_na(_derive_species_or_model(record)),
                _safe_na(_derive_source_caption(record)),
                _safe_na(_derive_notes(record)),
            ]
            provenance_data.append(row)

        if len(provenance_data) > 1:
            table = create_rl_table(provenance_data, doc_width)
            Story.append(table)
            Story.append(Spacer(1, 0.15*inch))

            # Study Metadata Details (Selected)
            try:
                Story.append(Paragraph("3.1.1 Study Metadata Details (selected)", styleH3))
                Story.append(Spacer(1, 0.05*inch))

                rich = []
                for r in experimental_data:
                    parsed = r.get("Parsed_Metadata") or {}
                    meta = r.get("MetaDict") or {}
                    # Heuristic: prefer records with assay/test info or many metadata keys
                    if any(k in parsed for k in ("Assay", "Test type", "Endpoint type")) or len(parsed) >= 10:
                        rich.append(r)
                    if len(rich) >= 6:
                        break

                if not rich:
                    rich = experimental_data[:3]

                for idx, rec in enumerate(rich, 1):
                    header = (
                        f"{rec.get('Endpoint','N/A')} • "
                        f"{_format_value_unit(rec.get('Value',''), rec.get('Unit',''))} • "
                        f"{_derive_source_caption(rec)}"
                    )
                    Story.append(Paragraph(header, styles.get('Heading4', styleH3)))
                    Story.append(Spacer(1, 0.05*inch))
                    pairs = _pick_metadata_pairs(rec)
                    tbl = _kv_table(pairs, doc_width)
                    if tbl:
                        Story.append(tbl)
                        Story.append(Spacer(1, 0.15*inch))
            except Exception as e:
                logger.warning(f"Study metadata details rendering failed: {e}")

            # Grouped by source sub-tables
            try:
                Story.append(Spacer(1, 0.05*inch))
                Story.append(Paragraph("3.1.2 Experimental Data by Source", styleH3))
                Story.append(Spacer(1, 0.05*inch))

                # Build groups
                groups = {}
                for r in experimental_data:
                    src = _derive_source_caption(r)
                    groups.setdefault(src, []).append(r)

                # Sort groups by size desc, then name
                for src, records in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                    Story.append(Paragraph(f"Source: {src} (showing up to 20 of {len(records)})", styles.get('Heading4', styleH3)))
                    Story.append(Spacer(1, 0.04*inch))

                    # Sort each group by year desc
                    rs = sorted(records, key=lambda r: (_extract_year_from_record(r) is None, _extract_year_from_record(r)), reverse=True)[:20]
                    table_data = [["Endpoint", "Value/Unit", "Year", "Species/Model", "DataId", "Notes"]]
                    for rec in rs:
                        table_data.append([
                            _safe_na(rec.get('Endpoint')),
                            _safe_na(_format_value_unit(rec.get('Value',''), rec.get('Unit',''))),
                            _safe_na(_extract_year_from_record(rec)),
                            _safe_na(_derive_species_or_model(rec)),
                            _safe_na(rec.get('DataId')),
                            _safe_na(_derive_notes(rec)),
                        ])

                    Story.append(create_rl_table(table_data, doc_width))
                    Story.append(Spacer(1, 0.12*inch))
            except Exception as e:
                logger.warning(f"Grouped source sub-tables failed: {e}")
        else:
            Story.append(Paragraph("No experimental data available.", styleN))
    else:
        Story.append(Paragraph("No experimental data available.", styleN))

    Story.append(Spacer(1, 0.2*inch))
    Story.append(Paragraph("3.2 QSAR Predictions (Applicability Domain)", styleH3))
    Story.append(Spacer(1, 0.1*inch))

    qsar_section = log_data.get("data_retrieval", {}).get("processed_qsar_toolbox_data", {}).get("qsar_models", {})
    qsar_processed = qsar_section.get("processed", {}) if isinstance(qsar_section, dict) else {}
    in_domain = qsar_processed.get("in_domain", []) if isinstance(qsar_processed, dict) else []
    if in_domain:
        table_data = [["Model", "Category", "Value", "Unit", "Runtime (s)", "Donator"]]
        for record in in_domain[:40]:
            table_data.append([
                record.get("caption", "Unnamed"),
                record.get("top_category", record.get("requested_position", "")),
                record.get("value", "") or "N/A",
                record.get("unit", ""),
                f"{record.get('runtime_seconds', 0.0):.2f}",
                record.get("donator", "Unknown")
            ])
        table = create_rl_table(table_data, doc_width)
        Story.append(table)
        Story.append(Spacer(1, 0.2*inch))
    else:
        summary = qsar_processed.get("summary", {}) if isinstance(qsar_processed, dict) else {}
        total = summary.get("total", 0)
        Story.append(Paragraph(
            f"No QSAR predictions were reported within applicability domains (evaluated {total} models).",
            styleN
        ))
        Story.append(Spacer(1, 0.2*inch))


    # --- 4. Specialist Agent Reports Section (UPDATED) ---
    Story.append(PageBreak())
    Story.append(Paragraph("4. Specialist Agent Reports", styleH2))
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
        "QSAR_Predictions",
        "Read_Across"
    ]

    for i, key in enumerate(report_order):
        report_content = specialist_reports.get(key, "Content not available.")
        title = f"4.{i+1} {key.replace('_', ' ')}"

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
