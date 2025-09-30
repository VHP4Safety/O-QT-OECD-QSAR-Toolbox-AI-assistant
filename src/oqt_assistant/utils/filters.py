# oqt_assistant/utils/filters.py
# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Define keywords for identifying record types.
# These are based on common terminology in QSAR Toolbox exports.

MAMMALIAN_TOX_KEYWORDS = [
    "LD50", "LC50", "oral toxicity", "dermal toxicity", "inhalation toxicity",
    "mouse", "rat", "rabbit", "guinea pig", "dog", "mammal",
    "repeated dose toxicity", "carcinogenicity", "mutagenicity",
    "reproductive toxicity", "developmental toxicity", "neurotoxicity",
    "skin irritation", "eye irritation", "sensitisation"
]

ADME_TK_KEYWORDS = [
    "ADME", "TK", "toxicokinetic", "pharmacokinetic", "PBPK",
    "clearance", "Cmax", "AUC", "volume of distribution",
    "half-life", "metabolic rate", "absorption", "distribution",
    "excretion", "bioavailability"
]

# Fields to check for keywords (Updated based on typical QSAR export structure)
FIELDS_TO_CHECK = ["Endpoint", "Test organism", "Species", "Route of administration", "Reference", "TestGuid", "DataType", "Reliability"]

def filter_experimental_records(
    records: List[Dict[str, Any]],
    exclude_adme_tk: bool = False,
    exclude_mammalian_tox: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filters experimental records based on exclusion criteria.
    Returns filtered records and counts of excluded records.
    """
    kept = []
    dropped_mammalian = 0
    dropped_adme = 0

    if not records:
        return [], {"dropped_mammalian": 0, "dropped_adme_tk": 0}

    # If no filters are active, return original list
    if not exclude_adme_tk and not exclude_mammalian_tox:
        return records, {"dropped_mammalian": 0, "dropped_adme_tk": 0}

    for record in records:
        if not isinstance(record, dict):
            kept.append(record) # Keep non-dict records (e.g., error messages)
            continue

        is_mammalian = False
        is_adme = False

        # Combine text from relevant fields and metadata for searching
        text_to_search = []
        for field in FIELDS_TO_CHECK:
            value = record.get(field)
            if value:
                text_to_search.append(str(value).lower())
        
        # Also check Parsed_Metadata if it exists (this structure is created by data_formatter.py)
        metadata = record.get("Parsed_Metadata", {})
        if isinstance(metadata, dict):
            for value in metadata.values():
                text_to_search.append(str(value).lower())

        combined_text = " ".join(text_to_search)

        # Check for keywords
        if exclude_mammalian_tox:
            if any(kw.lower() in combined_text for kw in MAMMALIAN_TOX_KEYWORDS):
                is_mammalian = True

        if exclude_adme_tk:
            if any(kw.lower() in combined_text for kw in ADME_TK_KEYWORDS):
                is_adme = True

        # Apply filtering logic
        # Note: A record can be both mammalian tox and ADME/TK.
        
        if is_mammalian and exclude_mammalian_tox:
            dropped_mammalian += 1
            continue # Skip this record

        if is_adme and exclude_adme_tk:
            dropped_adme += 1
            continue # Skip this record

        kept.append(record)

    logger.info(f"Filtering complete. Kept: {len(kept)}, Dropped Mammalian: {dropped_mammalian}, Dropped ADME/TK: {dropped_adme}")
    return kept, {"dropped_mammalian": dropped_mammalian, "dropped_adme_tk": dropped_adme}