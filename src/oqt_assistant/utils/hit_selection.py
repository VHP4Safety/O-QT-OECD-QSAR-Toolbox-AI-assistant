"""
Shared helper utilities for selecting the best QSAR Toolbox search hit.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from .data_formatter import format_chemical_data, process_properties
from .qsar_api import QSARToolboxAPI, QSARResponseError, QSARTimeoutError


def _normalize_identifier(text: str) -> str:
    """Canonicalize chemical names for comparison."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _extract_candidate_names(hit: Dict[str, Any]) -> List[str]:
    """Collect name fields from a Toolbox hit."""
    names: List[str] = []
    for key in ("Name", "ChemicalName", "IUPACName"):
        val = hit.get(key)
        if isinstance(val, str):
            names.append(val)
    extra = hit.get("Names")
    if isinstance(extra, list):
        names.extend(str(item) for item in extra if isinstance(item, str))
    return names


def rank_hits_by_quality(identifier: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort Toolbox search hits by how well they match the requested identifier."""
    ident_norm = _normalize_identifier(identifier)
    ident_digits = "".join(ch for ch in identifier if ch.isdigit())
    subtype_rank_map = {
        "monoconstituent": 0,
        "single constituent": 0,
        "simple substance": 0,
        "mono constituent": 0,
        "monomer": 0,
        "known": 1,
        "unknown": 2,
        "multiconstituent": 3,
        "multi-constituent": 3,
        "uvcb": 3,
        "mixture": 3,
    }

    ranked: List[Tuple[Tuple[int, int, int, int, int, int, int, int], Dict[str, Any]]] = []
    for idx, hit in enumerate(hits):
        subtype = str(hit.get("SubstanceType") or "")
        subtype_key = subtype.lower()
        subtype_rank = subtype_rank_map.get(subtype_key, 4)

        names = _extract_candidate_names(hit)
        normalized_names = [_normalize_identifier(n) for n in names if n]
        main_norm = _normalize_identifier(hit.get("Name") or hit.get("ChemicalName") or "")

        primary_exact = 0 if ident_norm and main_norm == ident_norm else 1
        exact_matches = sum(1 for n in normalized_names if ident_norm and n == ident_norm)
        contains_matches = sum(1 for n in normalized_names if ident_norm and ident_norm in n)
        cas_raw = str(hit.get("Cas") or hit.get("CAS") or "")
        cas_digits = "".join(ch for ch in cas_raw if ch.isdigit())
        cas_exact_match = bool(ident_digits) and cas_digits == ident_digits
        cas_priority = 0 if cas_exact_match else (0 if not ident_digits else 1)
        has_any_match = 0 if (exact_matches or contains_matches or cas_exact_match) else 1

        cas_rank = 0 if cas_digits and cas_digits != "0" else 1

        has_smiles = bool(hit.get("Smiles"))
        structure_rank = 0 if has_smiles else 1

        ranked.append(
            (
                (
                    cas_priority,
                    subtype_rank,
                    primary_exact,
                    has_any_match,
                    -exact_matches,
                    -contains_matches,
                    cas_rank,
                    structure_rank,
                    len(normalized_names),
                    idx,
                ),
                hit,
            )
        )

    ranked.sort(key=lambda item: item[0])
    return [hit for _, hit in ranked]


def select_hit_with_properties(
    api: QSARToolboxAPI,
    identifier: str,
    hits: List[Dict[str, Any]],
    *,
    logger: logging.Logger | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str, List[str]]:
    """
    Pick the best Toolbox hit for the identifier and ensure calculators can be retrieved.

    Returns (formatted_basic_info, processed_properties, chem_id, selection_notes).
    Raises RuntimeError if no usable hit can be confirmed.
    """

    log = logger or logging.getLogger(__name__)
    ordered_hits = rank_hits_by_quality(identifier, hits)

    selection_notes: List[str] = []

    for idx, candidate in enumerate(ordered_hits, start=1):
        basic_candidate = format_chemical_data(candidate)
        chem_id_candidate = basic_candidate.get("ChemId")

        if not chem_id_candidate:
            note = f"Hit {idx} missing ChemId."
            log.warning("  [Data Fetch] %s", note)
            selection_notes.append(note)
            continue

        log.info(
            "  [Data Fetch] Evaluating hit %s/%s: ChemId %s (%s)",
            idx,
            len(ordered_hits),
            chem_id_candidate,
            basic_candidate.get("SubstanceType", "Unknown"),
        )

        try:
            raw_props_candidate = api.apply_all_calculators(chem_id_candidate) or {}
            props_candidate = process_properties(raw_props_candidate)
        except (QSARResponseError, QSARTimeoutError) as exc:
            note = f"Calculator retrieval failed for ChemId {chem_id_candidate}: {exc}"
            log.warning("  [Data Fetch] %s", note)
            selection_notes.append(note)
            continue
        except Exception as exc:  # Catch unexpected issues and continue trying
            note = f"Unexpected failure for ChemId {chem_id_candidate}: {exc}"
            log.warning("  [Data Fetch] %s", note)
            selection_notes.append(note)
            continue

        log.info(
            "  [Data Fetch] Selected ChemId %s (%s); retrieved %s calculator values.",
            chem_id_candidate,
            basic_candidate.get("SubstanceType", "Unknown"),
            len(props_candidate),
        )
        return basic_candidate, props_candidate, chem_id_candidate, selection_notes

    detail = " | ".join(selection_notes) if selection_notes else "No usable hits."
    raise RuntimeError(f"Unable to retrieve usable Toolbox record for {identifier}. Details: {detail}")
