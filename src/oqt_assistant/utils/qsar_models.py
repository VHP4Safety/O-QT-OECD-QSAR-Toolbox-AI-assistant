# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers for selecting and executing QSAR models."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


def _dedupe_preserve_order(values: Iterable[Any]) -> List[Any]:
    seen: set[Any] = set()
    ordered: List[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _score_qsar_entry(entry: Mapping[str, Any]) -> float:
    """Heuristic priority score used when suggesting default QSAR models."""
    score = 0.0
    top_category = str(
        entry.get("TopCategory")
        or entry.get("RequestedPosition")
        or entry.get("Position")
        or ""
    ).lower()
    caption = str(entry.get("Caption") or entry.get("Name") or "").lower()
    donator = str(entry.get("Donator") or entry.get("donator") or "").lower()

    if entry.get("IsRegulatory") in (True, "true", "True"):
        score += 3.0
    if entry.get("Recommended") in (True, "true", "True"):
        score += 1.5

    if "toxic" in top_category or "tox" in top_category:
        score += 2.5
    if "carcin" in top_category:
        score += 2.0
    if "mutagen" in top_category:
        score += 1.5
    if "sensit" in top_category:
        score += 1.0

    if "oecd" in caption or "oecd" in donator:
        score += 1.0
    if "vega" in caption:
        score += 0.8
    if "epa" in donator or "reach" in caption:
        score += 0.5

    priority = entry.get("Priority") or entry.get("priority")
    if isinstance(priority, (int, float)):
        score += max(0.0, 10.0 - float(priority))

    return score


def derive_recommended_qsar_models(
    catalog: Optional[Iterable[Mapping[str, Any]]],
    *,
    max_models: int = 12,
) -> List[Dict[str, Any]]:
    """
    Produce a curated subset of QSAR models that serve as sensible defaults.

    The heuristic favours models flagged as regulatory, widely used toxicological
    endpoints, and well-known donators. The result is deduplicated and stable.
    """
    if not catalog:
        return []

    unique_entries: List[Dict[str, Any]] = []
    seen_guids: set[str] = set()

    for entry in catalog:
        if not isinstance(entry, Mapping):
            continue
        guid = entry.get("Guid") or entry.get("guid")
        if isinstance(guid, str):
            guid_key = guid
        elif guid is not None:
            guid_key = str(guid)
        else:
            guid_key = ""
        if guid_key and guid_key in seen_guids:
            continue
        if guid_key:
            seen_guids.add(guid_key)
        unique_entries.append(dict(entry))

    if not unique_entries:
        return []

    unique_entries.sort(
        key=lambda item: (
            -_score_qsar_entry(item),
            str(item.get("Caption") or item.get("Name") or "").lower(),
        )
    )

    if max_models is not None:
        return unique_entries[:max_models]
    return unique_entries


def format_qsar_model_label(entry: Mapping[str, Any]) -> str:
    """Return a human-friendly label for UI selections."""
    if not isinstance(entry, Mapping):
        return str(entry)

    caption = str(entry.get("Caption") or entry.get("Name") or entry.get("caption") or "Unnamed model").strip()
    top_category = entry.get("TopCategory") or entry.get("RequestedPosition") or entry.get("Position") or entry.get("Endpoint")
    donator = entry.get("Donator") or entry.get("donator") or entry.get("Developer")
    version = entry.get("Version") or entry.get("version")

    qualifiers: List[str] = []
    if top_category:
        qualifiers.append(str(top_category))
    if donator:
        qualifiers.append(str(donator))
    if version:
        qualifiers.append(f"v{version}")

    if qualifiers:
        return f"{caption} — {' • '.join(qualifiers)}"
    return caption


def run_qsar_predictions(
    api_client: Any,
    chem_id: str,
    *,
    selected_model_guids: Sequence[str],
    catalog: Optional[Iterable[Mapping[str, Any]]] = None,
    logger_override: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Execute QSAR models and normalize their responses.

    Args:
        api_client: `QSARToolboxAPI` instance (or analogue) providing `apply_qsar_model`.
        chem_id: Toolbox chemical identifier.
        selected_model_guids: Iterable of QSAR GUIDs to execute.
        catalog: Optional pre-fetched catalog used for metadata enrichment.
        logger_override: Optional logger to use instead of module logger.
    """
    log = logger_override or logger
    selected = [guid for guid in _dedupe_preserve_order(selected_model_guids) if guid]

    baseline = {
        "catalog_size": 0,
        "executed_models": 0,
        "predictions": [],
        "summary": {"total": 0, "in_domain": 0, "not_applicable": 0, "out_of_domain": 0, "errors": 0},
        "selected_model_guids": selected,
    }

    if not selected:
        return baseline

    if catalog is None:
        try:
            catalog = api_client.get_all_qsar_models_catalog()
        except Exception as exc:  # pragma: no cover - defensive guardrail
            log.debug("Unable to fetch QSAR catalog for metadata enrichment: %s", exc)
            catalog = []

    catalog_index: Dict[str, Mapping[str, Any]] = {}
    if catalog:
        for entry in catalog:
            if not isinstance(entry, Mapping):
                continue
            guid = entry.get("Guid") or entry.get("guid")
            if guid:
                catalog_index[str(guid)] = entry

    predictions: List[Dict[str, Any]] = []
    executed = 0
    counts = {"in_domain": 0, "not_applicable": 0, "out_of_domain": 0, "errors": 0}

    for guid in selected:
        guid_key = str(guid)
        meta = catalog_index.get(guid_key, {})
        base_record: Dict[str, Any] = {
            "guid": guid_key,
            "caption": meta.get("Caption") or meta.get("Name") or meta.get("caption") or guid_key,
            "top_category": meta.get("TopCategory") or meta.get("top_category") or meta.get("RequestedPosition") or "",
            "requested_position": meta.get("RequestedPosition") or meta.get("Position") or "",
            "donator": meta.get("Donator") or meta.get("donator") or "",
            "runtime_seconds": 0.0,
            "status": "ok",
        }

        start_time = time.perf_counter()
        try:
            raw_response = api_client.apply_qsar_model(guid_key, chem_id) or {}
            executed += 1
        except Exception as exc:
            runtime = time.perf_counter() - start_time
            base_record["runtime_seconds"] = round(runtime, 3)
            base_record["status"] = "error"
            base_record["error"] = str(exc)
            counts["errors"] += 1
            log.warning("QSAR model %s failed: %s", guid_key, exc)
            predictions.append(base_record)
            continue

        runtime = time.perf_counter() - start_time
        base_record["runtime_seconds"] = round(runtime, 3)

        response_status = str(raw_response.get("status", "ok")).lower()
        explicit_error = raw_response.get("error") or raw_response.get("Error") or raw_response.get("Message")
        if response_status == "error" or explicit_error:
            base_record["status"] = "error"
            base_record["error"] = explicit_error or raw_response.get("message", "Unknown error")
            counts["errors"] += 1
            predictions.append(base_record)
            continue

        domain_result = (
            raw_response.get("DomainResult")
            or raw_response.get("domain_result")
            or raw_response.get("ApplicabilityDomain")
            or "Unknown"
        )
        domain_explain = raw_response.get("DomainExplain") or raw_response.get("domain_explain")
        if domain_explain and not isinstance(domain_explain, (list, tuple)):
            domain_explain = [domain_explain]

        metadata = raw_response.get("MetaData") or raw_response.get("metadata") or []
        if isinstance(metadata, Mapping):
            metadata = [metadata]

        base_record.update(
            {
                "domain_result": domain_result or "Unknown",
                "domain_explain": domain_explain or [],
                "value": raw_response.get("Value") or raw_response.get("value"),
                "qualifier": raw_response.get("Qualifier") or raw_response.get("qualifier"),
                "unit": raw_response.get("Unit") or raw_response.get("unit"),
                "metadata": metadata,
                "endpoint": raw_response.get("Endpoint") or raw_response.get("endpoint"),
                "family": raw_response.get("Family") or raw_response.get("family") or meta.get("Family"),
            }
        )

        domain_lower = str(base_record["domain_result"]).lower()
        if domain_lower.startswith("in"):
            counts["in_domain"] += 1
        elif "not applicable" in domain_lower or "not_applicable" in domain_lower:
            counts["not_applicable"] += 1
        elif "out" in domain_lower or "ambig" in domain_lower:
            counts["out_of_domain"] += 1
        else:
            counts["out_of_domain"] += 1

        predictions.append(base_record)

    summary = {
        "total": len(predictions),
        "in_domain": counts["in_domain"],
        "not_applicable": counts["not_applicable"],
        "out_of_domain": counts["out_of_domain"],
        "errors": counts["errors"],
    }

    return {
        "catalog_size": len(catalog_index) if catalog else 0,
        "executed_models": executed,
        "predictions": predictions,
        "summary": summary,
        "selected_model_guids": selected,
    }
