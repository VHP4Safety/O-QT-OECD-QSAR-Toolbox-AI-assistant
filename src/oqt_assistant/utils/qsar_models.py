"""Helpers for executing QSAR model predictions with applicability-domain tracking."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from .qsar_api import (
    QSARToolboxAPI,
    QSARResponseError,
    QSARTimeoutError,
    QSARConnectionError,
)

# Heuristics used to derive a fast-yet-covering QSAR preset
FAST_QSAR_DONOR_KEYWORDS: tuple[str, ...] = (
    "caesar",
    "leadscope",
    "toxmatch",
    "isafe",
    "kreatis",
    "epa",
    "toxtree",
    "icat",
    "catmos",
    "vivo",
    "ecotox",
    "danish",  # Danish EPA models are generally fast
)

# Known slow model GUIDs to exclude from recommendations (>10 seconds runtime)
SLOW_QSAR_MODEL_GUIDS: set[str] = {
    # VEGA IRFMN models (10-90+ seconds)
    "5e9d7a1e-b0dc-4eb7-a8c0-622cc97c1758",  # VEGA - Algae Acute (EC50) - IRFMN (19.89s)
    "23797e8a-4c0e-4542-9fdb-4e769e6c9b98",  # VEGA - Algae Chronic (NOEC) - IRFMN (19.45s)
    "41927145-a60e-4b32-a201-a220411c4d06",  # VEGA - Fish Acute (LC50) - IRFMN (18.51s)
    "1ddfc8bc-e8ef-4a2c-b2e5-f38ac41f05f3",  # VEGA - Fish Chronic (NOEC) - IRFMN (16.08s)
    "6ece9151-0207-441c-8862-bd3abb5cf8c7",  # VEGA - Daphnia Magna Acute (EC50) - IRFMN (12.42s)
    "96cb4fe7-bc6d-4bb8-930f-df1a20cc145b",  # VEGA - Daphnia Magna Chronic (NOEC) - IRFMN (12.26s)
    "18f5a169-3985-4dde-89b4-54058b3633a1",  # VEGA - BCF model (Arnot-Gobas) (11.40s)
    "9dcaa97c-bd11-420a-8bc8-770324496760",  # VEGA - Carcinogenicity inhalation Slope Factor - IRFMN (11.29s)
    "06e49033-ac62-4773-8b44-49cb3ad99e84",  # VEGA - Carcinogenicity oral Slope Factor - IRFMN (12.98s)
    "66f66df4-cc3f-4c8b-a932-8ea409f17b44",  # VEGA - Carcinogenicity in female Rat (CORAL) (13.11s)
    "0e23306d-e68d-4e1d-a233-8d9cc47fc713",  # VEGA - Adipose tissue - blood model (INERIS) (15.73s)
    "bd99390e-49a9-4dea-a30c-1506c7869aa9",  # VEGA - Total body elimination half-life (QSARINS) (50.14s)
    "fa5b0f2e-1770-421c-9565-dfd9e4b54ddd",  # VEGA - Plasma Protein Binding - LogK (IRFMN) (21.62s)
    "a6107f3b-17a5-4338-9cb5-e393f7b8f2c0",  # VEGA - Hepatic Steatosis MIE - PXR up (TOXCAST) (22.57s)
    "734f4ab8-2d26-40fe-b345-469e2a50089a",  # VEGA - Hepatic Steatosis MIE - PPARg up (TOXCAST) (20.79s)
    "9e25a61c-7b07-40d1-9350-0584b229e49a",  # VEGA - Hepatic Steatosis MIE - NRF2 (TOXCAST) (18.49s)
    "ce2e200f-bfa9-4b7c-8396-185d1f9a278d",  # VEGA - Melting Point (CONCERT/KNN) (21.71s)
    # Ultra-slow models (>90 seconds)
    "2d396fc4-3d33-4795-a289-7e45c921c627",  # VEGA - Aromatase activity model (IRFMN) (92.09s)
    "6ecc55b8-cfde-4e91-ac78-13609e1d1168",  # VEGA - Thyroperoxidase Inhibitory Activity (OBERON) (92.10s)
    "54b0f681-c40a-49aa-9351-a2192cfdc9ab",  # VEGA - LogP model (ALogP) (92.07s)
    "44a22809-193b-4d4c-b88a-d68ab3837ee7",  # VEGA - LogP model (MLogP) (92.18s)
    "1cc9d278-17c9-416b-96fc-8fbe1d64680e",  # VEGA - LogP model (Meylan-Kowwin) (92.20s)
    # Problematic models (often return 500 errors)
    "eaecae22-11d4-403f-aaf4-db7020633c86",  # Daphnia magna 48h EC50 - Danish QSAR DB battery model
    "9c92bfa8-a740-4feb-ac9f-c6a8aa3c430c",  # Daphnia magna 48h EC50 - Danish QSAR DB Leadscope model
    "fd20442b-e70b-441e-9ebd-3b304e189c40",  # Daphnia magna 48h EC50 - Danish QSAR DB SciQSAR model
}

# Curated publication-grade preset (balanced coverage, 50 models total).
CURATED_QSAR_GUIDS: tuple[str, ...] = (
    "7d8932eb-9fb8-45af-92a0-e7dc11e5b2a1",
    "6592a47b-75e9-4055-80d0-dc7693077870",
    "873e450d-5bf1-4d81-b2f2-997ed3ab7f7a",
    "a227dc2c-4210-41da-b788-95e562560dd0",
    "44a22809-193b-4d4c-b88a-d68ab3837ee7",
    "0a590899-12a7-466c-a8dc-aa584870d0e6",
    "3fa56226-315a-4e18-9021-1df8eaee5fd3",
    "1e7d360f-85da-4957-84cf-17de3dee9538",
    "9bb94df1-4788-4f17-bbd9-ab4c562091d0",
    "53582332-9126-4f7a-9169-b5cd9f5746e4",
    "3dbfd255-d4ec-4d6c-ad2d-5da60e8bfa6a",
    "3b43193c-ef68-4fbc-8bca-a345b804aedc",
    "0415371c-5a96-4382-820e-14c2d3214913",
    "64331e8b-92b7-4175-84c2-5af114b0cf84",
    "bc0a335b-d524-4c11-9346-1318ff353c01",
    "0aca345d-becd-478a-a680-5203f94405fc",
    "b7cb24cc-a3e5-4eb1-b7e0-f269e98ddb62",
    "37301de8-71bb-4fa5-af06-5a4a9f728292",
    "8dbe60c0-1254-4c38-b480-5838e3cf5cd8",
    "5656b084-830e-4581-b734-402e90bfc7bb",
    "2bf928f6-5430-44fe-a20d-2df08d011d07",
    "249fd511-d84e-4502-9d47-6c4076db6917",
    "bd99390e-49a9-4dea-a30c-1506c7869aa9",
    "f2d0c665-8816-423b-9e62-96ac5a6061a9",
    "cfbc2235-1372-44d9-9734-c8795d437591",
    "e5e2dab3-ae15-254b-d7a8-5e2c210c2175",
    "c0480dd9-3959-1866-5a3c-19b21c905034",
    "a2029ec9-dc8a-708a-02ec-ff11b86488ea",
    "e6a7660e-900d-83be-6479-58cd31e025b1",
    "b3d3b207-cef1-fb5b-4a44-60d5371e413c",
    "7b65a29d-927e-1bd9-cbcc-9b9d645e9a19",
    "26cb5dca-78d8-dc9b-0007-373280b78019",
    "f9e4594d-4303-3c2e-cae5-a57589fba82d",
    "68f2567f-71db-478e-adf6-39360aa00f59",
    "c0109778-3b84-4311-834a-c97c94f75244",
    "0502e196-f736-42cb-bbe1-833d5efa5273",
    "249a3822-8126-416f-a85c-fc24039e607c",
    "773465df-3388-49cd-b4c8-253508a5aa6e",
    "72dd9f8c-dd69-41b5-854c-1aa422a6e371",
    "8d694a3a-16a8-4500-9c12-41a14672884c",
    "23ded23c-3538-4af6-9dfc-cbdc08f05dbc",
    "0e2788dc-969c-448c-a151-149138fd32d9",
    "d4eefe3f-45f0-498a-96f6-9b42fae26a8b",
    "e4296a17-76a4-4099-8172-8e7d0590d393",
    "d3df8799-2543-446e-ae56-93ecab87f124",
    "54b0f681-c40a-49aa-9351-a2192cfdc9ab",
    "16251e73-9841-497b-b293-80b0a040f1fa",
    "1cc9d278-17c9-416b-96fc-8fbe1d64680e",
    "185c398c-22cc-4577-bd1a-a6ce6c81d0b6",
    "c8055d19-f4f0-4de9-8a5d-558d56d3b593",
)

CURATED_QSAR_LIMIT = len(CURATED_QSAR_GUIDS)

# (keyword, weight) pairs used to prioritize common regulatory endpoints
QSAR_PRIORITY_KEYWORDS: tuple[tuple[str, float], ...] = (
    ("skin", 0.6),
    ("sensiti", 0.6),
    ("mutagen", 0.55),
    ("carcin", 0.55),
    ("acute", 0.45),
    ("repro", 0.6),
    ("development", 0.6),
    ("dartu", 0.5),
    ("neuro", 0.5),
    ("endocrine", 0.5),
    ("bioaccum", 0.45),
    ("pbt", 0.45),
    ("inhalation", 0.45),
    ("dermal", 0.4),
    ("aquatic", 0.4),
)

# Maximum number of models included in the recommended preset
RECOMMENDED_QSAR_LIMIT = 12


def _normalize_domain_result(value: Any) -> str:
    if not value:
        return "Unknown"
    return str(value).strip() or "Unknown"


def format_qsar_model_label(entry: Dict[str, Any]) -> str:
    """Generate a concise display label for a QSAR model entry."""
    caption = entry.get("Caption") or entry.get("Name") or entry.get("Guid") or "Unknown model"
    raw_position = entry.get("RequestedPosition") or entry.get("Position") or ""
    top_category = raw_position.split("#")[0].strip() if isinstance(raw_position, str) else ""
    donor = entry.get("Donator") or ""

    parts: List[str] = [caption.strip()]
    if top_category:
        parts.append(f"[{top_category}]")
    if donor:
        parts.append(f"({donor})")
    return " ".join(part for part in parts if part)


def derive_recommended_qsar_models(
    catalog: Sequence[Dict[str, Any]],
    limit: int = RECOMMENDED_QSAR_LIMIT,
) -> List[Dict[str, Any]]:
    """Select a fast-yet-diverse subset of QSAR models from the Toolbox catalog.

    The heuristic favours common regulatory endpoints and models donated by
    well-known fast providers while excluding known slow models (>10s runtime)
    and attempting to cover distinct top-level categories.
    """
    if not catalog:
        return []

    # Prefer the hand-curated publication preset when caller requests >=50 models
    if limit >= len(CURATED_QSAR_GUIDS):
        guid_to_entry: Dict[str, Dict[str, Any]] = {}
        for entry in catalog:
            guid = entry.get("Guid")
            if guid:
                guid_to_entry.setdefault(guid, entry)

        curated_entries: List[Dict[str, Any]] = []
        for guid in CURATED_QSAR_GUIDS:
            entry = guid_to_entry.get(guid)
            if entry:
                curated_entries.append(entry)

        if curated_entries:
            if len(curated_entries) >= limit:
                return curated_entries[:limit]

            # Need to top-up with heuristic selections for any missing slots
            remaining_catalog = [
                entry for entry in catalog if entry.get("Guid") not in {e.get("Guid") for e in curated_entries}
            ]
            top_up = derive_recommended_qsar_models(remaining_catalog, limit - len(curated_entries))
            return curated_entries + top_up

    scored: List[tuple[float, Dict[str, Any]]] = []
    seen_guids: set[str] = set()
    for entry in catalog:
        guid = entry.get("Guid")
        if not guid or guid in seen_guids:
            continue
        seen_guids.add(guid)
        
        # Skip known slow models
        if guid in SLOW_QSAR_MODEL_GUIDS:
            continue

        caption = (entry.get("Caption") or "").lower()
        top_category = (entry.get("RequestedPosition") or entry.get("Position") or "")
        top_category_lower = top_category.lower()
        donor = (entry.get("Donator") or "").lower()

        score = 5.0
        if any(keyword in donor for keyword in FAST_QSAR_DONOR_KEYWORDS):
            score -= 1.5

        for keyword, weight in QSAR_PRIORITY_KEYWORDS:
            if keyword in caption or keyword in top_category_lower:
                score -= weight

        scored.append((score, entry))

    # Sort by score, then caption to ensure deterministic ordering
    scored.sort(key=lambda item: (item[0], (item[1].get("Caption") or "").lower()))

    selected: List[Dict[str, Any]] = []
    used_categories: set[str] = set()

    # First pass: ensure coverage by picking distinct categories where possible
    for _, entry in scored:
        if len(selected) >= limit:
            break
        top_category = (entry.get("RequestedPosition") or entry.get("Position") or "")
        category_key = top_category.split("#")[0].strip().lower()
        if category_key and category_key in used_categories:
            continue
        selected.append(entry)
        used_categories.add(category_key)

    # Second pass: fill remaining slots regardless of category
    if len(selected) < limit:
        for _, entry in scored:
            if len(selected) >= limit:
                break
            if entry in selected:
                continue
            selected.append(entry)

    return selected


def run_qsar_predictions(
    api: QSARToolboxAPI,
    chem_id: str,
    *,
    selected_model_guids: Optional[Sequence[str]] = None,
    max_models: int | None = None,
    exclude_guids: Optional[Sequence[str]] = None,
    exclude_contains: Optional[Sequence[str]] = None,
    per_model_timeout_s: int | None = None,
    total_budget_s: int | None = None,
    logger=None,
) -> Dict[str, Any]:
    """Apply the chosen QSAR models to the target chemical.

    Args:
        api: Prepared QSARToolboxAPI instance (session.open() recommended before calling).
        chem_id: Toolbox chemical identifier.
        selected_model_guids: Optional explicit list of model GUIDs to execute (in order).
        max_models: Optional limit to cap the number of models executed.
        logger: Optional logger for progress updates.

    Returns:
        Dictionary containing raw predictions and summary counts.
    """
    catalog = api.get_all_qsar_models_catalog() or []

    exclude_guid_set = {str(g).lower() for g in (exclude_guids or []) if isinstance(g, str)}
    exclude_substrings = tuple(str(s).lower() for s in (exclude_contains or []) if isinstance(s, str))

    if exclude_guid_set or exclude_substrings:
        filtered_global: List[Dict[str, Any]] = []
        for entry in catalog:
            guid_lower = str(entry.get("Guid") or "").lower()
            if exclude_guid_set and guid_lower in exclude_guid_set:
                continue
            caption_lower = (entry.get("Caption") or "").lower()
            position_lower = (entry.get("RequestedPosition") or entry.get("Position") or "").lower()
            if exclude_substrings and any(sub in caption_lower or sub in position_lower for sub in exclude_substrings):
                continue
            filtered_global.append(entry)
        catalog = filtered_global

    total_discovered = len(catalog)

    # Build execution list
    if selected_model_guids:
        guid_map = {entry.get("Guid"): entry for entry in catalog if entry.get("Guid")}
        filtered_catalog: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for guid in selected_model_guids:
            if not guid:
                continue
            entry = None
            # Preserve case-insensitive matching to be defensive
            if guid in guid_map:
                entry = guid_map[guid]
            else:
                match = next(
                    (e for key, e in guid_map.items() if isinstance(key, str) and key.lower() == str(guid).lower()),
                    None,
                )
                entry = match
            if not entry:
                continue
            real_guid = entry.get("Guid")
            if real_guid in seen:
                continue
            seen.add(real_guid)
            filtered_catalog.append(entry)
    else:
        filtered_catalog = list(catalog)

    if max_models is not None:
        filtered_catalog = filtered_catalog[:max_models]

    predictions: List[Dict[str, Any]] = []

    total_models = len(filtered_catalog)
    executed = 0

    start_overall = time.perf_counter()
    for idx, entry in enumerate(filtered_catalog, start=1):
        guid = entry.get("Guid")
        caption = entry.get("Caption", "")
        position = entry.get("RequestedPosition", entry.get("Position", ""))
        donator = entry.get("Donator", "")
        top_category = position.split("#")[0] if isinstance(position, str) else ""

        if not guid:
            continue
        if max_models is not None and executed >= max_models:
            break

        if exclude_guid_set or exclude_substrings:
            guid_lower_loop = str(guid).lower() if guid else ""
            caption_lower_loop = caption.lower() if isinstance(caption, str) else ""
            position_lower_loop = position.lower() if isinstance(position, str) else ""
            if exclude_guid_set and guid_lower_loop in exclude_guid_set:
                continue
            if exclude_substrings and any(sub in caption_lower_loop or sub in position_lower_loop for sub in exclude_substrings):
                continue

        if logger:
            logger.info(
                "  [QSAR] %d/%d -> %s (%s)",
                idx,
                total_models,
                caption or guid,
                position or "Unknown position",
            )

        start = time.perf_counter()
        status = "ok"
        domain_result = "Unknown"
        domain_explain = None
        metadata = []
        value = ""
        unit = ""
        error_message = ""

        try:
            timeout_tuple = None
            if per_model_timeout_s and per_model_timeout_s > 0:
                timeout_tuple = (5, per_model_timeout_s)
            result = api.apply_qsar_model(guid, chem_id, timeout=timeout_tuple) or {}
            runtime = time.perf_counter() - start
            domain_result = _normalize_domain_result(result.get("DomainResult"))
            domain_explain = result.get("DomainExplain")
            metadata = result.get("MetaData") or []
            value = (result.get("Value") or "").strip()
            unit = (result.get("Unit") or "").strip()
        except (QSARTimeoutError, QSARResponseError, QSARConnectionError) as exc:
            runtime = time.perf_counter() - start
            status = "error"
            error_message = str(exc)
            metadata = []
            value = ""
            unit = ""
        except Exception as exc:  # pragma: no cover - defensive catch
            runtime = time.perf_counter() - start
            status = "error"
            error_message = str(exc)
            metadata = []
            value = ""
            unit = ""

        prediction = {
            "guid": guid,
            "caption": caption,
            "requested_position": position,
            "top_category": top_category,
            "donator": donator,
            "runtime_seconds": round(runtime, 4),
            "domain_result": domain_result,
            "domain_explain": domain_explain,
            "value": value,
            "unit": unit,
            "metadata": metadata,
            "status": status,
            "error": error_message,
        }
        predictions.append(prediction)

        executed += 1

        # Check total budget after each model
        if total_budget_s and total_budget_s > 0:
            elapsed_overall = time.perf_counter() - start_overall
            if elapsed_overall > total_budget_s:
                if logger:
                    logger.info(
                        "  [QSAR] Total time budget exceeded (%.1fs > %.1fs). Stopping further QSAR models.",
                        elapsed_overall,
                        total_budget_s,
                    )
                break

    summary = _summarize_predictions(predictions)
    return {
        "discovered_catalog_size": total_discovered,
        "catalog_size": total_models,
        "executed_models": executed,
        "predictions": predictions,
        "selected_model_guids": [entry.get("Guid") for entry in filtered_catalog if entry.get("Guid")],
        "summary": summary,
    }


def _summarize_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "total": len(predictions),
        "ok": 0,
        "out_of_domain": 0,
        "not_applicable": 0,
        "errors": 0,
        "in_domain": 0,
    }

    for pred in predictions:
        status = pred.get("status")
        if status == "error":
            counts["errors"] += 1
            continue

        domain = (pred.get("domain_result") or "").lower()
        if domain.startswith("in"):
            counts["ok"] += 1
            counts["in_domain"] += 1
        elif "not applicable" in domain or "not_applicable" in domain:
            counts["ok"] += 1
            counts["not_applicable"] += 1
        elif "out" in domain or "ambig" in domain:
            counts["out_of_domain"] += 1
        else:
            counts["ok"] += 1

    return counts
