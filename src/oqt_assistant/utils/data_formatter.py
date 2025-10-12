# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""
Data formatting utilities for QSAR Toolbox responses
"""
from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple
import numpy as np
import logging
import json
# Imports for robust metadata parsing
import csv
import io
import ast  # Import ast for safe evaluation
from datetime import datetime

logger = logging.getLogger(__name__)

_DIGITS_RE = re.compile(r"^\d+$")  # helper: matches "0", "1", …
_YEAR_RE = re.compile(r"(19|20|21)\d{2}")  # simple 4-digit year finder (1900–2199)

def _canonical_name(outer_key: str, rec: dict) -> str:
    """Return the best human-readable name for a calculator record."""
    if not isinstance(rec, dict):
        return outer_key
    cand = (
        rec.get("Parameter")
        or rec.get("Name")
        or rec.get("CalculatorName")
        or ""
    ).strip()
    is_outer_key_informative = not _DIGITS_RE.match(outer_key) and outer_key.lower() != "unknown"
    if is_outer_key_informative:
        return outer_key
    return cand or outer_key


def parse_qsar_metadata_string(metadata_input: Any) -> Dict[str, str]:
    """Parses the QSAR Toolbox specific metadata format robustly using a hybrid approach."""
    parsed_metadata = {}
    metadata_list = []

    # 1) dict passthrough
    if isinstance(metadata_input, dict):
        return {str(k): str(v) for k, v in metadata_input.items()}

    # 2) list passthrough
    if isinstance(metadata_input, list):
        metadata_list = metadata_input

    # 3) string → try literal_eval then CSV fallbacks
    elif isinstance(metadata_input, str):
        content = metadata_input.strip()
        try:
            evaluated_input = ast.literal_eval(content)
            if isinstance(evaluated_input, list):
                metadata_list = evaluated_input
        except (ValueError, SyntaxError):
            pass

        if not metadata_list:
            csv_content = content
            if csv_content.startswith("[") and csv_content.endswith("]"):
                csv_content = csv_content[1:-1]
            try:
                for quote in ("'", '"'):
                    reader = csv.reader(io.StringIO(csv_content), quotechar=quote, skipinitialspace=True)
                    row = next(reader, [])
                    row = [item.strip() for item in row if item.strip()]
                    if row and any("=" in item for item in row):
                        metadata_list = row
                        break
            except csv.Error as e:
                logger.warning(f"CSV reader error during fallback: {e}. Input starts with: {content[:80]}")

    # 4) list of "key=value"
    for item in metadata_list:
        if isinstance(item, str):
            parts = item.strip().split("=", 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                if value.startswith("(") and value.endswith(")") and value.count("(") == 1 and value.count(")") == 1:
                    value = value[1:-1].strip()
                if key:
                    parsed_metadata[key] = value
    return parsed_metadata


def format_calculator_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a calculator result into a clean dictionary"""
    if not isinstance(result, dict):
        return {}
    calc_type_raw = result.get("CalculatorType", "Unknown")
    data_type_raw = result.get("DataType", "")
    if isinstance(calc_type_raw, str) and calc_type_raw.lower() == "measured" or "experimental" in str(data_type_raw).lower():
        calc_type = "Measured"
    elif isinstance(calc_type_raw, str) and calc_type_raw.lower() in ["qsar", "calculator", "unknown"]:
        calc_type = "QSAR"
    else:
        calc_type = calc_type_raw
    return {
        "name": result.get("CalculatorName", "Unknown"),
        "type": calc_type,
        "value": result.get("Calculation", {}).get("Value"),
        "unit": result.get("Calculation", {}).get("Unit", ""),
        "min_value": result.get("Calculation", {}).get("MinValue"),
        "max_value": result.get("Calculation", {}).get("MaxValue"),
        "family": result.get("Calculation", {}).get("Family", ""),
    }


def process_properties(raw_properties: Dict[str, Any] | List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Processes raw properties data (either list or indexed dictionary from QSAR API)
    into a clean, structured dictionary suitable for analysis and display.
    """
    processed_props = {}
    items_to_process = []
    if isinstance(raw_properties, dict):
        items_to_process = raw_properties.items()
    elif isinstance(raw_properties, list):
        items_to_process = enumerate(raw_properties)
    elif raw_properties is not None:
        logger.warning(f"Received unexpected properties data type: {type(raw_properties)}")
        return {}

    for outer_key, rec in items_to_process:
        outer_key = str(outer_key)
        if not isinstance(rec, dict):
            continue
        formatted = format_calculator_result(rec)
        if formatted and formatted.get("value") is not None:
            canonical = _canonical_name(outer_key, rec)
            processed_props[canonical] = {
                "value": formatted["value"],
                "unit": formatted["unit"],
                "type": formatted["type"],
                "family": formatted["family"],
                "calculator_name": formatted["name"],
                # NEW: carry-through of per-calculator provenance block when present
                "model_metadata": rec.get("ModelInfo")
            }
    return processed_props


def format_profiler_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a profiler result into a clean dictionary"""
    if not isinstance(result, dict):
        return {}
    return {
        "name": result.get("Caption", "Unknown"),
        "type": result.get("Type", "Unknown"),
        "id": result.get("Guid", ""),
    }


def format_chemical_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format chemical data into a clean dictionary"""
    if not isinstance(data, dict):
        return {}
    # Robust name resolution
    name = (data.get("Name")
            or data.get("IUPACName")
            or (data.get("Names")[0] if isinstance(data.get("Names"), list) and data.get("Names") else None)
            or data.get("name")
            or data.get("iupacname"))
    # Robust CAS; hyphenate if plain digits are supplied
    raw_cas = (data.get("Cas") or data.get("CAS") or data.get("CasNo") or data.get("CasNumber") or "")
    def _hyphenate_cas(s: str) -> str:
        digits = "".join(ch for ch in str(s) if ch.isdigit())
        if len(digits) >= 5:
            return f"{digits[:-3]}-{digits[-3:-1]}-{digits[-1]}"
        return str(s)
    cas_fmt = _hyphenate_cas(raw_cas) if raw_cas else ""
    return {
        "SubstanceType": data.get("SubstanceType", "Unknown"),
        "ChemId": data.get("ChemId", ""),
        "Cas": cas_fmt or raw_cas,
        "ECNumber": data.get("ECNumber"),
        "Smiles": data.get("Smiles", ""),
        "Names": data.get("Names", []),
        "Name": name,
        "IUPACName": data.get("IUPACName"),
        "CasSmilesRelation": data.get("CasSmilesRelation", ""),
    }


def _coerce_year_from_text(text: str) -> int | None:
    """Extract a 4‑digit year if present and plausible."""
    if not text:
        return None
    m = _YEAR_RE.search(str(text))
    if m:
        try:
            yr = int(m.group(0))
            # light plausibility filter
            if 1850 <= yr <= datetime.now().year + 1:
                return yr
        except Exception:
            return None
    return None


def _parse_metadata_items(items: List[str]) -> Dict[str, str]:
    """Normalize Toolbox metadata list ['Label: value', ...] into a lower-cased dict."""
    out = {}
    for raw in items or []:
        if not isinstance(raw, str):
            continue
        if ":" in raw:
            k, v = raw.split(":", 1)
            out[k.strip().lower()] = v.strip()
        else:
            out[raw.strip().lower()] = ""
    return out


def process_experimental_metadata(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Processes the Metadata field in experimental data records, parsing it into a structured dictionary.
    Adds `Publication_Year` when possible for sorting."""
    if not data:
        return []

    processed_data = []

    for record in data:
        try:
            if not isinstance(record, dict):
                logger.warning(f"Skipping experimental record because it is not a dictionary: {type(record)}")
                processed_data.append({"Processing_Error": f"Record was not a dictionary: {type(record)}", "Raw_Value": str(record), "Parsed_Metadata": {}})
                continue

            new_record = record.copy()

            # Case-insensitive 'Metadata' removal
            metadata_key = next((k for k in list(new_record.keys()) if k.lower() == "metadata"), None)
            new_record["Parsed_Metadata"] = {}
            raw_metadata = new_record.pop(metadata_key, None) if metadata_key else None
            raw_metadata_list = raw_metadata if isinstance(raw_metadata, list) else []

            if raw_metadata:
                try:
                    parsed_data = parse_qsar_metadata_string(raw_metadata)
                except Exception as parse_e:
                    logger.error(f"Unexpected error during metadata parsing: {parse_e}")
                    parsed_data = {}
                if parsed_data:
                    new_record["Parsed_Metadata"] = parsed_data
                elif str(raw_metadata).strip() not in ["[]", "{}", ""]:
                    new_record["Parsed_Metadata"] = {
                        "Parsing_Note": "Could not automatically parse metadata structure.",
                        "Raw_Value": str(raw_metadata),
                    }

            # Normalize metadata to dict for ease of access
            meta_dict = _parse_metadata_items(raw_metadata_list)
            new_record["MetaDict"] = meta_dict
            
            # NEW: Convenience fields for QC
            new_record["Reliability"] = (
                meta_dict.get("reliability") or
                meta_dict.get("reliability (klimisch)") or
                new_record["Parsed_Metadata"].get("Reliability") or
                ""
            )
            new_record["AdequacyOfStudy"] = (
                meta_dict.get("adequacy of study") or
                meta_dict.get("adequacy") or
                ""
            )

            # --- Publication_Year extraction (robust) ---
            year_val = None

            # 1) explicit keys inside parsed metadata
            for k in ("Year", "PublicationYear", "StudyYear", "Ref Year", "RefYear"):
                if isinstance(new_record["Parsed_Metadata"], dict) and k in new_record["Parsed_Metadata"]:
                    year_val = _coerce_year_from_text(new_record["Parsed_Metadata"][k])
                    if year_val:
                        break

            # 2) free text fields commonly containing a year
            if not year_val:
                for k in ("Reference", "Notes", "Citation", "Source"):
                    year_val = _coerce_year_from_text(new_record.get(k))
                    if year_val:
                        break

            new_record["Publication_Year"] = year_val

            processed_data.append(new_record)

        except Exception as e:
            logger.error(f"Error processing experimental record: {e}. Adding error indicator and continuing.")
            try:
                error_record = record.copy() if isinstance(record, dict) else {}
                error_record["Processing_Error"] = str(e)
                error_record.setdefault("Parsed_Metadata", {})
                error_record.setdefault("MetaDict", {})
                error_record.setdefault("Reliability", "")
                error_record.setdefault("AdequacyOfStudy", "")
                keys_to_remove = [k for k in error_record.keys() if k.lower() == "metadata"]
                for key in keys_to_remove:
                    error_record.pop(key, None)
                error_record.setdefault("Publication_Year", None)
                processed_data.append(error_record)
            except Exception:
                pass

    return processed_data


def process_qsar_predictions(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Categorize QSAR predictions by applicability domain and prepare for reporting."""
    if not predictions:
        return {
            "all": [],
            "in_domain": [],
            "not_applicable": [],
            "out_of_domain": [],
            "errors": [],
            "summary": {"total": 0, "in_domain": 0, "not_applicable": 0, "out_of_domain": 0, "errors": 0},
        }

    all_records = []
    in_domain = []
    not_applicable = []
    out_of_domain = []
    errors = []

    for entry in predictions:
        domain = (entry.get("domain_result") or "").strip()
        status = entry.get("status", "ok")
        normalized = {
            "caption": entry.get("caption", ""),
            "guid": entry.get("guid"),
            "top_category": entry.get("top_category", ""),
            "requested_position": entry.get("requested_position", ""),
            "donator": entry.get("donator", ""),
            "runtime_seconds": entry.get("runtime_seconds", 0.0),
            "domain_result": domain or "Unknown",
            "domain_explain": entry.get("domain_explain"),
            "value": entry.get("value", ""),
            "unit": entry.get("unit", ""),
            "metadata": entry.get("metadata", []),
            "status": status,
            "error": entry.get("error", ""),
        }

        all_records.append(normalized)

        if status == "error":
            errors.append(normalized)
            continue

        domain_lower = (domain or "").lower()
        if domain_lower.startswith("in"):
            in_domain.append(normalized)
        elif "not applicable" in domain_lower or "not_applicable" in domain_lower:
            not_applicable.append(normalized)
        elif "out" in domain_lower or "ambig" in domain_lower:
            out_of_domain.append(normalized)
        else:
            # Treat unknown domain as out-of-domain to be conservative
            out_of_domain.append(normalized)

    summary = {
        "total": len(all_records),
        "in_domain": len(in_domain),
        "not_applicable": len(not_applicable),
        "out_of_domain": len(out_of_domain),
        "errors": len(errors),
    }

    return {
        "all": all_records,
        "in_domain": in_domain,
        "not_applicable": not_applicable,
        "out_of_domain": out_of_domain,
        "errors": errors,
        "summary": summary,
    }


def clean_response_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and format the complete API response data.
    This function is primarily used by the legacy/wizard workflow,
    as the main app.py now handles processing progressively.
    """
    cleaned = {
        "chemical_data": {"basic_info": {}, "properties": {}},
        "experimental_data": [],
        "profiling": {},
        "metabolism": {},
    }

    # Chemical Data
    if "chemical_data" in data and isinstance(data.get("chemical_data"), dict) and "basic_info" in data["chemical_data"]:
        basic_info_raw = data["chemical_data"]["basic_info"]
    elif "ChemId" in data:
        basic_info_raw = data
    else:
        basic_info_raw = {}
    cleaned["chemical_data"]["basic_info"] = format_chemical_data(basic_info_raw)

    # Properties
    if "chemical_data" in data and isinstance(data.get("chemical_data"), dict) and "properties" in data["chemical_data"]:
        properties_raw = data["chemical_data"]["properties"]
    elif "properties" in data:
        properties_raw = data["properties"]
    else:
        properties_raw = {}
    try:
        cleaned["chemical_data"]["properties"] = process_properties(properties_raw)
    except Exception as e:
        logger.error(f"Error during properties processing in clean_response_data: {e}")
        cleaned["chemical_data"]["properties"] = {}

    # Experimental Data
    if "experimental_data" in data and data["experimental_data"]:
        try:
            cleaned["experimental_data"] = process_experimental_metadata(data["experimental_data"])
        except Exception as e:
            logger.error(f"Error during experimental metadata processing in clean_response_data: {e}")
            cleaned["experimental_data"] = [{"Error": str(e), "Parsed_Metadata": {}} for _ in data["experimental_data"]]

    # Metabolism
    if "metabolism" in data and isinstance(data["metabolism"], dict):
        cleaned["metabolism"] = data["metabolism"]

    # Profiling
    if "profiling" in data:
        if isinstance(data["profiling"], dict):
            cleaned["profiling"] = data["profiling"]
        elif isinstance(data["profiling"], list):
            cleaned["profiling"] = {"available_profilers": data["profiling"], "status": "List format detected"}
        else:
            cleaned["profiling"] = {"status": "Error", "note": f"Unexpected profiling data type: {type(data['profiling'])}"}

    return cleaned


def safe_json(obj):
    """Safely serialize Python objects to JSON, handling common non-serializable types."""
    import json as _json, decimal
    def _coerce(o):
        if isinstance(o, (decimal.Decimal, np.generic)):
            return float(o)
        try:
            _json.dumps(o)
            return o
        except TypeError:
            return str(o)
    try:
        return _json.dumps(obj, default=_coerce, indent=2)
    except TypeError as e:
        print(f"Warning: Could not fully serialize object with safe_json: {e}")
        return _json.dumps(str(obj), indent=2)


# ========== QPRF/RAAF Formatting Functions ==========

def format_qprf_substance_identity(chemical_data: Dict[str, Any], iuclid_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Format substance identity for QPRF §2.1-2.6
    
    Args:
        chemical_data: Basic chemical information
        iuclid_data: IUCLID enrichment data (optional)
    
    Returns:
        Formatted substance identity block
    """
    identity = {
        "cas": chemical_data.get("Cas", ""),
        "ec_number": chemical_data.get("ECNumber", ""),
        "chemical_name": chemical_data.get("Name", ""),
        "iupac_name": chemical_data.get("IUPACName", ""),
        "smiles": chemical_data.get("Smiles", ""),
        "canonical_smiles": chemical_data.get("canonical_smiles", ""),
        "connectivity": chemical_data.get("connectivity", ""),
        "stereochemistry_note": chemical_data.get("stereochemistry_note", ""),
        "regulatory_ids": []
    }
    
    # Add IUCLID IDs if available
    if iuclid_data and iuclid_data.get("entity_ids"):
        for entity in iuclid_data["entity_ids"]:
            identity["regulatory_ids"].append({
                "type": "IUCLID",
                "id": entity.get("entity_id"),
                "name": entity.get("name"),
                "url": iuclid_data.get("echa_url")
            })
    
    return identity


def format_qprf_model_info(model_data: Dict[str, Any], model_type: str = "Calculator") -> Dict[str, Any]:
    """Format model information for QPRF §3.1-3.2
    
    Args:
        model_data: Model metadata from API
        model_type: Type of model (Calculator, QSAR, Profiler, Simulator)
    
    Returns:
        Formatted model information block
    """
    qprf_metadata = model_data.get("qprf_metadata", {})
    
    model_info = {
        "model_type": model_type,
        "model_name": qprf_metadata.get("name", model_data.get("name", "Unknown")),
        "model_version": qprf_metadata.get("additional_info", ""),
        "description": qprf_metadata.get("description", ""),
        "developer": qprf_metadata.get("donator", ""),
        "authors": qprf_metadata.get("authors", ""),
        "reference_url": qprf_metadata.get("url", ""),
        "disclaimer": qprf_metadata.get("disclaimer", ""),
        "qmrf_reference": None  # To be filled if QSAR model
    }
    
    return model_info


def format_qprf_prediction(prediction_data: Dict[str, Any], endpoint_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Format prediction for QPRF §4.1-4.2
    
    Args:
        prediction_data: Prediction result
        endpoint_info: Endpoint metadata (optional)
    
    Returns:
        Formatted prediction block
    """
    prediction = {
        "endpoint": endpoint_info.get("endpoint_name", "") if endpoint_info else "",
        "endpoint_path": endpoint_info.get("rigid_path", "") if endpoint_info else "",
        "predicted_value": prediction_data.get("value"),
        "qualifier": prediction_data.get("qualifier", ""),
        "unit": prediction_data.get("unit", ""),
        "family": prediction_data.get("family", ""),
        "min_value": prediction_data.get("min_value"),
        "max_value": prediction_data.get("max_value")
    }
    
    return prediction


def format_qprf_applicability_domain(domain_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format applicability domain for QPRF §6.1
    
    Args:
        domain_data: Domain assessment from QSAR model
    
    Returns:
        Formatted AD block
    """
    ad_info = {
        "result": domain_data.get("result", "Unknown"),
        "explanation": domain_data.get("explain", []),
        "method": "Model-specific applicability domain assessment",
        "limitations": []
    }
    
    # Add interpretation
    if ad_info["result"] == "In":
        ad_info["interpretation"] = "The query chemical is within the applicability domain of the model."
    elif ad_info["result"] == "Out":
        ad_info["interpretation"] = "The query chemical is outside the applicability domain of the model. Prediction may be unreliable."
    else:
        ad_info["interpretation"] = "Applicability domain assessment is ambiguous. Use prediction with caution."
    
    return ad_info


def format_qprf_mechanistic_info(profiling_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Format mechanistic information for QPRF §7.3
    
    Args:
        profiling_data: Profiling results with metadata
    
    Returns:
        List of mechanistic information blocks
    """
    mechanistic_info = []
    
    if not profiling_data or "results" not in profiling_data:
        return mechanistic_info
    
    for profiler_name, profiler_result in profiling_data.get("results", {}).items():
        qprf_meta = profiler_result.get("qprf_metadata", {})
        
        # Extract triggered categories
        categories = profiler_result.get("result", [])
        if not isinstance(categories, list):
            categories = [categories] if categories else []
        
        # Extract literature
        literature = qprf_meta.get("literature", {})
        literature_refs = []
        for category, refs in literature.items():
            if isinstance(refs, list):
                literature_refs.extend(refs)
        
        mech_block = {
            "profiler_name": profiler_name,
            "profiler_type": profiler_result.get("type", ""),
            "description": qprf_meta.get("description", ""),
            "developer": qprf_meta.get("donator", ""),
            "triggered_categories": categories,
            "literature_references": literature_refs[:10],  # Limit to 10 refs
            "interpretation": f"Chemical triggers {len(categories)} alert(s) in {profiler_name}"
        }
        
        mechanistic_info.append(mech_block)
    
    return mechanistic_info


def format_qprf_metabolic_info(metabolism_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format metabolic information for QPRF §7.3.e
    
    Args:
        metabolism_data: Metabolism results with simulator metadata
    
    Returns:
        Formatted metabolic information block
    """
    if not metabolism_data:
        return {}
    
    metabolic_info = {
        "simulators_used": [],
        "metabolites_identified": len(metabolism_data.get("metabolites", [])),
        "metabolite_details": []
    }
    
    # Add simulator information
    for simulator in metabolism_data.get("simulators", []):
        metabolic_info["simulators_used"].append({
            "name": simulator.get("name", "Unknown"),
            "description": simulator.get("description", ""),
            "developer": simulator.get("donator", "")
        })
    
    # Add metabolite details
    for metabolite in metabolism_data.get("metabolites", [])[:20]:  # Limit to 20
        metabolic_info["metabolite_details"].append({
            "smiles": metabolite.get("smiles", ""),
            "chem_id": metabolite.get("chemId", ""),
            "generation": metabolite.get("generation", 1)
        })
    
    metabolic_info["interpretation"] = (
        f"Metabolism simulation identified {metabolic_info['metabolites_identified']} "
        f"potential metabolite(s) using {len(metabolic_info['simulators_used'])} simulator(s)."
    )
    
    return metabolic_info


def format_qprf_provenance(databases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format database provenance information
    
    Args:
        databases: List of database metadata
    
    Returns:
        Formatted provenance blocks
    """
    provenance = []
    
    for db in databases:
        provenance.append({
            "database_name": db.get("Caption", "Unknown"),
            "source_id": db.get("SourceId"),
            "url_base": db.get("UrlBase", ""),
            "guid": db.get("Guid", "")
        })
    
    return provenance


def create_qprf_report_structure(qprf_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a complete QPRF report structure from enriched data
    
    Args:
        qprf_data: Enriched QPRF data from QPRFEnricher
    
    Returns:
        Complete QPRF report structure following QPRF v2.0 format
    """
    report = {
        "qprf_version": "2.0",
        "report_date": qprf_data.get("report_metadata", {}).get("date", datetime.now().isoformat()),
        
        # Section 1: QPRF Information
        "section_1": {
            "date": qprf_data.get("report_metadata", {}).get("date"),
            "author": "OECD QSAR Toolbox AI Assistant"
        },
        
        # Section 2: Substance
        "section_2": format_qprf_substance_identity(
            qprf_data.get("substance_identity", {}),
            qprf_data.get("substance_identity", {}).get("iuclid")
        ),
        
        # Section 3: Model/Software (to be filled per prediction)
        "section_3": {
            "software": qprf_data.get("report_metadata", {}).get("software", {})
        },
        
        # Section 4: Prediction (to be filled per prediction)
        "section_4": {},
        
        # Section 5: Input (covered in section 2)
        "section_5": {
            "input_structure": qprf_data.get("substance_identity", {}).get("smiles"),
            "canonical_smiles": qprf_data.get("substance_identity", {}).get("canonical_smiles"),
            "connectivity": qprf_data.get("substance_identity", {}).get("connectivity")
        },
        
        # Section 6: Applicability Domain (to be filled per prediction)
        "section_6": {},
        
        # Section 7: Adequacy
        "section_7": {
            "mechanistic_information": format_qprf_mechanistic_info(qprf_data.get("profiling", {})),
            "metabolic_information": format_qprf_metabolic_info(qprf_data.get("metabolism", {}))
        },
        
        # Section 8: Regulatory Purpose (to be filled by user)
        "section_8": {
            "regulatory_purpose": "To be specified by user",
            "approach_for_regulatory_interpretation": "To be specified by user"
        }
    }
    
    return report


def format_raaf_checklist(qprf_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create RAAF assessment checklist from QPRF data
    
    Args:
        qprf_data: Enriched QPRF data
    
    Returns:
        RAAF checklist structure
    """
    substance = qprf_data.get("substance_identity", {})
    iuclid = substance.get("iuclid", {})
    profiling = qprf_data.get("profiling", {})
    metabolism = qprf_data.get("metabolism", {})
    
    checklist = {
        "identity_complete": {
            "cas_present": bool(substance.get("Cas")),
            "ec_present": bool(substance.get("ECNumber")),
            "smiles_present": bool(substance.get("Smiles")),
            "iuclid_id_present": bool(iuclid.get("entity_ids")),
            "status": "Complete" if all([
                substance.get("Cas"),
                substance.get("Smiles")
            ]) else "Incomplete"
        },
        
        "hypothesis_type": {
            "common_compound": bool(metabolism and metabolism.get("metabolites")),
            "common_mechanism": bool(profiling and profiling.get("results")),
            "suggested_scenario": "To be determined based on profiling and metabolism results"
        },
        
        "mechanistic_basis": {
            "profilers_applied": len(profiling.get("results", {})) if profiling else 0,
            "categories_triggered": sum(
                len(p.get("result", [])) for p in profiling.get("results", {}).values()
            ) if profiling else 0,
            "literature_available": any(
                p.get("qprf_metadata", {}).get("literature")
                for p in profiling.get("results", {}).values()
            ) if profiling else False
        },
        
        "metabolic_coverage": {
            "simulators_used": len(metabolism.get("simulators", [])) if metabolism else 0,
            "metabolites_identified": len(metabolism.get("metabolites", [])) if metabolism else 0,
            "coverage_adequate": bool(metabolism and metabolism.get("metabolites"))
        },
        
        "overall_assessment": "Preliminary - requires expert review"
    }
    
    return checklist
