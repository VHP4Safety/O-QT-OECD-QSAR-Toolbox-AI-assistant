# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

"""
Data formatting utilities for QSAR Toolbox responses
"""
import re
from typing import Dict, Any, List
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
                keys_to_remove = [k for k in error_record.keys() if k.lower() == "metadata"]
                for key in keys_to_remove:
                    error_record.pop(key, None)
                error_record.setdefault("Publication_Year", None)
                processed_data.append(error_record)
            except Exception:
                pass

    return processed_data


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
