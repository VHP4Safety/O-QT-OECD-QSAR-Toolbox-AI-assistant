# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

"""
Data formatting utilities for QSAR Toolbox responses
"""
import re
from typing import Dict, Any, List
import numpy as np

_DIGITS_RE = re.compile(r"^\d+$")   # helper: matches "0", "1", …

def _canonical_name(outer_key: str, rec: dict) -> str:
    """Return the best human-readable name for a calculator record."""
    cand = (
        rec.get("Parameter") or
        rec.get("Name")      or
        rec.get("CalculatorName") or
        ""
    ).strip()

    # Check if the outer_key itself is informative (not a digit string and not "unknown")
    is_outer_key_informative = not _DIGITS_RE.match(outer_key) and outer_key.lower() != "unknown"

    if is_outer_key_informative:
        # If outer_key is informative (e.g., "Boiling Point", "LogP" from test data), use it directly.
        # This ensures that tests expecting these specific keys will pass.
        return outer_key
    else:
        # If outer_key is not informative (e.g., "0", "1", "unknown"),
        # try to use the candidate name from the record's content.
        if cand:
            return cand
        else:
            # If no candidate name is found in the record, fall back to the original (non-informative) outer_key.
            return outer_key

def format_calculator_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a calculator result into a clean dictionary"""
    if not isinstance(result, dict):
        return {}
        
    return {
        "name": result.get("CalculatorName", "Unknown"),
        "type": result.get("CalculatorType", "Unknown"),
        "value": result.get("Calculation", {}).get("Value"),
        "unit": result.get("Calculation", {}).get("Unit", ""),
        "min_value": result.get("Calculation", {}).get("MinValue"),
        "max_value": result.get("Calculation", {}).get("MaxValue"),
        "family": result.get("Calculation", {}).get("Family", "")
    }

def format_profiler_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format a profiler result into a clean dictionary"""
    if not isinstance(result, dict):
        return {}
        
    return {
        "name": result.get("Caption", "Unknown"),
        "type": result.get("Type", "Unknown"),
        "id": result.get("Guid", "")
    }

def format_chemical_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Format chemical data into a clean dictionary"""
    if not isinstance(data, dict):
        return {}
        
    return {
        "type": data.get("SubstanceType", "Unknown"),
        "id": data.get("ChemId", ""),
        "cas": data.get("Cas", ""),
        "ec_number": data.get("ECNumber"),
        "smiles": data.get("Smiles", ""),
        "names": data.get("Names", []),
        "cas_smiles_relation": data.get("CasSmilesRelation", "")
    }

def clean_response_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and format the complete API response data"""
    cleaned = {
        "chemical_data": {},
        "properties": {},
        "experimental_data": [],
        "profiling": {}
    }
    
    # Clean chemical data
    if "chemical_data" in data and "basic_info" in data["chemical_data"]:
        cleaned["chemical_data"] = format_chemical_data(data["chemical_data"]["basic_info"])
    
    # Clean properties
    if "chemical_data" in data and "properties" in data["chemical_data"]:
        properties = {}
        for outer_key, rec in data["chemical_data"]["properties"].items():
            formatted = format_calculator_result(rec)
            if formatted:
                canonical = _canonical_name(outer_key, rec)
                properties[canonical] = {
                    "value": formatted["value"],
                    "unit": formatted["unit"],
                    "type": formatted["type"],
                    "family": formatted["family"],
                    "calculator_name": formatted["name"]
                }
        cleaned["properties"] = properties
    
    # Clean experimental data
    if "experimental_data" in data:
        cleaned["experimental_data"] = data["experimental_data"]
    
    # Clean profiling data
    if "profiling" in data:
        try:
            # Handle the newest structure with available_profilers
            if isinstance(data["profiling"], dict) and "available_profilers" in data["profiling"]:
                # Pass it through directly
                cleaned["profiling"] = data["profiling"]
            # Handle the structure with both profilers and results
            elif isinstance(data["profiling"], dict) and ("profilers" in data["profiling"] or "results" in data["profiling"]):
                cleaned_profiling = {}
                
                # Handle profilers
                if "profilers" in data["profiling"]:
                    profilers = {}
                    for key, prof in data["profiling"]["profilers"].items():
                        if isinstance(prof, dict):
                            formatted = format_profiler_result(prof)
                            if formatted:
                                profilers[formatted["name"]] = {
                                    "type": formatted["type"],
                                    "id": formatted["id"]
                                }
                    cleaned_profiling["profilers"] = profilers
                    
                # Handle chemical-specific profiling results
                if "results" in data["profiling"]:
                    # Just pass through the chemical-specific results as they may have varied structure
                    cleaned_profiling["results"] = data["profiling"]["results"]
                
                cleaned["profiling"] = cleaned_profiling
            else:
                # Legacy format handling
                profilers = {}
                for key, prof in data["profiling"].items():
                    if isinstance(prof, dict):
                        formatted = format_profiler_result(prof)
                        if formatted:
                            profilers[formatted["name"]] = {
                                "type": formatted["type"],
                                "id": formatted["id"]
                            }
                cleaned["profiling"] = profilers
        except Exception as e:
            # Fallback if anything goes wrong with profiling data
            print(f"Error cleaning profiling data: {e}")
            cleaned["profiling"] = {
                "status": "Error parsing profiling data",
                "error": str(e),
                "note": "The profiling data could not be processed due to an error"
            }
    
    return cleaned


# --- Safe JSON Serialization ---

# Removed @lru_cache decorator as it cannot handle unhashable dict inputs
def safe_json(obj):
    """Safely serialize Python objects to JSON, handling common non-serializable types."""
    import json, decimal, numpy as np
    def _coerce(o):
        if isinstance(o, (decimal.Decimal, np.generic)):
            return float(o)
        # Add other type checks here if needed (e.g., datetime, UUID)
        # Fallback to string representation
        try:
            # Attempt standard JSON serialization first
            json.dumps(o)
            return o
        except TypeError:
            return str(o) # Convert problematic types to string

    # Use a recursive approach or multiple passes if needed for deeply nested structures
    # For this case, a simple default handler might suffice
    try:
        return json.dumps(obj, default=_coerce, indent=2)
    except TypeError as e:
        print(f"Warning: Could not fully serialize object with safe_json: {e}")
        # Fallback to a basic string representation of the whole object if top-level fails
        return json.dumps(str(obj), indent=2)
