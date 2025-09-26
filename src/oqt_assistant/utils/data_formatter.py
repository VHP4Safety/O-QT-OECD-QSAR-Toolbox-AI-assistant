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
import ast # Import ast for safe evaluation

logger = logging.getLogger(__name__)

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
        return outer_key
    else:
        if cand:
            return cand
        else:
            return outer_key

# UPDATED HELPER FUNCTION (Hybrid approach: ast.literal_eval prioritized, CSV fallback)
def parse_qsar_metadata_string(metadata_input: Any) -> Dict[str, str]:
    """Parses the QSAR Toolbox specific metadata format robustly using a hybrid approach."""
    parsed_metadata = {}
    metadata_list = []

    # 1. Handle non-string/list types first
    if isinstance(metadata_input, dict):
        # If it's already a dict, return it directly
        return {str(k): str(v) for k, v in metadata_input.items()}
    
    if isinstance(metadata_input, list):
        metadata_list = metadata_input

    # 2. Handle string input (most complex case)
    elif isinstance(metadata_input, str):
        content = metadata_input.strip()
        
        # Strategy A: Try ast.literal_eval (best for Python list representations like the user's example)
        try:
            # This is the key addition to handle the exact format reported by the user
            evaluated_input = ast.literal_eval(content)
            if isinstance(evaluated_input, list):
                metadata_list = evaluated_input
        except (ValueError, SyntaxError):
            # If ast.literal_eval fails, proceed to Strategy B
            pass # Continue to Strategy B
            
        # Strategy B: Try CSV reader fallback (robust against commas within values)
        if not metadata_list: # Only try CSV if ast didn't populate the list
            
            # Prepare content for CSV reader if it looks like a list
            csv_content = content
            if csv_content.startswith("[") and csv_content.endswith("]"):
                csv_content = csv_content[1:-1]

            try:
                # Try single quotes
                reader = csv.reader(io.StringIO(csv_content), quotechar="'", skipinitialspace=True)
                temp_list = []
                found_row = False
                for row in reader:
                    temp_list = [item.strip() for item in row if item.strip()]
                    found_row = True
                    break 
                
                if found_row and temp_list and any('=' in item for item in temp_list):
                    metadata_list = temp_list
                else:
                    # Try double quotes
                    reader = csv.reader(io.StringIO(csv_content), quotechar='"', skipinitialspace=True)
                    for row in reader:
                        temp_list = [item.strip() for item in row if item.strip()]
                        break
                    
                    if temp_list and any('=' in item for item in temp_list):
                        metadata_list = temp_list
                    else:
                         # If CSV fails too, we cannot reliably parse it.
                        if csv_content.strip():
                            logger.debug(f"Both ast.literal_eval and CSV parser failed for metadata string: {metadata_input[:100]}...")
                        return {}

            except csv.Error as e:
                logger.warning(f"CSV reader error during fallback: {e}. Input: {metadata_input[:100]}...")
                return {}

    # 3. Process the resulting list (Key=Value pairs)
    for item in metadata_list:
        if isinstance(item, str):
            item = item.strip()
            # Find the first '=' sign
            parts = item.split('=', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                
                # Clean up potential surrounding parentheses (e.g., (1985))
                # Only strip if they enclose the whole value and are balanced (simple check)
                if value.startswith("(") and value.endswith(")"):
                     if value.count('(') == 1 and value.count(')') == 1:
                         value = value[1:-1].strip()
                
                if key:
                    parsed_metadata[key] = value
    
    return parsed_metadata


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

# UPDATED FUNCTION (Robust error handling and case-insensitive removal)
def process_experimental_metadata(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Processes the Metadata field in experimental data records, parsing it into a structured dictionary."""
    if not data:
        return []

    processed_data = []

    for record in data:
        try:
            # Ensure record is a dictionary before proceeding
            if not isinstance(record, dict):
                logger.warning(f"Skipping experimental record because it is not a dictionary: {type(record)}")
                # Add a placeholder record indicating the issue
                processed_data.append({"Processing_Error": f"Record was not a dictionary: {type(record)}", "Raw_Value": str(record), "Parsed_Metadata": {}})
                continue

            # Create a copy to modify
            new_record = record.copy()
            
            # Handle 'Metadata' case-insensitively
            metadata_key = None
            # Iterate over keys to find the metadata field regardless of case
            for key in list(new_record.keys()): 
                if key.lower() == 'metadata':
                    metadata_key = key
                    break
            
            # Initialize the new structured field
            new_record['Parsed_Metadata'] = {}
            raw_metadata = None

            # Safely extract and remove the raw 'Metadata' field using the found key
            # This guarantees 'Metadata' (in any capitalization) is removed from the record.
            if metadata_key:
                raw_metadata = new_record.pop(metadata_key, None) 

            if raw_metadata:
                # Parse the metadata using the dedicated robust parser
                try:
                    parsed_data = parse_qsar_metadata_string(raw_metadata)
                except Exception as parse_e:
                    logger.error(f"Unexpected error during metadata parsing: {parse_e}")
                    parsed_data = {}

                if parsed_data:
                    new_record['Parsed_Metadata'] = parsed_data
                else:
                    # If parsing fails, store the raw metadata for inspection in the structured field
                    # We check if the raw metadata was not just an empty representation.
                    if str(raw_metadata).strip() not in ["[]", "{}", ""]:
                         # Store raw value in Parsed_Metadata dict for visibility in the UI viewer
                         new_record['Parsed_Metadata'] = {"Parsing_Note": "Could not automatically parse metadata structure.", "Raw_Value": str(raw_metadata)}
                    
            processed_data.append(new_record)
        
        except Exception as e:
            # Catch exceptions during the processing of a single record
            logger.error(f"Error processing experimental record: {e}. Adding error indicator and continuing.")
            # Add a modified version of the record indicating an error
            try:
                error_record = record.copy() if isinstance(record, dict) else {}
                error_record['Processing_Error'] = str(e)
                # Ensure Parsed_Metadata exists even in error state
                if 'Parsed_Metadata' not in error_record:
                    error_record['Parsed_Metadata'] = {}
                
                # Ensure Metadata is removed even in error state (case-insensitive cleanup)
                keys_to_remove = [k for k in error_record.keys() if k.lower() == 'metadata']
                for key in keys_to_remove:
                    error_record.pop(key, None)

                processed_data.append(error_record)
            except Exception:
                # If even creating an error record fails, just skip it.
                pass

    return processed_data


# UPDATED FUNCTION (Simplified fallback logic)
def clean_response_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and format the complete API response data"""
    cleaned = {
        "chemical_data": {},
        "properties": {},
        "experimental_data": [],
        "profiling": {},
        "metabolism": {}
    }
    
    # Clean chemical data
    if "chemical_data" in data and data.get("chemical_data") and "basic_info" in data["chemical_data"]:
        cleaned["chemical_data"] = format_chemical_data(data["chemical_data"]["basic_info"])
    
    # Clean properties
    if "chemical_data" in data and data.get("chemical_data") and "properties" in data["chemical_data"]:
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
    
    # UPDATED: Clean experimental data using the robust processing function
    if "experimental_data" in data and data["experimental_data"]:
        try:
            # The processing function now handles internal errors and ensures a consistent output structure.
            cleaned["experimental_data"] = process_experimental_metadata(data["experimental_data"])
        except Exception as e:
            logger.error(f"CRITICAL Error during experimental metadata processing: {e}. Falling back to safe data structure.")
            
            # Critical Fallback: If the processing function itself crashes (highly unlikely now), we must provide a safe fallback.
            # We ensure the structure expected by the frontend ('Parsed_Metadata' exists, 'Metadata' removed) exists.
            fallback_data = []
            try:
                for record in data["experimental_data"]:
                    if isinstance(record, dict):
                        record_copy = record.copy()
                        
                        # Handle 'Metadata' field in fallback: move it to Parsed_Metadata['Raw_Value'] and remove it (case-insensitive)
                        metadata_key = None
                        for key in list(record_copy.keys()):
                            if key.lower() == 'metadata':
                                metadata_key = key
                                break
                        
                        raw_meta = None
                        if metadata_key:
                             raw_meta = record_copy.pop(metadata_key, None)

                        record_copy['Parsed_Metadata'] = {"Error": f"Fallback processing used (Processing crashed: {str(e)})", "Raw_Value": str(raw_meta)}
                        fallback_data.append(record_copy)
                    else:
                         fallback_data.append({"Error": "Record is not a dictionary (Fallback)", "Raw_Value": str(record), "Parsed_Metadata": {}})

                cleaned["experimental_data"] = fallback_data
            except Exception as fallback_e:
                 logger.error(f"Fallback mechanism failed: {fallback_e}. Returning empty list.")
                 cleaned["experimental_data"] = [] # Absolute fallback

    
    # Clean metabolism data
    if "metabolism" in data and isinstance(data["metabolism"], dict):
        cleaned["metabolism"] = data["metabolism"]

    
    # Clean profiling data (Keep existing logic)
    if "profiling" in data:
        try:
            if isinstance(data["profiling"], dict) and ("results" in data["profiling"] or "available_profilers" in data["profiling"]):
                cleaned["profiling"] = data["profiling"]
            elif isinstance(data["profiling"], dict):
                logger.warning("Profiling data is not in the standardized format. Attempting legacy parsing.")
                cleaned["profiling"] = data["profiling"] 
            elif isinstance(data["profiling"], list):
                cleaned["profiling"] = {"available_profilers": data["profiling"], "status": "List format detected"}
        
        except Exception as e:
            print(f"Error cleaning profiling data: {e}")
            cleaned["profiling"] = {
                "status": "Error parsing profiling data",
                "error": str(e),
                "note": "The profiling data could not be processed due to an error"
            }
    
    return cleaned


# --- Safe JSON Serialization ---
# (Keep existing logic)
def safe_json(obj):
    """Safely serialize Python objects to JSON, handling common non-serializable types."""
    import json, decimal, numpy as np
    def _coerce(o):
        if isinstance(o, (decimal.Decimal, np.generic)):
            return float(o)
        try:
            json.dumps(o)
            return o
        except TypeError:
            return str(o)

    try:
        return json.dumps(obj, default=_coerce, indent=2)
    except TypeError as e:
        print(f"Warning: Could not fully serialize object with safe_json: {e}")
        return json.dumps(str(obj), indent=2)