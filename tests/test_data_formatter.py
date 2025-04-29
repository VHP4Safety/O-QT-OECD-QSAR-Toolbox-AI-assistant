# tests/test_data_formatter.py
import pytest
# Import the module itself for cache clearing
from streamlined_qsar_app.utils import data_formatter
# Assuming utils is importable from tests directory. Adjust if necessary.
from streamlined_qsar_app.utils.data_formatter import format_calculator_result, format_chemical_data, clean_response_data, safe_json # [cite: 74, 76, 77, 91]

# Removed cache clearing fixture as safe_json is no longer cached

def test_format_calculator_result_basic():
    """Test basic formatting of calculator result""" # [cite: 74]
    raw_data = {
        "CalculatorName": "Test Calc",
        "CalculatorType": "PhysChem",
        "Calculation": {"Value": 10.5, "Unit": "°C", "Family": "Boiling Point"}
    }
    expected = {
        "name": "Test Calc", "type": "PhysChem", "value": 10.5,
        "unit": "°C", "min_value": None, "max_value": None, "family": "Boiling Point"
    }
    assert format_calculator_result(raw_data) == expected # [cite: 74]

def test_format_calculator_result_missing_data():
    """Test formatting with missing data""" # [cite: 74]
    raw_data = {"CalculatorName": "Partial Calc"}
    expected = {
        "name": "Partial Calc", "type": "Unknown", "value": None,
        "unit": "", "min_value": None, "max_value": None, "family": ""
    }
    assert format_calculator_result(raw_data) == expected # [cite: 74]

def test_format_chemical_data_full():
    """Test formatting full chemical data""" # [cite: 76]
    raw_data = {
        "SubstanceType": "MonoConstituent", "ChemId": "chem456", "Cas": "123-45-6",
        "ECNumber": "200-000-0", "Smiles": "CCO", "Names": ["Ethanol", "Ethyl alcohol"]
    }
    expected = {
        "type": "MonoConstituent", "id": "chem456", "cas": "123-45-6",
        "ec_number": "200-000-0", "smiles": "CCO", "names": ["Ethanol", "Ethyl alcohol"],
        "cas_smiles_relation": ""
    }
    assert format_chemical_data(raw_data) == expected # [cite: 76]

def test_clean_response_data_structure():
    """Test the overall structure of cleaned data""" # [cite: 77]
    raw_api_response = { # Simulate structure from app.py [cite: 43]
        'chemical_data': {
            'basic_info': {'ChemId': 'chem789', 'Names': ['TestChem']}, # [cite: 76]
            'properties': {'Boiling Point': {'Calculation': {'Value': 100}}} # [cite: 78]
        },
        'experimental_data': [{'Endpoint': 'LC50', 'Value': 50}], # [cite: 80]
        'profiling': {'available_profilers': [{'name': 'DNA Binder'}]} # [cite: 81, 280]
    }
    cleaned = clean_response_data(raw_api_response) # [cite: 77]
    assert "chemical_data" in cleaned
    assert "properties" in cleaned
    assert "experimental_data" in cleaned
    assert "profiling" in cleaned
    assert cleaned["chemical_data"]["id"] == "chem789" # [cite: 76]
    assert "Boiling Point" in cleaned["properties"] # [cite: 78]
    assert len(cleaned["experimental_data"]) == 1 # [cite: 80]
    assert "available_profilers" in cleaned["profiling"] # [cite: 81, 280]

def test_safe_json_numpy():
    """Test safe_json with numpy types""" # [cite: 91]
    # Need to import numpy if it's used in the function being tested
    try:
        import numpy as np
        data = {"value": np.float64(10.5)}
        # Expect valid JSON string with float
        assert '"value": 10.5' in safe_json(data) # [cite: 91]
    except ImportError:
        pytest.skip("Numpy not installed, skipping numpy test")


# --- Add more tests for: ---
# - Edge cases in formatters (empty dicts, lists, None values)
# - Different profiling data structures handled by clean_response_data [cite: 81, 82, 86, 90]
# - Other types handled by safe_json if you add them (Decimal, datetime) [cite: 91]
