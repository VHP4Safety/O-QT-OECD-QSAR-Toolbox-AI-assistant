# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: MIT

# tests/test_qsar_api.py
import pytest
import requests # Need to import requests for the exception
from unittest.mock import patch, MagicMock
# Import the module itself to access its functions for cache clearing
from qsar_assistant.utils import qsar_api
from qsar_assistant.utils.qsar_api import QSARToolboxAPI, QSARConnectionError, QSARTimeoutError, QSARResponseError, SearchOptions # [cite: 94, 95]

# --- Fixture to clear caches before each test ---
@pytest.fixture(autouse=True)
def clear_qsar_caches():
    """Clears the LRU caches for decorated functions in qsar_api before each test."""
    qsar_api.QSARToolboxAPI.search_by_name.cache_clear()
    qsar_api.QSARToolboxAPI.search_by_smiles.cache_clear()
    qsar_api.QSARToolboxAPI.get_all_chemical_data.cache_clear()
    qsar_api.QSARToolboxAPI.apply_all_calculators.cache_clear()
    qsar_api.QSARToolboxAPI.get_chemical_profiling.cache_clear()
    # No need to yield anything for setup/teardown

# --- Fixture for the API Client ---
@pytest.fixture
def api_client():
    # Assuming utils is importable from tests directory. May need path adjustments.
    # If running pytest from the root 'streamlined_qsar_app' directory, this should work.
    # If running from 'cleaned toolbox', adjust the import path or configure pytest paths.
    # For now, assuming standard pytest discovery from root.
    from qsar_assistant.utils.qsar_api import QSARToolboxAPI, SearchOptions
    return QSARToolboxAPI(base_url="http://mock-qsar-api:5000/api", max_retries=1) # [cite: 95]

# --- Mocking Requests ---
# Need to adjust the patch target based on where requests is imported in qsar_api.py
# Assuming 'requests' is imported directly in 'utils.qsar_api'
@patch('qsar_assistant.utils.qsar_api.requests.Session.request') # [cite: 95]
def test_search_by_name_success(mock_request, api_client):
    """Test successful search by name"""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = [{"ChemId": "123", "Name": "TestChem"}] # Example response
    mock_request.return_value = mock_response

    # Need SearchOptions from the correct import
    from qsar_assistant.utils.qsar_api import SearchOptions
    result = api_client.search_by_name("TestChem", SearchOptions.EXACT_MATCH) # [cite: 102]

    assert result == [{"ChemId": "123", "Name": "TestChem"}]
    mock_request.assert_called_once()
    # Add more assertions on the request args if needed (URL, method, params)

@patch('qsar_assistant.utils.qsar_api.requests.Session.request') # [cite: 95]
def test_search_by_smiles_timeout(mock_request, api_client):
    """Test search by SMILES timeout""" # [cite: 104]
    # Need QSARTimeoutError from the correct import
    from qsar_assistant.utils.qsar_api import QSARTimeoutError
    mock_request.side_effect = requests.exceptions.Timeout("Request timed out")

    with pytest.raises(QSARTimeoutError): # [cite: 94]
        api_client.search_by_smiles("CCO") # [cite: 104]

@patch('qsar_assistant.utils.qsar_api.requests.Session.request') # [cite: 95]
def test_get_all_chemical_data_error_response(mock_request, api_client):
    """Test handling of API error response"""
    # Need QSARResponseError from the correct import
    from qsar_assistant.utils.qsar_api import QSARResponseError
    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_request.return_value = mock_response

    with pytest.raises(QSARResponseError, match="API returned status 500"): # [cite: 94, 99]
        api_client.get_all_chemical_data("chem123") # [cite: 106]

# --- Add more tests for: ---
# - Other API methods (apply_all_calculators, get_chemical_profiling etc.) [cite: 108, 119]
# - Connection errors (requests.exceptions.ConnectionError -> QSARConnectionError) [cite: 94]
# - Different search options [cite: 102]
# - Handling empty responses [cite: 99]
# - Retry logic (might require more complex mocking) [cite: 95, 96]
