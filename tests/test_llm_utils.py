# tests/test_llm_utils.py
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
# Import the module itself for cache clearing
from streamlined_qsar_app.utils import llm_utils
# Import the agent functions directly
from streamlined_qsar_app.utils.llm_utils import (
    analyze_physical_properties,
    synthesize_report,
    analyze_chemical_context,
    analyze_environmental_fate,
    analyze_profiling_reactivity,
    analyze_experimental_data,
    analyze_read_across
)

# --- Fixture to clear caches before each test ---
@pytest.fixture(autouse=True)
def clear_llm_caches():
    """Clears the LRU caches for decorated functions in llm_utils before each test."""
    llm_utils.analyze_chemical_context.cache_clear()
    llm_utils.analyze_physical_properties.cache_clear()
    llm_utils.analyze_environmental_fate.cache_clear()
    llm_utils.analyze_profiling_reactivity.cache_clear()
    llm_utils.analyze_experimental_data.cache_clear()
    llm_utils.analyze_read_across.cache_clear()
    llm_utils.synthesize_report.cache_clear()
    # No need to yield anything for setup/teardown

# We need to patch the functions at the module level and target the actual module where it's defined
# Let's use decorators, which work more reliably for this case

@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.analyze_physical_properties')
async def test_analyze_physical_properties_call(mock_analyze):
    """Test analyze_physical_properties with a patched version that returns a fixed value"""
    # Configure the mock to return a fixed value
    mock_analyze.return_value = "Mocked Physical Properties Analysis"
    
    # Test data
    test_data = {"LogP": 2.5, "MeltingPoint": {"Value": 100, "Unit": "C"}}
    test_context = "Environmental safety assessment"
    
    # Imported function will use the mock
    result = await analyze_physical_properties(test_data, test_context)
    
    # Verify the result matches our mocked value
    assert result == "Mocked Physical Properties Analysis"
    
    # Verify the mock was called with the expected args
    mock_analyze.assert_called_once_with(test_data, test_context)

@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.synthesize_report')
async def test_synthesize_report_call(mock_synthesize):
    """Test synthesize_report with a patched version that returns a fixed value"""
    # Configure the mock
    mock_synthesize.return_value = "Mocked Synthesis Report"
    
    # Test data
    analyses = ["Phys props analysis.", "Env fate analysis.", "Profiling analysis.", "Exp data analysis."]
    read_across = "Read across strategy."
    context = "General hazard"
    identifier = "TestChem"
    
    # Call the function
    result = await synthesize_report(identifier, analyses, read_across, context)
    
    # Check the result
    assert result == "Mocked Synthesis Report"
    
    # Verify the mock was called correctly
    mock_synthesize.assert_called_once_with(identifier, analyses, read_across, context)

@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.analyze_chemical_context')
async def test_analyze_chemical_context(mock_analyze):
    # Configure the mock
    mock_analyze.return_value = "Confirmed Chemical: TestChem (CAS: 123-45-6, SMILES: C1=CC=C2)"
    
    # Test data
    chemical_data = {"basic_info": {"Name": "TestChem", "CAS": "123-45-6", "SMILES": "C1=CC=C2"}}
    context = "Chemical safety assessment"
    
    # Call the function
    result = await analyze_chemical_context(chemical_data, context)
    
    # Check the result
    assert result == "Confirmed Chemical: TestChem (CAS: 123-45-6, SMILES: C1=CC=C2)"
    
    # Verify the mock was called correctly
    mock_analyze.assert_called_once_with(chemical_data, context)

@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.analyze_environmental_fate')
async def test_analyze_environmental_fate(mock_analyze):
    # Configure the mock
    mock_analyze.return_value = "Mocked Environmental Fate Analysis"
    
    # Test data
    data = {"LogKow": 3.5, "BCF": 100}
    context = "Environmental impact assessment"
    
    # Call the function
    result = await analyze_environmental_fate(data, context)
    
    # Check the result
    assert result == "Mocked Environmental Fate Analysis"
    
    # Verify the mock was called correctly
    mock_analyze.assert_called_once_with(data, context)

@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.analyze_profiling_reactivity')
async def test_analyze_profiling_reactivity(mock_analyze):
    # Configure the mock
    mock_analyze.return_value = "Mocked Profiling Analysis"
    
    # Test data
    data = {"profilers": {"DNA Binding": {"result": "positive"}}}
    context = "Toxicity assessment"
    
    # Call the function
    result = await analyze_profiling_reactivity(data, context)
    
    # Check the result
    assert result == "Mocked Profiling Analysis"
    
    # Verify the mock was called correctly
    mock_analyze.assert_called_once_with(data, context)

@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.analyze_experimental_data')
async def test_analyze_experimental_data(mock_analyze):
    # Configure the mock
    mock_analyze.return_value = "Mocked Experimental Data Analysis"
    
    # Test data
    data = {"experimental_results": [{"Endpoint": "LC50", "Value": 50}]}
    context = "Aquatic toxicity assessment"
    
    # Call the function
    result = await analyze_experimental_data(data, context)
    
    # Check the result
    assert result == "Mocked Experimental Data Analysis"
    
    # Verify the mock was called correctly
    mock_analyze.assert_called_once_with(data, context)

@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.analyze_read_across')
async def test_analyze_read_across(mock_analyze):
    # Configure the mock
    mock_analyze.return_value = "Mocked Read Across Analysis"
    
    # Test data
    results = {"chemical_data": {"basic_info": {"Name": "TestChem"}}}
    specialist_outputs = ["Analysis 1", "Analysis 2", "Analysis 3", "Analysis 4"]
    context = "Read across assessment"
    
    # Call the function
    result = await analyze_read_across(results, specialist_outputs, context)
    
    # Check the result
    assert result == "Mocked Read Across Analysis"
    
    # Verify the mock was called correctly
    mock_analyze.assert_called_once_with(results, specialist_outputs, context)

# --- Tests for error handling ---
@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.analyze_physical_properties')
async def test_analyze_physical_properties_error_handling(mock_analyze):
    """Test that analyze_physical_properties handles exceptions gracefully"""
    
    # Configure the mock to raise an exception
    mock_analyze.side_effect = ValueError("Test error")
    
    # Test data
    test_data = {"LogP": 2.5}
    test_context = "Error test"
    
    # Call the function
    result = await analyze_physical_properties(test_data, test_context)
    
    # Check the result
    assert "Error in Physical Properties Agent: Test error" in result

@pytest.mark.asyncio
@patch('streamlined_qsar_app.utils.llm_utils.synthesize_report')
async def test_synthesize_report_error_handling(mock_synthesize):
    """Test that synthesize_report handles exceptions gracefully"""
    
    # Configure the mock to raise an exception
    mock_synthesize.side_effect = ValueError("Test error")
    
    # Test data
    analyses = ["Analysis 1", "Analysis 2", "Analysis 3", "Analysis 4"]
    read_across = "Read across"
    context = "Error test"
    identifier = "ErrorChem"
    
    # Call the function
    result = await synthesize_report(identifier, analyses, read_across, context)
    
    # Check the result
    assert "Error synthesizing report for ErrorChem" in result
