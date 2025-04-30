# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: MIT

# tests/test_llm_utils.py
import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
# Import the module itself for cache clearing
from qsar_assistant.utils import llm_utils
# Import the agent functions directly
from qsar_assistant.utils.llm_utils import (
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
# Patch the actual ainvoke method called by the LangChain chain

@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_analyze_physical_properties_call(mock_ainvoke):
    """Test analyze_physical_properties with a mocked LLM ainvoke response"""
    # Configure the mock ainvoke method directly
    # The StrOutputParser at the end of the chain will return this string directly
    mock_ainvoke.return_value = "Mocked Physical Properties Analysis"

    # Test data
    test_data = {"LogP": 2.5, "MeltingPoint": {"Value": 100, "Unit": "C"}}
    test_context = "Environmental safety assessment"

    # Call the actual agent function (it will construct the chain and call ainvoke, which is now mocked)
    result = await analyze_physical_properties(test_data, test_context)

    # Verify the result matches the mocked ainvoke response
    assert result == "Mocked Physical Properties Analysis"

    # Optionally, verify ainvoke was called
    mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_synthesize_report_call(mock_ainvoke):
    """Test synthesize_report with a mocked LLM ainvoke response"""
    # Configure the mock ainvoke method
    mock_ainvoke.return_value = "Mocked Synthesis Report"

    # Test data
    analyses = ["Phys props analysis.", "Env fate analysis.", "Profiling analysis.", "Exp data analysis."]
    read_across = "Read across strategy."
    context = "General hazard"
    identifier = "TestChem"

    # Call the actual agent function
    result = await synthesize_report(identifier, analyses, read_across, context)

    # Check the result matches the mocked ainvoke response
    assert result == "Mocked Synthesis Report"

    # Optionally, verify ainvoke was called
    mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_analyze_chemical_context(mock_ainvoke):
    """Test analyze_chemical_context with a mocked LLM ainvoke response"""
    # Configure the mock ainvoke method
    mock_ainvoke.return_value = "Confirmed Chemical: TestChem (CAS: 123-45-6, SMILES: C1=CC=C2)"

    # Test data
    chemical_data = {"basic_info": {"Name": "TestChem", "CAS": "123-45-6", "SMILES": "C1=CC=C2"}}
    context = "Chemical safety assessment"

    # Call the actual agent function
    result = await analyze_chemical_context(chemical_data, context)

    # Check the result matches the mocked ainvoke response
    # Note: The agent function might strip whitespace, so compare stripped result
    assert result.strip() == "Confirmed Chemical: TestChem (CAS: 123-45-6, SMILES: C1=CC=C2)"

    # Optionally, verify ainvoke was called
    mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_analyze_environmental_fate(mock_ainvoke):
    """Test analyze_environmental_fate with a mocked LLM ainvoke response"""
    # Configure the mock ainvoke method
    mock_ainvoke.return_value = "Mocked Environmental Fate Analysis"

    # Test data
    data = {"LogKow": 3.5, "BCF": 100}
    context = "Environmental impact assessment"

    # Call the actual agent function
    result = await analyze_environmental_fate(data, context)

    # Check the result matches the mocked ainvoke response
    assert result == "Mocked Environmental Fate Analysis"

    # Optionally, verify ainvoke was called
    mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_analyze_profiling_reactivity(mock_ainvoke):
    """Test analyze_profiling_reactivity with a mocked LLM ainvoke response"""
    # Configure the mock ainvoke method
    mock_ainvoke.return_value = "Mocked Profiling Analysis"

    # Test data
    data = {"profilers": {"DNA Binding": {"result": "positive"}}}
    context = "Toxicity assessment"

    # Call the actual agent function
    result = await analyze_profiling_reactivity(data, context)

    # Check the result matches the mocked ainvoke response
    assert result == "Mocked Profiling Analysis"

    # Optionally, verify ainvoke was called
    mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_analyze_experimental_data(mock_ainvoke):
    """Test analyze_experimental_data with a mocked LLM ainvoke response"""
    # Configure the mock ainvoke method
    mock_ainvoke.return_value = "Mocked Experimental Data Analysis"

    # Test data
    data = {"experimental_results": [{"Endpoint": "LC50", "Value": 50}]}
    context = "Aquatic toxicity assessment"

    # Call the actual agent function
    result = await analyze_experimental_data(data, context)

    # Check the result matches the mocked ainvoke response
    assert result == "Mocked Experimental Data Analysis"

    # Optionally, verify ainvoke was called
    mock_ainvoke.assert_called_once()

@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_analyze_read_across(mock_ainvoke):
    """Test analyze_read_across with a mocked LLM ainvoke response"""
    # Configure the mock ainvoke method
    mock_ainvoke.return_value = "Mocked Read Across Analysis"

    # Test data
    results = {"chemical_data": {"basic_info": {"Name": "TestChem"}}}
    specialist_outputs = ["Analysis 1", "Analysis 2", "Analysis 3", "Analysis 4"]
    context = "Read across assessment"

    # Call the actual agent function
    result = await analyze_read_across(results, specialist_outputs, context)

    # Check the result matches the mocked ainvoke response
    assert result == "Mocked Read Across Analysis"

    # Optionally, verify ainvoke was called
    mock_ainvoke.assert_called_once()

# --- Tests for error handling ---
@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_analyze_physical_properties_error_handling(mock_ainvoke):
    """Test that analyze_physical_properties handles LLM ainvoke exceptions gracefully"""

    # Configure the mock ainvoke method to raise an exception
    mock_ainvoke.side_effect = ValueError("Test LLM error")

    # Test data
    test_data = {"LogP": 2.5}
    test_context = "Error test"

    # Call the actual agent function
    result = await analyze_physical_properties(test_data, test_context)

    # Check that the agent function returned the expected error message, capturing the raised ValueError
    assert "Error in Physical Properties Agent: Test LLM error" in result

@pytest.mark.asyncio
@patch('langchain_openai.ChatOpenAI.ainvoke', new_callable=AsyncMock)
async def test_synthesize_report_error_handling(mock_ainvoke):
    """Test that synthesize_report handles LLM ainvoke exceptions gracefully"""

    # Configure the mock ainvoke method to raise an exception
    mock_ainvoke.side_effect = ValueError("Test LLM error")

    # Test data
    analyses = ["Analysis 1", "Analysis 2", "Analysis 3", "Analysis 4"]
    read_across = "Read across"
    context = "Error test"
    identifier = "ErrorChem"

    # Call the actual agent function
    result = await synthesize_report(identifier, analyses, read_across, context)

    # Check that the agent function returned the expected error message, capturing the raised ValueError
    assert "Error synthesizing report for ErrorChem: Error in Synthesizer Agent: Test LLM error" in result
