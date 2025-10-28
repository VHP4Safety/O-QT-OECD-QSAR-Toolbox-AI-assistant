"""Tests for the asynchronous LLM agent helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from langchain_core.runnables import RunnableLambda

from oqt_assistant.utils import llm_utils
from oqt_assistant.utils.llm_utils import (
    analyze_chemical_context,
    analyze_environmental_fate,
    analyze_experimental_data,
    analyze_physical_properties,
    analyze_profiling_reactivity,
    analyze_read_across,
    synthesize_report,
)


FAKE_LLM_CONFIG = {
    "provider": "OpenAI",
    "api_key": "test-key",
    "model_id": "gpt-4o",
    "model_name": "gpt-4o",
    "temperature": 0.0,
    "max_tokens": 2000,
}


def run_async(awaitable):
    """Execute a coroutine and return its result."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def clear_llm_caches():
    """Ensure cached LangChain chains do not leak between tests."""
    llm_utils.analyze_chemical_context.cache_clear()
    llm_utils.analyze_physical_properties.cache_clear()
    llm_utils.analyze_environmental_fate.cache_clear()
    llm_utils.analyze_profiling_reactivity.cache_clear()
    llm_utils.analyze_experimental_data.cache_clear()
    llm_utils.analyze_read_across.cache_clear()
    llm_utils.synthesize_report.cache_clear()


@patch("oqt_assistant.utils.llm_utils.initialize_llm")
def test_analyze_physical_properties_call(mock_init_llm):
    mock_init_llm.return_value = RunnableLambda(lambda *_: "Mocked Physical Properties Analysis")
    result = run_async(analyze_physical_properties({"LogP": 2.5}, "context", FAKE_LLM_CONFIG))
    assert result == "Mocked Physical Properties Analysis"


@patch("oqt_assistant.utils.llm_utils.initialize_llm")
def test_synthesize_report_call(mock_init_llm):
    mock_init_llm.return_value = RunnableLambda(lambda *_: "Mocked Synthesis Report")
    result = run_async(
        synthesize_report(
            "TestChem",
            ["A", "B", "C", "D"],
            "Read across",
            "General context",
            FAKE_LLM_CONFIG,
        )
    )
    assert result == "Mocked Synthesis Report"


@patch("oqt_assistant.utils.llm_utils.initialize_llm")
def test_analyze_chemical_context(mock_init_llm):
    mock_init_llm.return_value = RunnableLambda(
        lambda *_: "Confirmed Chemical: TestChem (CAS: 123-45-6, SMILES: C1=CC)"
    )
    result = run_async(
        analyze_chemical_context(
            {"basic_info": {"Name": "TestChem", "CAS": "123-45-6", "SMILES": "C1=CC"}},
            "context",
            FAKE_LLM_CONFIG,
        )
    )
    assert "TestChem" in result


@patch("oqt_assistant.utils.llm_utils.initialize_llm")
def test_analyze_environmental_fate(mock_init_llm):
    mock_init_llm.return_value = RunnableLambda(lambda *_: "Mocked Environmental Fate Analysis")
    result = run_async(analyze_environmental_fate({"LogKow": 3.5}, "context", FAKE_LLM_CONFIG))
    assert result == "Mocked Environmental Fate Analysis"


@patch("oqt_assistant.utils.llm_utils.initialize_llm")
def test_analyze_profiling_reactivity(mock_init_llm):
    mock_init_llm.return_value = RunnableLambda(lambda *_: "Mocked Profiling Analysis")
    result = run_async(analyze_profiling_reactivity({"profilers": {}}, "context", FAKE_LLM_CONFIG))
    assert result == "Mocked Profiling Analysis"


@patch("oqt_assistant.utils.llm_utils.initialize_llm")
def test_analyze_experimental_data(mock_init_llm):
    mock_init_llm.return_value = RunnableLambda(lambda *_: "Mocked Experimental Data Analysis")
    result = run_async(analyze_experimental_data({"experimental_results": []}, "context", FAKE_LLM_CONFIG))
    assert result == "Mocked Experimental Data Analysis"


@patch("oqt_assistant.utils.llm_utils.initialize_llm")
def test_analyze_read_across(mock_init_llm):
    mock_init_llm.return_value = RunnableLambda(lambda *_: "Mocked Read Across Analysis")
    result = run_async(
        analyze_read_across(
            {"chemical_data": {"basic_info": {"Name": "TestChem"}}},
            specialist_outputs=["A", "B", "C", "D"],
            context="context",
            llm_config=FAKE_LLM_CONFIG,
        )
    )
    assert result == "Mocked Read Across Analysis"


@patch("oqt_assistant.utils.llm_utils.initialize_llm")
def test_analyze_physical_properties_error_handling(mock_init_llm):
    def _raise(*_args, **_kwargs):
        raise ValueError("Test LLM error")

    mock_init_llm.return_value = RunnableLambda(_raise)
    result = run_async(analyze_physical_properties({"LogP": 2.5}, "context", FAKE_LLM_CONFIG))
    assert "Content not available" in result
