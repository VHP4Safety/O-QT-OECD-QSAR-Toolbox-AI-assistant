# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache 2.0

"""
Simple API behavior tests for the streamlined_qsar_app.

These tests focus on validating the input/output behavior of the agent functions,
not the internal implementation details.
"""
import pytest
import asyncio
import json
from unittest.mock import patch

class TestChemicalAnalysis:
    """Test the chemical analysis functionality by checking input-output behavior."""

    @pytest.mark.asyncio
    async def test_chemical_data_formatting(self):
        """Test the data formatting doesn't throw errors and has expected keys."""
        from oqt_assistant.utils.data_formatter import format_chemical_data, format_calculator_result, clean_response_data

        # Test format_chemical_data with minimal data
        chemical_data_result = format_chemical_data({"SubstanceType": "Test", "ChemId": "123"})
        assert chemical_data_result["type"] == "Test"
        assert chemical_data_result["id"] == "123"
        assert "names" in chemical_data_result
        assert "smiles" in chemical_data_result

        # Test format_calculator_result with minimal data
        calc_result = format_calculator_result({"CalculatorName": "Test Calc"})
        assert calc_result["name"] == "Test Calc"
        assert "value" in calc_result
        assert "unit" in calc_result

        # Test clean_response_data with minimal structure
        clean_data = clean_response_data({
            "chemical_data": {"basic_info": {"SubstanceType": "Test"}},
            "experimental_data": [{"Endpoint": "Test"}],
            "profiling": {"available_profilers": [{"name": "Test"}]}
        })
        assert "chemical_data" in clean_data
        assert "properties" in clean_data
        assert "experimental_data" in clean_data
        assert "profiling" in clean_data

    @pytest.mark.asyncio
    async def test_safe_json_serialization(self):
        """Test the safe_json serialization function handles complex types."""
        from oqt_assistant.utils.data_formatter import safe_json
        
        # Test dict serialization
        data = {"name": "test", "value": 123}
        json_str = safe_json(data)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["name"] == "test"
        assert parsed["value"] == 123

    @pytest.mark.asyncio
    async def test_qsar_api_error_classes(self):
        """Test that QSAR API error classes are correctly defined."""
        from oqt_assistant.utils.qsar_api import QSARConnectionError, QSARTimeoutError, QSARResponseError
        
        # Test that error classes exist and are Exception subclasses
        assert issubclass(QSARConnectionError, Exception)
        assert issubclass(QSARTimeoutError, Exception) 
        assert issubclass(QSARResponseError, Exception)
        
        # Test error instantiation
        error = QSARConnectionError("Test error message")
        assert str(error) == "Test error message"

    @pytest.mark.asyncio
    async def test_agent_functions_exist(self):
        """Test that the agent functions exist and have the expected signature."""
        from oqt_assistant.utils.llm_utils import (
            analyze_chemical_context,
            analyze_physical_properties,
            analyze_environmental_fate,
            analyze_profiling_reactivity,
            analyze_experimental_data,
            analyze_read_across,
            synthesize_report
        )
        
        # Check function signatures without executing them
        assert callable(analyze_chemical_context)
        assert callable(analyze_physical_properties)
        assert callable(analyze_environmental_fate)
        assert callable(analyze_profiling_reactivity)
        assert callable(analyze_experimental_data)
        assert callable(analyze_read_across)
        assert callable(synthesize_report)

        # Verify the functions are async
        import inspect
        assert inspect.iscoroutinefunction(analyze_chemical_context)
        assert inspect.iscoroutinefunction(analyze_physical_properties)
        assert inspect.iscoroutinefunction(analyze_environmental_fate)
        assert inspect.iscoroutinefunction(analyze_profiling_reactivity)
        assert inspect.iscoroutinefunction(analyze_experimental_data)
        assert inspect.iscoroutinefunction(analyze_read_across)
        assert inspect.iscoroutinefunction(synthesize_report)
