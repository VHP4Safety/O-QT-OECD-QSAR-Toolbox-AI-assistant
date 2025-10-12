# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""Integration-level tests for orchestration helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from oqt_assistant import app


class _DummyAPI:
    """Minimal QSAR Toolbox facade capturing call flow."""

    def __init__(self, *, raise_on_optional: bool = False) -> None:
        self.raise_on_optional = raise_on_optional
        self.session = SimpleNamespace(get=lambda *_args, **_kwargs: None)
        self.calls: list[str] = []

    # Connection bootstrap -------------------------------------------------
    def get_version(self) -> dict[str, Any]:
        self.calls.append("get_version")
        return {"version": "6.0"}

    def get_simulators(self) -> list[dict[str, Any]]:
        self.calls.append("get_simulators")
        return [{"Guid": "sim-1", "Caption": "Simulator 1"}]

    def get_profilers(self) -> list[dict[str, Any]]:
        self.calls.append("get_profilers")
        return [{"Guid": "prof-1"}]

    def get_all_qsar_models_catalog(self) -> list[dict[str, Any]]:
        self.calls.append("get_all_qsar_models_catalog")
        return [{"Guid": "qsar-1", "Caption": "QSAR"}]

    # Search helpers -------------------------------------------------------
    def search_by_name(self, identifier: str, **_kwargs) -> list[dict[str, Any]]:
        self.calls.append(f"search_by_name:{identifier}")
        return [{"ChemId": "chem-1", "Name": identifier}]

    def search_by_smiles(self, *_args, **_kwargs) -> list[dict[str, Any]]:
        self.calls.append("search_by_smiles")
        return []

    # Optional heavy calls -------------------------------------------------
    def apply_simulator(self, *_args, **_kwargs) -> list[dict[str, Any]]:
        self.calls.append("apply_simulator")
        if self.raise_on_optional:
            raise AssertionError("Simulators should be skipped")
        return [{"Metabolite": "M1"}]

    def get_all_chemical_data(self, *_args, **_kwargs) -> list[dict[str, Any]]:
        self.calls.append("get_all_chemical_data")
        if self.raise_on_optional:
            raise AssertionError("Experimental payload should be skipped")
        return [{"Endpoint": "LC50"}]

    def get_chemical_profiling(self, *_args, **_kwargs) -> dict[str, Any]:
        self.calls.append("get_chemical_profiling")
        if self.raise_on_optional:
            raise AssertionError("Profiling should be skipped")
        return {"alerts": []}


def test_perform_chemical_analysis_with_minimal_scope(monkeypatch, streamlit_stub):
    """When optional scopes are disabled the heavy endpoints stay untouched."""
    streamlit_stub.session_state.clear()
    app.initialize_session_state()

    dummy_api = _DummyAPI(raise_on_optional=True)
    monkeypatch.setattr(app, "QSARToolboxAPI", lambda *args, **kwargs: dummy_api)

    monkeypatch.setattr(
        app,
        "select_hit_with_properties",
        lambda api, identifier, hits, logger=None: (
            {"ChemId": "chem-1", "SubstanceType": "Test"},
            {"LogP": {"value": 1.1, "unit": "logP"}},
            "chem-1",
            [],
        ),
    )

    scope = {
        "include_properties": False,
        "include_experimental": False,
        "include_profiling": False,
        "include_qsar": False,
        "selected_profiler_guids": [],
        "selected_qsar_model_guids": [],
    }

    result = app.perform_chemical_analysis(
        identifier="TestChem",
        search_type="name",
        context="context",
        simulator_guids=[],
        qsar_config={"api_url": "http://fake"},
        scope_config=scope,
    )

    assert result["chemical_data"]["basic_info"]["ChemId"] == "chem-1"
    assert result["metabolism"]["status"] == "Skipped"
    assert result["experimental_data"] == []
    assert result["profiling"] == {}
    assert result["qsar_models"]["raw"]["predictions"] == []


def test_perform_chemical_analysis_runs_full_scope(monkeypatch, streamlit_stub):
    """Full-scope execution orchestrates simulators, experiments, profiling and QSAR."""
    streamlit_stub.session_state.clear()
    app.initialize_session_state()

    dummy_api = _DummyAPI()
    monkeypatch.setattr(app, "QSARToolboxAPI", lambda *args, **kwargs: dummy_api)

    monkeypatch.setattr(
        app,
        "select_hit_with_properties",
        lambda api, identifier, hits, logger=None: (
            {"ChemId": "chem-1", "SubstanceType": "Test"},
            {"LogP": {"value": 2.5, "unit": "logP"}},
            "chem-1",
            [],
        ),
    )

    monkeypatch.setattr(
        app,
        "run_qsar_predictions",
        lambda *_args, **_kwargs: {
            "catalog_size": 10,
            "executed_models": 1,
            "predictions": [{"Endpoint": "LC50", "Value": 1.23}],
            "summary": {"total": 1},
            "selected_model_guids": ["qsar-1"],
        },
    )

    scope = {
        "include_properties": True,
        "include_experimental": True,
        "include_profiling": True,
        "include_qsar": True,
        "selected_profiler_guids": ["prof-1"],
        "selected_qsar_model_guids": ["qsar-1"],
        "include_slow_profilers": False,
    }

    result = app.perform_chemical_analysis(
        identifier="TestChem",
        search_type="name",
        context="context",
        simulator_guids=["sim-1"],
        qsar_config={"api_url": "http://fake"},
        scope_config=scope,
    )

    assert result["metabolism"]["status"] in {"Success", "Partial Success"}
    assert result["experimental_data"]
    assert result["profiling"]["alerts"] == []
    assert result["qsar_models"]["raw"]["summary"]["total"] == 1
    assert result["qsar_models"]["processed"]


def test_perform_chemical_analysis_requires_api_url(monkeypatch, streamlit_stub):
    """QSAR configuration without an endpoint should raise immediately."""
    streamlit_stub.session_state.clear()
    app.initialize_session_state()

    monkeypatch.setattr(app, "QSARToolboxAPI", lambda *args, **kwargs: _DummyAPI())
    monkeypatch.setattr(
        app,
        "select_hit_with_properties",
        lambda *_args, **_kwargs: (
            {"ChemId": "chem-1"},
            {},
            "chem-1",
            [],
        ),
    )

    with pytest.raises(ValueError):
        app.perform_chemical_analysis(
            identifier="TestChem",
            search_type="name",
            context="context",
            simulator_guids=[],
            qsar_config={},
        )
