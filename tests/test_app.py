# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for high-level application helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from oqt_assistant import app


def test_initialize_session_state_defaults(streamlit_stub):
    """Verify default session keys and configuration snapshots."""
    streamlit_stub.session_state.clear()

    app.initialize_session_state()

    state = streamlit_stub.session_state
    assert state.analysis_results is None
    assert state.final_report is None
    assert state.comprehensive_log is None
    assert state.progress_value == 0.0
    assert state.progress_description == ""
    assert state.retry_count == 0
    assert state.max_retries == 15

    # LLM configuration defaults
    assert state.llm_config["provider"] == "OpenAI"
    assert state.llm_config["model_name"] == "gpt-4.1-nano"
    assert state.llm_config["temperature"] == 0.15
    assert state.llm_config["max_tokens"] == 10_000

    # QSAR connection defaults
    assert "http" in state.qsar_config["api_url"]
    assert state.qsar_config["config_complete"] is True


def test_initialize_session_state_is_idempotent(streamlit_stub):
    """Calling the initializer twice must preserve manual tweaks."""
    streamlit_stub.session_state.clear()
    app.initialize_session_state()

    streamlit_stub.session_state.llm_config["provider"] = "OpenRouter"
    streamlit_stub.session_state.qsar_config["api_url"] = "http://example/api"

    app.initialize_session_state()

    assert streamlit_stub.session_state.llm_config["provider"] == "OpenRouter"
    assert streamlit_stub.session_state.qsar_config["api_url"] == "http://example/api"


def test_update_progress_creates_progress_bar(streamlit_stub):
    """Progress helpers update state and surface a reusable bar."""
    streamlit_stub.session_state.clear()
    app.initialize_session_state()

    app.update_progress(0.5, "Halfway there")

    state = streamlit_stub.session_state
    assert state.progress_value == 0.5
    assert state.progress_description == "Halfway there"
    assert hasattr(state, "progress_bar")
    assert state.progress_bar.value == 0.5
    assert state.progress_bar.text == "Status: Halfway there"


def test_check_connection_populates_catalog(monkeypatch, streamlit_stub):
    """Connection checks should populate simulator and QSAR caches."""
    streamlit_stub.session_state.clear()
    app.initialize_session_state()

    class DummyAPI:
        session = MagicMock()

        def get_version(self):
            return {"version": "6.0"}

        def get_simulators(self):
            return [{"Guid": "sim-1", "Caption": "Simulator"}]

        def get_profilers(self):
            return [{"Guid": "prof-1"}]

        def get_all_qsar_models_catalog(self):
            return [{"Guid": "qsar-1", "Caption": "QSAR"}]

    monkeypatch.setattr(
        app,
        "derive_recommended_qsar_models",
        lambda catalog: catalog,
    )

    ok = app.check_connection(DummyAPI())

    assert ok is True
    assert streamlit_stub.session_state.connection_status is True
    assert streamlit_stub.session_state.available_simulators[0]["Guid"] == "sim-1"
    assert streamlit_stub.session_state.available_qsar_models[0]["Guid"] == "qsar-1"
