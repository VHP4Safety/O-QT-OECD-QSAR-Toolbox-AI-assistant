# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import oqt_assistant.cli as cli


def test_cli_main_ui(monkeypatch):
    """UI command should delegate to _run_streamlit with passthrough args."""

    called = {}

    def fake_run_streamlit(args):
        called["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_streamlit", fake_run_streamlit)

    exit_code = cli.main(["ui"])

    assert exit_code == 0
    assert called["args"] == []


def test_cli_analyze_tmpdir(monkeypatch, tmp_path: Path):
    """CLI analyze command should write log, markdown, and PDF outputs."""

    class FakeQSAR:  # minimal API facade
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def get_simulators(self):
            return [{"Guid": "sim-1", "Caption": "Simulator"}]

        def get_profilers(self):
            return [{"Guid": "prof-1", "Caption": "Profiler"}]

        def get_all_qsar_models_catalog(self):
            return [{"Guid": "qsar-1", "Caption": "Model"}]

    def fake_recommended(catalog):
        return catalog[:1]

    async def fake_execute(*_args, **_kwargs):
        stub = cli.app.st
        stub.session_state.analysis_results = {"result": "ok"}
        stub.session_state.final_report = "Report body"
        stub.session_state.comprehensive_log = {
            "metadata": {},
            "configuration": {},
            "inputs": {},
            "data_retrieval": {"processed_qsar_toolbox_data": {}},
            "analysis": {
                "specialist_agent_outputs": {},
                "synthesized_report": "Report body",
            },
        }

    monkeypatch.setattr("oqt_assistant.utils.qsar_api.QSARToolboxAPI", FakeQSAR)
    monkeypatch.setattr("oqt_assistant.utils.qsar_models.derive_recommended_qsar_models", fake_recommended)
    monkeypatch.setattr(
        "oqt_assistant.utils.pdf_generator.generate_pdf_report", lambda data: io.BytesIO(b"%PDF-1.4")
    )
    import oqt_assistant.app as app_module
    monkeypatch.setattr(cli, "app", app_module, raising=False)
    monkeypatch.setattr(app_module, "execute_analysis_async", AsyncMock(side_effect=fake_execute))

    exit_code = cli.main(
        [
            "analyze",
            "Acetone",
            "--api-url",
            "http://example/api/v6",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    log_path = tmp_path / "Acetone_log.json"
    report_md = tmp_path / "Acetone_report.md"
    report_pdf = tmp_path / "Acetone_report.pdf"

    assert log_path.exists()
    assert report_md.exists()
    assert report_pdf.exists()

    data = json.loads(log_path.read_text())
    assert data["analysis"]["synthesized_report"] == "Report body"


def test_package_main_delegates(monkeypatch):
    """Running python -m oqt_assistant should reuse the CLI entry point."""

    calls = {}

    def fake_main(argv=None):
        calls["argv"] = argv
        return 5

    monkeypatch.setattr("oqt_assistant.cli.main", fake_main)

    from oqt_assistant import __main__ as package_main

    assert package_main.main(["demo"]) == 5
    assert calls["argv"] == ["demo"]
