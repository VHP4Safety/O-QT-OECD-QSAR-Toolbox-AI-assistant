# SPDX-FileCopyrightText: 2025 Ivo Djidrovski <i.djidrovski@uu.nl>
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

def _run_streamlit(ui_args: List[str]) -> int:
    """Launch the Streamlit UI in the current Python environment."""
    app_path = Path(__file__).parent / "app.py"
    if not app_path.exists():
        print(f"Could not locate app.py (looked in {app_path})", file=sys.stderr)
        return 1

    command = [sys.executable, "-m", "streamlit", "run", str(app_path), *ui_args]
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"Streamlit exited with code {exc.returncode}", file=sys.stderr)
        return exc.returncode


class _SessionState(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class _ProgressBar:
    def __init__(self, value: float = 0.0, text: Optional[str] = None) -> None:
        self.value = value
        self.text = text

    def progress(self, value: float, text: Optional[str] = None) -> "_ProgressBar":
        self.value = value
        self.text = text
        msg = text or ""
        print(f"[progress] {value * 100:.0f}% {msg}")
        return self

    def empty(self) -> None:
        self.value = 0.0
        self.text = None


class _Sidebar:
    def markdown(self, *args, **kwargs) -> None:
        return None

    def subheader(self, *args, **kwargs) -> None:
        return None

    def download_button(self, *args, **kwargs) -> None:
        return None

    def info(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None

    def button(self, *args, **kwargs) -> bool:
        return False

    def text_input(self, *args, value: str = "", **kwargs) -> str:
        return value

    def number_input(self, *args, value: Union[float, int] = 0, **kwargs) -> Union[float, int]:
        return value

    def expander(self, *args, **kwargs):
        class _Context:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc) -> bool:
                return False

            def markdown(self_inner, *args, **kwargs):
                return None

            def caption(self_inner, *args, **kwargs):
                return None

            def number_input(self_inner, *args, value: Union[float, int] = 0, **kwargs):
                return value

        return _Context()


class _StreamlitCLIStub:
    """Lightweight Streamlit facade for CLI execution."""

    class errors:
        StreamlitAPIException = Exception

    def __init__(self) -> None:
        self.session_state = _SessionState()
        self.sidebar = _Sidebar()

    # UI primitives ---------------------------------------------------------
    def header(self, *args, **kwargs) -> None:
        return None

    def subheader(self, *args, **kwargs) -> None:
        return None

    def markdown(self, *args, **kwargs) -> None:
        if args:
            print(args[0])

    def info(self, *args, **kwargs) -> None:
        if args:
            print(f"[info] {args[0]}")

    def warning(self, *args, **kwargs) -> None:
        if args:
            print(f"[warning] {args[0]}", file=sys.stderr)

    def error(self, *args, **kwargs) -> None:
        if args:
            print(f"[error] {args[0]}", file=sys.stderr)

    def success(self, *args, **kwargs) -> None:
        if args:
            print(f"[success] {args[0]}")

    def caption(self, *args, **kwargs) -> None:
        return None

    def write(self, *args, **kwargs) -> None:
        if args:
            print(args[0])

    def download_button(self, *args, **kwargs) -> None:
        return None

    def progress(self, value: float = 0.0, text: Optional[str] = None) -> _ProgressBar:
        return _ProgressBar(value, text)

    def rerun(self) -> None:
        return None


def _build_scope_config(args, api_url: str) -> tuple[Dict[str, Any], List[str]]:
    """Derive scope configuration and GUID mappings for CLI execution."""
    from oqt_assistant.utils.qsar_api import QSARToolboxAPI
    from oqt_assistant.utils.qsar_models import derive_recommended_qsar_models

    api_client = QSARToolboxAPI(base_url=api_url, timeout=(10, 60), max_retries=3)

    scope_config: Dict[str, Any] = {
        "include_properties": not args.skip_properties,
        "include_experimental": not args.skip_experimental,
        "include_profiling": not args.skip_profiling,
        "include_qsar": not args.skip_qsar,
        "include_slow_profilers": args.include_slow_profilers,
        "exclude_adme_tk": args.exclude_adme_tk,
        "exclude_mammalian_tox": args.exclude_mammalian_tox,
        "selected_profiler_guids": [],
        "selected_qsar_model_guids": [],
    }

    simulators = {sim.get("Guid"): sim.get("Caption") for sim in api_client.get_simulators() or []}
    profilers_catalog = api_client.get_profilers() or []
    profilers = {prof.get("Guid"): prof.get("Caption") for prof in profilers_catalog if isinstance(prof, dict)}

    if args.profiler_all:
        scope_config["selected_profiler_guids"] = list(profilers.keys())
    elif args.profiler:
        scope_config["selected_profiler_guids"] = args.profiler

    if not args.skip_qsar:
        catalog = api_client.get_all_qsar_models_catalog() or []
        if args.qsar_guid:
            selected = args.qsar_guid
        elif args.qsar_mode == "none":
            selected = []
        elif args.qsar_mode == "all":
            selected = [entry.get("Guid") for entry in catalog if entry.get("Guid")]
        else:  # recommended (default)
            recommended_entries = derive_recommended_qsar_models(catalog)
            selected = [entry.get("Guid") for entry in recommended_entries if entry.get("Guid")]
        scope_config["selected_qsar_model_guids"] = selected

    simulator_selection: List[str] = []
    if args.simulator_all:
        simulator_selection = list(simulators.keys())
    elif args.simulator:
        simulator_selection = args.simulator

    return scope_config, simulator_selection


def _run_cli_analysis(args) -> int:
    from oqt_assistant import app
    from oqt_assistant.utils.pdf_generator import generate_pdf_report

    stub = _StreamlitCLIStub()
    app.st = stub  # type: ignore[attr-defined]

    app.initialize_session_state()

    # Override configuration with CLI arguments / environment variables
    api_url = args.api_url or os.getenv("QSAR_TOOLBOX_API_URL")
    if not api_url:
        print("QSAR Toolbox API URL must be provided via --api-url or QSAR_TOOLBOX_API_URL env var.", file=sys.stderr)
        return 2

    stub.session_state.qsar_config["api_url"] = api_url
    stub.session_state.qsar_config["config_complete"] = True

    if args.provider:
        stub.session_state.llm_config["provider"] = args.provider
    if args.model_name:
        stub.session_state.llm_config["model_name"] = args.model_name
    if args.api_key:
        stub.session_state.llm_config["api_key"] = args.api_key
    if args.temperature is not None:
        stub.session_state.llm_config["temperature"] = args.temperature
    if args.max_output_tokens is not None:
        stub.session_state.llm_config["max_tokens"] = args.max_output_tokens

    scope_config, simulator_guids = _build_scope_config(args, api_url)

    if scope_config["selected_qsar_model_guids"]:
        print(f"Running {len(scope_config['selected_qsar_model_guids'])} QSAR models.")
    if scope_config["selected_profiler_guids"]:
        print(f"Running {len(scope_config['selected_profiler_guids'])} profilers.")
    if simulator_guids:
        print(f"Running {len(simulator_guids)} metabolism simulators.")

    try:
        asyncio.run(
            app.execute_analysis_async(
                identifier=args.identifier,
                search_type=args.search_type,
                context=args.context or "",
                simulator_guids=simulator_guids,
                llm_config=stub.session_state.llm_config,
                qsar_config=stub.session_state.qsar_config,
                scope_config=scope_config,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive handling
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 3

    results = stub.session_state.get("analysis_results")
    final_report = stub.session_state.get("final_report")
    log_data = stub.session_state.get("comprehensive_log")

    if not results or not log_data:
        print("Analysis did not return results.", file=sys.stderr)
        return 4

    output_dir = Path(args.output_dir or "cli_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_identifier = "_".join(args.identifier.strip().split()) or "analysis"

    json_path = output_dir / f"{safe_identifier}_log.json"
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(log_data, fp, indent=2)
    print(f"Saved comprehensive log to {json_path}")

    report_text = final_report or "[Report content not available]"
    report_path = output_dir / f"{safe_identifier}_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"Saved synthesized report to {report_path}")

    try:
        pdf_payload = generate_pdf_report(log_data)
        if hasattr(pdf_payload, "getvalue"):
            pdf_bytes = pdf_payload.getvalue()
        elif isinstance(pdf_payload, (bytes, bytearray, memoryview)):
            pdf_bytes = bytes(pdf_payload)
        else:
            raise TypeError(f"Unexpected PDF payload type: {type(pdf_payload)!r}")
        pdf_path = output_dir / f"{safe_identifier}_report.pdf"
        pdf_path.write_bytes(pdf_bytes)
        print(f"Saved PDF report to {pdf_path}")
    except Exception as exc:  # pragma: no cover - PDF generation fallback
        print(f"PDF generation failed: {exc}", file=sys.stderr)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="O-QT Assistant CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    ui_parser = subparsers.add_parser("ui", help="Launch the Streamlit user interface")
    ui_parser.add_argument("--", dest="streamlit_args", nargs=argparse.REMAINDER, help="Arguments forwarded to streamlit run")

    analyze = subparsers.add_parser("analyze", help="Run a headless analysis from the command line")
    analyze.add_argument("identifier", help="Chemical identifier")
    analyze.add_argument("--search-type", choices=["name", "cas", "smiles"], default="name")
    analyze.add_argument("--context", help="Analysis context text")
    analyze.add_argument("--api-url", help="QSAR Toolbox API URL (overrides QSAR_TOOLBOX_API_URL)")
    analyze.add_argument("--output-dir", help="Directory to store CLI outputs", default="cli_runs")

    analyze.add_argument("--provider", help="LLM provider override")
    analyze.add_argument("--model-name", help="LLM model display name override")
    analyze.add_argument("--api-key", help="LLM API key override")
    analyze.add_argument("--temperature", type=float, help="LLM temperature override")
    analyze.add_argument("--max-output-tokens", type=int, help="LLM max output token override")

    analyze.add_argument("--simulator", action="append", help="Metabolism simulator GUID to run (can be repeated)")
    analyze.add_argument("--simulator-all", action="store_true", help="Run all available simulators")
    analyze.add_argument("--profiler", action="append", help="Profiler GUID to run (can be repeated)")
    analyze.add_argument("--profiler-all", action="store_true", help="Run all available profilers")
    analyze.add_argument("--qsar-mode", choices=["recommended", "all", "none"], default="recommended", help="Preset for QSAR execution")
    analyze.add_argument("--qsar-guid", action="append", help="Explicit QSAR model GUIDs to run")
    analyze.add_argument("--include-slow-profilers", action="store_true", help="Include profilers flagged as slow")
    analyze.add_argument("--skip-properties", action="store_true", help="Skip fetching calculator properties")
    analyze.add_argument("--skip-experimental", action="store_true", help="Skip experimental data retrieval")
    analyze.add_argument("--skip-profiling", action="store_true", help="Skip profiling data retrieval")
    analyze.add_argument("--skip-qsar", action="store_true", help="Skip QSAR model execution")
    analyze.add_argument("--exclude-adme-tk", action="store_true", help="Exclude ADME/TK experimental records")
    analyze.add_argument("--exclude-mammalian-tox", action="store_true", help="Exclude mammalian toxicity experimental records")

    parser.set_defaults(command="ui")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ui":
        streamlit_args = getattr(args, "streamlit_args", None) or []
        return _run_streamlit(streamlit_args)

    if args.command == "analyze":
        return _run_cli_analysis(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
