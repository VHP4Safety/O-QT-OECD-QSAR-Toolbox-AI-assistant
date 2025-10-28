"""Common test configuration and fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

# Ensure ``src`` is importable when running tests from the repository root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Compatibility shim for older langchain-core / langchain-openai combinations.
try:  # pragma: no cover - defensive test helper
    import langchain_core.messages.ai as _lc_ai  # type: ignore

    if not hasattr(_lc_ai, "InputTokenDetails"):
        class InputTokenDetails(dict):
            pass

        _lc_ai.InputTokenDetails = InputTokenDetails

    if not hasattr(_lc_ai, "OutputTokenDetails"):
        class OutputTokenDetails(dict):
            pass

        _lc_ai.OutputTokenDetails = OutputTokenDetails

    if not hasattr(_lc_ai, "UsageMetadata"):
        class UsageMetadata(dict):
            pass

        _lc_ai.UsageMetadata = UsageMetadata
except Exception:
    pass


class _SessionState(dict):
    """Lightweight stand-in for Streamlit's SessionState."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, item: str) -> None:
        try:
            del self[item]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise AttributeError(item) from exc


class _ProgressBar:
    """Minimal progress bar supporting ``progress`` and ``empty``."""

    def __init__(self, value: float = 0.0, text: str | None = None) -> None:
        self.value = value
        self.text = text

    def progress(self, value: float, text: str | None = None) -> "_ProgressBar":
        self.value = value
        self.text = text
        return self

    def empty(self) -> None:
        self.value = 0.0
        self.text = None


class _Sidebar(SimpleNamespace):
    """Subset of sidebar helpers used in tests."""

    def markdown(self, *_args, **_kwargs) -> None:
        return None

    def subheader(self, *_args, **_kwargs) -> None:
        return None

    def download_button(self, *_args, **_kwargs) -> None:
        return None

    def info(self, *_args, **_kwargs) -> None:
        return None

    def error(self, *_args, **_kwargs) -> None:
        return None

    def button(self, *_args, **_kwargs) -> bool:
        return False

    def text_input(self, *_args, value: str = "", **_kwargs) -> str:
        return value

    def selectbox(self, *_args, options: list[str], index: int = 0, **_kwargs) -> str:
        return options[index]

    def number_input(
        self,
        *_args,
        value: int | float = 0,
        **_kwargs,
    ) -> int | float:
        return value

    def expander(self, *_args, **_kwargs):
        class _Context:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_exc) -> bool:
                return False

            def number_input(self_inner, *_args, value: int | float = 0, **_kwargs):
                return value

            def markdown(self_inner, *_args, **_kwargs) -> None:
                return None

            def caption(self_inner, *_args, **_kwargs) -> None:
                return None

        return _Context()


class _StreamlitStub:
    """Subset of Streamlit APIs exercised by the test-suite."""

    class errors:
        StreamlitAPIException = Exception

    def __init__(self) -> None:
        self.session_state = _SessionState()
        self.sidebar = _Sidebar()
        self._callbacks: list[Callable[..., None]] = []

    # Basic UI primitives -------------------------------------------------
    def header(self, *_args, **_kwargs) -> None:
        return None

    def subheader(self, *_args, **_kwargs) -> None:
        return None

    def markdown(self, *_args, **_kwargs) -> None:
        return None

    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None

    def error(self, *_args, **_kwargs) -> None:
        return None

    def download_button(self, *_args, **_kwargs) -> None:
        return None

    # Progress handling ---------------------------------------------------
    def progress(self, value: float = 0.0, text: str | None = None) -> _ProgressBar:
        return _ProgressBar(value, text)

    # Misc helpers --------------------------------------------------------
    def rerun(self) -> None:
        return None


@pytest.fixture(autouse=True)
def streamlit_stub(monkeypatch: pytest.MonkeyPatch):
    """Provide a deterministic Streamlit facade for unit tests."""
    import oqt_assistant.app as app_module

    stub = _StreamlitStub()
    monkeypatch.setattr(app_module, "st", stub)
    yield stub
