# File: src/qsar_assistant/utils/llm_provider.py
from typing import Optional
import os
import sys

# ---------------------------------------------------------------------------
# Compatibility shim: newer langchain_openai imports token detail classes that
# were added in langchain-core 0.2.x. When running with the 0.1.x series the
# import fails. We defensively provide lightweight stand-ins so that importing
# langchain_openai succeeds during tests (where ChatOpenAI is fully mocked).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - defensive compatibility path
    import langchain_core.messages.ai as _lc_ai

    if not hasattr(_lc_ai, "InputTokenDetails"):
        class InputTokenDetails(dict):
            """Fallback token accounting container."""

        _lc_ai.InputTokenDetails = InputTokenDetails

    if not hasattr(_lc_ai, "OutputTokenDetails"):
        class OutputTokenDetails(dict):
            """Fallback token accounting container."""

        _lc_ai.OutputTokenDetails = OutputTokenDetails

    if not hasattr(_lc_ai, "UsageMetadata"):
        class UsageMetadata(dict):
            """Fallback usage metadata container."""

        _lc_ai.UsageMetadata = UsageMetadata
except Exception:  # pragma: no cover - best-effort shim
    pass

def _is_gpt5_family(model_id: str) -> bool:
    """Check if a model ID belongs to the GPT-5 family (reasoning models)."""
    m = model_id.lower()
    # handle OpenAI and OpenRouter IDs (e.g., "openai/gpt-5-mini")
    return ("gpt-5" in m) and ("-chat" not in m)  # treat gpt-5-chat-like as non-reasoning if you ever add it

def _map_model_id(model: str) -> str:
    """Map unstable/unsupported IDs to stable OpenAI chat models.

    - gpt-4.1, gpt-4.1-mini/nano use the Responses API and may not be supported by ChatOpenAI.
      Map them to gpt-4o / gpt-4o-mini for compatibility.
    - gpt-5 family may be routed elsewhere; leave handling to _is_gpt5_family.
    """
    m = (model or "").lower()
    if m == "gpt-4.1":
        return "gpt-4o"
    if m in ("gpt-4.1-mini", "gpt-4.1-nano"):
        return "gpt-4o-mini"
    return model


def get_llm(provider: str, model: str, temperature: float, max_output_tokens: int, *,
            reasoning_effort: Optional[str] = None, api_key: Optional[str] = None,
            timeout: Optional[float] = None):
    """
    Returns a LangChain-compatible chat model, or None for data-only mode.
    provider: "openai" | "openai-compatible" | "none"
    """
    responses_models = {"gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"}

    if provider == "none":
        return None

    if provider == "openai":
        if model and model.lower() in responses_models:
            from .openai_responses_chat import OpenAIResponsesChat
            return OpenAIResponsesChat(
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                api_key=api_key,
                timeout=timeout,
            )
        # Normalize model IDs to stable chat-completions compatible ones
        model = _map_model_id(model)
        from langchain_openai import ChatOpenAI
        if _is_gpt5_family(model):
            # GPT‑5 series (reasoning): use max_completion_tokens; do not send temperature
            model_kwargs = {"max_completion_tokens": max_output_tokens}
            if reasoning_effort:
                model_kwargs["reasoning"] = {"effort": reasoning_effort}
            return ChatOpenAI(model=model, model_kwargs=model_kwargs, api_key=api_key)
        else:
            return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_output_tokens, api_key=api_key)

    # OpenAI-compatible (e.g., local vLLM/TGI with an OAI surface)
    if provider == "openai-compatible":
        from langchain_openai import ChatOpenAI
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")  # some local stacks ignore key
        # Normalize model IDs
        model = _map_model_id(model)
        if _is_gpt5_family(model):
            model_kwargs = {"max_completion_tokens": max_output_tokens}
            if reasoning_effort:
                model_kwargs["reasoning"] = {"effort": reasoning_effort}
            return ChatOpenAI(model=model, model_kwargs=model_kwargs, base_url=base_url, api_key=api_key)
        else:
            return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_output_tokens,
                              base_url=base_url, api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider}")
