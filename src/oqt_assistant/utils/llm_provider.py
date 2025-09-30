# File: src/qsar_assistant/utils/llm_provider.py
from typing import Optional
import os

def _is_gpt5_family(model_id: str) -> bool:
    """Check if a model ID belongs to the GPT-5 family (reasoning models)."""
    m = model_id.lower()
    # handle OpenAI and OpenRouter IDs (e.g., "openai/gpt-5-mini")
    return ("gpt-5" in m) and ("-chat" not in m)  # treat gpt-5-chat-like as non-reasoning if you ever add it

def get_llm(provider: str, model: str, temperature: float, max_output_tokens: int, *,
            reasoning_effort: Optional[str] = None):
    """
    Returns a LangChain-compatible chat model, or None for data-only mode.
    provider: "openai" | "openai-compatible" | "none"
    """
    if provider == "none":
        return None

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if _is_gpt5_family(model):
            # GPT‑5 series (reasoning): use max_completion_tokens; do not send temperature
            model_kwargs = {"max_completion_tokens": max_output_tokens}
            if reasoning_effort:
                model_kwargs["reasoning"] = {"effort": reasoning_effort}
            return ChatOpenAI(model=model, model_kwargs=model_kwargs)
        else:
            return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_output_tokens)

    # OpenAI-compatible (e.g., local vLLM/TGI with an OAI surface)
    if provider == "openai-compatible":
        from langchain_openai import ChatOpenAI
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")  # some local stacks ignore key
        if _is_gpt5_family(model):
            model_kwargs = {"max_completion_tokens": max_output_tokens}
            if reasoning_effort:
                model_kwargs["reasoning"] = {"effort": reasoning_effort}
            return ChatOpenAI(model=model, model_kwargs=model_kwargs, base_url=base_url, api_key=api_key)
        else:
            return ChatOpenAI(model=model, temperature=temperature, max_tokens=max_output_tokens,
                              base_url=base_url, api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: {provider}")