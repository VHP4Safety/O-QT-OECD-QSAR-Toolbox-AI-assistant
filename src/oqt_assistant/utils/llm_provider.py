# File: src/qsar_assistant/utils/llm_provider.py
from typing import Optional
import os

def get_llm(provider: str, model: str, temperature: float, max_output_tokens: int):
    """
    Returns a LangChain-compatible chat model, or None for data-only mode.
    provider: "openai" | "openai-compatible" | "none"
    """
    if provider == "none":
        return None

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
        )

    # OpenAI-compatible (e.g., local vLLM/TGI with an OAI surface)
    if provider == "openai-compatible":
        from langchain_openai import ChatOpenAI
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")  # some local stacks ignore key
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            base_url=base_url,
            api_key=api_key,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")