import os
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils import get_from_dict_or_env
from openai import AsyncOpenAI, OpenAI


def _message_content_to_text(message: BaseMessage) -> str:
    """Normalize LangChain message content into a plain string."""
    content = message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _message_role(message: BaseMessage) -> str:
    """Map LangChain message types to OpenAI Responses roles."""
    msg_type = getattr(message, "type", "").lower()
    if msg_type == "system":
        return "system"
    if msg_type in ("human", "user"):
        return "user"
    if msg_type in ("ai", "assistant"):
        return "assistant"
    if msg_type == "tool":
        return "tool"
    return msg_type or "user"


class OpenAIResponsesChat(BaseChatModel):
    """Minimal LangChain-compatible wrapper around the OpenAI Responses API."""

    model: str
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    api_key: Optional[str] = None
    timeout: Optional[float] = None
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    _client: OpenAI
    _aclient: AsyncOpenAI

    def model_post_init(self, _context: Any) -> None:
        resolved_api_key = get_from_dict_or_env(
            {"api_key": self.api_key or os.getenv("OPENAI_API_KEY")},
            "api_key",
            "OPENAI_API_KEY",
        )
        client_options: Dict[str, Any] = {"api_key": resolved_api_key}
        if self.timeout is not None:
            client_options["timeout"] = self.timeout

        object.__setattr__(self, "_client", OpenAI(**client_options))
        object.__setattr__(self, "_aclient", AsyncOpenAI(**client_options))
        object.__setattr__(self, "api_key", resolved_api_key)
        object.__setattr__(self, "request_timeout", self.timeout)
        object.__setattr__(self, "max_retries", 2)

    @property
    def _llm_type(self) -> str:
        return "openai-responses"

    # ---- Helpers -----------------------------------------------------------------
    def _build_payload(
        self,
        messages: List[BaseMessage],
        *,
        stop: Optional[List[str]] = None,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [],
        }

        if self.temperature is not None:
            payload["temperature"] = self.temperature

        if self.top_p is not None:
            payload["top_p"] = self.top_p

        if self.max_output_tokens is not None:
            payload["max_output_tokens"] = self.max_output_tokens

        if stop:
            payload["stop"] = stop

        for message in messages:
            text = _message_content_to_text(message).strip()
            if not text:
                continue

            role = _message_role(message)
            content_type = "input_text"
            if role == "assistant":
                content_type = "output_text"

            payload["input"].append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": content_type,
                            "text": text,
                        }
                    ],
                }
            )

        if extra_kwargs:
            payload.update(extra_kwargs)

        return payload

    def _to_chat_result(self, response) -> ChatResult:
        output_text = getattr(response, "output_text", None) or ""
        ai_message = AIMessage(
            content=output_text,
            additional_kwargs={
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
            },
        )
        generation = ChatGeneration(
            message=ai_message,
            text=output_text,
            generation_info={
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
            },
        )

        usage = getattr(response, "usage", None)
        token_usage: Dict[str, Any] = {}
        if usage is not None:
            token_usage = {
                "prompt_tokens": getattr(usage, "input_tokens", None),
                "completion_tokens": getattr(usage, "output_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return ChatResult(
            generations=[generation],
            llm_output={
                "token_usage": token_usage,
                "response_id": getattr(response, "id", None),
                "model": getattr(response, "model", None),
            },
        )

    # ---- BaseChatModel overrides --------------------------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop=stop, extra_kwargs=kwargs)
        response = self._client.responses.create(**payload)
        return self._to_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop=stop, extra_kwargs=kwargs)
        response = await self._aclient.responses.create(**payload)
        return self._to_chat_result(response)

    # ---- Convenience methods ------------------------------------------------------
    def with_options(self, **options: Any) -> "OpenAIResponsesChat":
        """Support LangChain's `.with_options()` API."""
        combined = {
            "model": self.model,
            "temperature": options.get("temperature", self.temperature),
            "max_output_tokens": options.get(
                "max_output_tokens", self.max_output_tokens
            ),
            "top_p": options.get("top_p", self.top_p),
            "api_key": options.get("api_key", self.api_key),
            "timeout": options.get("timeout", self.timeout),
        }
        return OpenAIResponsesChat(**combined)
