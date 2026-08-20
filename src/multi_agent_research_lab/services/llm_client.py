"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client with OpenRouter and OpenAI support."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            api_key = self.settings.effective_api_key
            if not api_key:
                raise ValueError(
                    "No API key found. Please set OPENROUTER_API_KEY (or OPENAI_API_KEY) in .env"
                )

            base_url = self.settings.effective_base_url
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion from OpenRouter/OpenAI with retry and timeout."""
        client = self._get_client()
        model = self.settings.effective_model

        @retry(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
        )
        def _call_api() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.settings.llm_max_tokens,
                timeout=self.settings.timeout_seconds,
            )

        response = _call_api()
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = getattr(response, "usage", None)

        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
