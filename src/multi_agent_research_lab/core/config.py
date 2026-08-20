"""Application configuration.

Keep config small and explicit. Do not read environment variables directly in agents.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    openrouter_api_key: str | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL"
    )
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini", validation_alias="OPENROUTER_MODEL"
    )

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="multi-agent-research-lab", validation_alias="LANGSMITH_PROJECT"
    )

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    llm_max_tokens: int = Field(default=2000, ge=100, le=32000, validation_alias="LLM_MAX_TOKENS")
    max_iterations: int = Field(default=6, ge=1, le=20, validation_alias="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=60, ge=5, le=600, validation_alias="TIMEOUT_SECONDS")

    @property
    def effective_api_key(self) -> str | None:
        """Return the active API key (prefers OpenRouter, falls back to OpenAI)."""
        return self.openrouter_api_key or self.openai_api_key

    @property
    def effective_base_url(self) -> str:
        """Return the active base URL (OpenRouter or custom OpenAI base URL)."""
        if self.openrouter_api_key:
            return self.openrouter_base_url
        if self.openai_base_url:
            return self.openai_base_url
        # If openrouter_api_key is not set but openrouter_model is configured,
        # default to OpenRouter base url
        return self.openrouter_base_url

    @property
    def effective_model(self) -> str:
        """Return the active model name."""
        if self.openrouter_api_key:
            return self.openrouter_model
        if self.openai_api_key and not self.openrouter_api_key:
            return self.openai_model
        return self.openrouter_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
