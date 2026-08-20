from multi_agent_research_lab.core.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.openai_model
    assert settings.openrouter_model
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.max_iterations >= 1


def test_settings_openrouter_precedence() -> None:
    settings = Settings(
        openrouter_api_key="sk-or-test",
        openrouter_model="google/gemini-2.0-flash-001",
        openai_api_key="sk-openai-test",
        openai_model="gpt-4o",
    )
    assert settings.effective_api_key == "sk-or-test"
    assert settings.effective_model == "google/gemini-2.0-flash-001"
    assert settings.effective_base_url == "https://openrouter.ai/api/v1"


def test_settings_openai_fallback() -> None:
    settings = Settings(
        openrouter_api_key=None,
        openai_api_key="sk-openai-test",
        openai_model="gpt-4o",
    )
    assert settings.effective_api_key == "sk-openai-test"
    assert settings.effective_model == "gpt-4o"
