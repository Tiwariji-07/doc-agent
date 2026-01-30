"""
LLM Provider factory.

Creates the appropriate provider based on configuration.
"""

import logging

from src.config.settings import Settings, get_settings
from src.core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def get_llm_provider(settings: Settings | None = None) -> BaseLLMProvider:
    """
    Create an LLM provider based on settings.

    Args:
        settings: Optional settings instance. Uses get_settings() if not provided.

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider is unknown or required config is missing
    """
    if settings is None:
        settings = get_settings()

    provider = settings.ai_provider.lower()
    logger.info(f"Initializing LLM provider: {provider}")

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")

        from src.core.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
        )

    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")

        from src.core.providers.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )

    elif provider == "ollama":
        from src.core.providers.ollama import OllamaProvider

        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    else:
        raise ValueError(
            f"Unknown AI provider: {provider}. "
            "Supported providers: anthropic, openai, ollama"
        )
