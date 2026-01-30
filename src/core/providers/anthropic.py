"""
Anthropic Claude provider implementation.

Features:
- Prompt caching for reduced latency and cost
- Streaming support
"""

import logging
from typing import AsyncGenerator, Optional

import anthropic

from src.core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with prompt caching support."""

    def __init__(self, api_key: str, model: str):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., "claude-sonnet-4-5-20250929")
        """
        self._api_key = api_key
        self._model = model
        self._client: Optional[anthropic.AsyncAnthropic] = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_caching(self) -> bool:
        return True

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client

    def _build_cached_system(self, system_prompt: str) -> list[dict]:
        """Build system prompt with cache control for Anthropic."""
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a complete response using Claude."""
        client = self._get_client()
        cached_system = self._build_cached_system(system_prompt)

        try:
            response = await client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=cached_system,
                messages=[{"role": "user", "content": user_message}],
            )

            # Log cache usage
            if hasattr(response, "usage"):
                usage = response.usage
                cache_read = getattr(usage, "cache_read_input_tokens", 0)
                cache_create = getattr(usage, "cache_creation_input_tokens", 0)
                if cache_read or cache_create:
                    logger.info(f"Anthropic cache - read: {cache_read}, created: {cache_create}")

            return response.content[0].text

        except anthropic.BadRequestError as e:
            # Fallback to non-cached if cache not supported
            logger.warning(f"Anthropic cache not supported, falling back: {e}")
            response = await client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text

    async def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Claude."""
        client = self._get_client()
        cached_system = self._build_cached_system(system_prompt)

        try:
            async with client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=cached_system,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text

                # Log cache usage
                final_message = await stream.get_final_message()
                if hasattr(final_message, "usage"):
                    usage = final_message.usage
                    cache_read = getattr(usage, "cache_read_input_tokens", 0)
                    cache_create = getattr(usage, "cache_creation_input_tokens", 0)
                    if cache_read or cache_create:
                        logger.info(f"Anthropic stream cache - read: {cache_read}, created: {cache_create}")

        except anthropic.BadRequestError as e:
            logger.warning(f"Anthropic stream cache not supported, falling back: {e}")
            async with client.messages.stream(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
