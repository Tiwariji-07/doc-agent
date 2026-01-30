"""
Ollama provider implementation.

Uses OpenAI-compatible API for local Ollama models.
Ollama provides an OpenAI-compatible endpoint at /v1/chat/completions.
"""

import logging
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI

from src.core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama provider using OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str):
        """
        Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL (e.g., "http://localhost:11434")
            model: Model name (e.g., "llama3.2")
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client: Optional[AsyncOpenAI] = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_caching(self) -> bool:
        return False

    def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI-compatible client for Ollama."""
        if self._client is None:
            # Ollama exposes OpenAI-compatible API at /v1
            self._client = AsyncOpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="ollama",  # Ollama doesn't require auth, but client needs a value
            )
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a complete response using Ollama."""
        client = self._get_client()

        try:
            response = await client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise RuntimeError(f"Ollama error: {e}. Is Ollama running at {self._base_url}?")

    async def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using Ollama."""
        client = self._get_client()

        try:
            stream = await client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise RuntimeError(f"Ollama error: {e}. Is Ollama running at {self._base_url}?")
