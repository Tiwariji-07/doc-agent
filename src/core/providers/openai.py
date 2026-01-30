"""
OpenAI provider implementation.

Supports GPT-4, GPT-4o, and other OpenAI models.
"""

import logging
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI

from src.core.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, api_key: str, model: str):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4o")
        """
        self._api_key = api_key
        self._model = model
        self._client: Optional[AsyncOpenAI] = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_caching(self) -> bool:
        return False  # OpenAI doesn't have explicit prompt caching like Anthropic

    def _get_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a complete response using OpenAI."""
        client = self._get_client()

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

    async def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response using OpenAI."""
        client = self._get_client()

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
