"""
Base class for LLM providers.

Defines the interface that all provider implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        ...

    @property
    @abstractmethod
    def supports_caching(self) -> bool:
        """Return whether this provider supports prompt caching."""
        ...

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a complete response.

        Args:
            system_prompt: The system prompt to use
            user_message: The user's message/query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            The generated text response
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response.

        Args:
            system_prompt: The system prompt to use
            user_message: The user's message/query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they are generated
        """
        ...
