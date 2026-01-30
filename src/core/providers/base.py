"""
Base class for LLM providers.

Defines the interface that all provider implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional


@dataclass
class TokenUsage:
    """Token usage statistics from LLM response."""
    
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0  # Anthropic-specific
    cache_creation_tokens: int = 0  # Anthropic-specific
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class GenerationResult:
    """Result from a generation request, including text and usage."""
    
    text: str
    usage: Optional[TokenUsage] = None


@dataclass
class StreamResult:
    """Final result after streaming, with accumulated usage."""
    
    usage: Optional[TokenUsage] = None


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
    ) -> GenerationResult:
        """
        Generate a complete response.

        Args:
            system_prompt: The system prompt to use
            user_message: The user's message/query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with text and token usage
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str | StreamResult, None]:
        """
        Generate a streaming response.

        Args:
            system_prompt: The system prompt to use
            user_message: The user's message/query
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Text chunks as they are generated, then StreamResult with usage at end
        """
        ...

