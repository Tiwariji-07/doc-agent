"""
LLM Provider abstraction layer.

Provides a unified interface for multiple AI providers:
- Anthropic (Claude)
- OpenAI (GPT)
- Ollama (local models)
"""

from src.core.providers.base import BaseLLMProvider
from src.core.providers.factory import get_llm_provider

__all__ = ["BaseLLMProvider", "get_llm_provider"]
