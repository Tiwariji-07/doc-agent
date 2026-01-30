"""
Pricing configuration for LLM providers.

Contains token pricing per model for cost calculation.
Prices are in USD per 1 million tokens.
"""

from typing import Optional

# Pricing per 1M tokens (USD)
# Updated: January 2025
PRICING = {
    "anthropic": {
        "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    "ollama": {
        # Ollama is free/local, no API costs
        "*": {"input": 0.0, "output": 0.0},
    },
}


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> float:
    """
    Calculate the cost for a query based on token usage.
    
    Args:
        provider: The LLM provider (anthropic, openai, ollama)
        model: The model name
        input_tokens: Number of input tokens (can be None)
        output_tokens: Number of output tokens (can be None)
        
    Returns:
        Cost in USD (0.0 if tokens are None or model not found)
    """
    if input_tokens is None and output_tokens is None:
        return 0.0
    
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    
    provider_pricing = PRICING.get(provider, {})
    
    # Check for model-specific pricing, fallback to wildcard
    model_pricing = provider_pricing.get(model) or provider_pricing.get("*")
    
    if not model_pricing:
        return 0.0
    
    # Price per 1M tokens, convert to actual cost
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    
    return round(input_cost + output_cost, 6)


def get_model_pricing(provider: str, model: str) -> dict:
    """Get pricing info for a specific model."""
    provider_pricing = PRICING.get(provider, {})
    return provider_pricing.get(model) or provider_pricing.get("*") or {"input": 0, "output": 0}
