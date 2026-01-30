"""
Pydantic models for analytics events.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class QueryEvent(BaseModel):
    """Event tracked for each user query."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Query info
    query: str
    session_id: Optional[str] = None
    
    # Provider info
    provider: str  # anthropic, openai, ollama
    model: str
    
    # Performance
    response_time_ms: Optional[int] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    
    # Results
    sources_count: int = 0
    cache_hit: bool = False
    
    # Error tracking
    error: Optional[str] = None


class FeedbackEvent(BaseModel):
    """User feedback on a response."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    query_id: str  # Links to QueryEvent.id
    helpful: bool  # True = 👍, False = 👎
    comment: Optional[str] = None


class AnalyticsOverview(BaseModel):
    """Dashboard overview stats."""
    
    # Time range
    period: str  # "today", "week", "month"
    
    # Volume
    total_queries: int
    total_feedback: int
    
    # Quality
    positive_feedback: int
    negative_feedback: int
    feedback_score: float  # 0.0 to 1.0
    
    # Performance
    avg_response_time_ms: float
    cache_hit_rate: float
    error_rate: float
    
    # Cache stats
    cache_hit_count: int = 0
    cache_miss_count: int = 0
    cache_savings_usd: float = 0.0  # Estimated savings from cache hits
    
    # Token usage
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0
    
    # Cost breakdown by model
    cost_by_model: list[dict] = []  # [{"model": "gpt-4o", "cost": 1.25, "queries": 50}]
    
    # Response time trend (hourly buckets)
    response_time_trend: list[dict] = []  # [{"hour": "10:00", "avg_ms": 1200}]
    
    # Provider breakdown
    provider_distribution: dict[str, int]  # {"anthropic": 100, "openai": 50}
    
    # Top queries
    top_queries: list[dict]  # [{"query": "...", "count": 10}]
    
    # Pending review
    pending_review_count: int

