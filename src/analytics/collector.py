"""
Fire-and-forget event collector.

Pushes events to Redis queue for background processing.
"""

import json
import logging
from typing import Optional

import redis.asyncio as redis

from src.config.settings import get_settings
from src.analytics.models import QueryEvent, FeedbackEvent, AnalyticsOverview

logger = logging.getLogger(__name__)

# Singleton Redis client
_redis_client: Optional[redis.Redis] = None


async def _get_redis() -> redis.Redis:
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(settings.redis_url)
    return _redis_client


async def track_query(event: QueryEvent) -> None:
    """
    Track a query event (fire-and-forget).
    
    Pushes to Redis queue for background processing.
    Does not block or raise exceptions to caller.
    """
    settings = get_settings()
    if not settings.analytics_enabled:
        return
    
    try:
        client = await _get_redis()
        event_data = event.model_dump_json()
        await client.lpush(f"{settings.analytics_redis_key}:queries", event_data)
        logger.debug(f"Tracked query event: {event.id}")
    except Exception as e:
        # Never fail the main request due to analytics
        logger.warning(f"Failed to track query event: {e}")


async def track_feedback(event: FeedbackEvent) -> None:
    """
    Track a feedback event (fire-and-forget).
    
    Pushes to Redis queue for background processing.
    """
    settings = get_settings()
    if not settings.analytics_enabled:
        return
    
    try:
        client = await _get_redis()
        event_data = event.model_dump_json()
        await client.lpush(f"{settings.analytics_redis_key}:feedback", event_data)
        logger.debug(f"Tracked feedback event: {event.id}")
    except Exception as e:
        logger.warning(f"Failed to track feedback event: {e}")


async def get_analytics() -> "Analytics":
    """Get analytics instance for queries."""
    return Analytics()


class Analytics:
    """Analytics query interface."""
    
    async def get_overview(self, period: str = "today") -> AnalyticsOverview:
        """Get dashboard overview stats."""
        from src.analytics.queries import get_overview_stats
        return await get_overview_stats(period)
    
    async def get_recent_queries(self, limit: int = 50) -> list[dict]:
        """Get recent queries."""
        from src.analytics.queries import get_recent_queries
        return await get_recent_queries(limit)
    
    async def get_pending_feedback(self, limit: int = 50) -> list[dict]:
        """Get feedback needing review."""
        from src.analytics.queries import get_pending_feedback
        return await get_pending_feedback(limit)
