"""
Analytics module for docs-agent.

Provides non-blocking event tracking with Redis queue and Postgres storage.
"""

from src.analytics.collector import track_query, track_feedback, get_analytics
from src.analytics.models import QueryEvent, FeedbackEvent, AnalyticsOverview
from src.analytics.pricing import calculate_cost, PRICING

__all__ = [
    "track_query",
    "track_feedback",
    "get_analytics",
    "QueryEvent",
    "FeedbackEvent",
    "AnalyticsOverview",
    "calculate_cost",
    "PRICING",
]
