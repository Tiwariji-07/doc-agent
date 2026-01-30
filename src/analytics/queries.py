"""
Dashboard query functions for analytics.

Provides stats and data retrieval from SQLite.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from src.analytics.db import get_db
from src.analytics.models import AnalyticsOverview
from src.analytics.pricing import calculate_cost

logger = logging.getLogger(__name__)


def _get_date_filter(period: str) -> tuple[str, str]:
    """Get date range for period."""
    now = datetime.utcnow()
    
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now - timedelta(days=7)
    elif period == "month":
        start = now - timedelta(days=30)
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    return start.isoformat(), now.isoformat()


async def get_overview_stats(period: str = "today") -> AnalyticsOverview:
    """Get dashboard overview statistics."""
    db = await get_db()
    conn = await db.get_connection()
    
    start_date, end_date = _get_date_filter(period)
    
    # Total queries
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM query_events WHERE timestamp >= ? AND timestamp <= ?",
        (start_date, end_date),
    )
    total_queries = (await cursor.fetchone())[0]
    
    # Feedback stats
    cursor = await conn.execute(
        """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN helpful = 1 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN helpful = 0 THEN 1 ELSE 0 END) as negative
        FROM feedback 
        WHERE timestamp >= ? AND timestamp <= ?
        """,
        (start_date, end_date),
    )
    feedback_row = await cursor.fetchone()
    total_feedback = feedback_row[0] or 0
    positive_feedback = feedback_row[1] or 0
    negative_feedback = feedback_row[2] or 0
    feedback_score = positive_feedback / max(total_feedback, 1)
    
    # Performance stats
    cursor = await conn.execute(
        """
        SELECT 
            AVG(response_time_ms) as avg_time,
            COUNT(CASE WHEN cache_hit = 1 THEN 1 END) as cache_hits,
            COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as errors,
            COUNT(*) as total
        FROM query_events
        WHERE timestamp >= ? AND timestamp <= ?
        """,
        (start_date, end_date),
    )
    perf_row = await cursor.fetchone()
    avg_response_time = perf_row[0] or 0
    cache_hits = perf_row[1] or 0
    errors = perf_row[2] or 0
    total_for_rates = perf_row[3] or 1  # Avoid division by zero
    cache_hit_rate = cache_hits / total_for_rates
    error_rate = errors / total_for_rates
    
    # Provider distribution
    cursor = await conn.execute(
        """
        SELECT provider, COUNT(*) as count
        FROM query_events
        WHERE timestamp >= ? AND timestamp <= ?
        GROUP BY provider
        """,
        (start_date, end_date),
    )
    provider_rows = await cursor.fetchall()
    provider_distribution = {row[0]: row[1] for row in provider_rows}
    
    # Top queries
    cursor = await conn.execute(
        """
        SELECT query, COUNT(*) as count
        FROM query_events
        WHERE timestamp >= ? AND timestamp <= ?
        GROUP BY query
        ORDER BY count DESC
        LIMIT 10
        """,
        (start_date, end_date),
    )
    top_query_rows = await cursor.fetchall()
    top_queries = [{"query": row[0], "count": row[1]} for row in top_query_rows]
    
    # Pending review count
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE helpful = 0 AND reviewed = 0"
    )
    pending_review = (await cursor.fetchone())[0]
    
    # Token and cost stats (with breakdown by model)
    cursor = await conn.execute(
        """
        SELECT 
            SUM(COALESCE(tokens_input, 0)) as total_input,
            SUM(COALESCE(tokens_output, 0)) as total_output,
            COUNT(*) as query_count,
            provider, model
        FROM query_events
        WHERE timestamp >= ? AND timestamp <= ?
        GROUP BY provider, model
        """,
        (start_date, end_date),
    )
    token_rows = await cursor.fetchall()
    
    total_tokens_input = 0
    total_tokens_output = 0
    total_cost = 0.0
    cost_by_model = []
    
    for row in token_rows:
        input_tokens = row[0] or 0
        output_tokens = row[1] or 0
        query_count = row[2] or 0
        provider = row[3]
        model = row[4]
        
        total_tokens_input += input_tokens
        total_tokens_output += output_tokens
        model_cost = calculate_cost(provider, model, input_tokens, output_tokens)
        total_cost += model_cost
        
        if model_cost > 0 or query_count > 0:
            cost_by_model.append({
                "model": model,
                "provider": provider,
                "cost": round(model_cost, 4),
                "queries": query_count,
                "tokens_input": input_tokens,
                "tokens_output": output_tokens,
            })
    
    # Sort by cost descending
    cost_by_model.sort(key=lambda x: x["cost"], reverse=True)
    
    # Calculate avg cost per non-cached query for savings estimate
    cache_miss_count = total_for_rates - cache_hits
    avg_cost_per_query = total_cost / max(cache_miss_count, 1) if cache_miss_count > 0 else 0
    cache_savings = cache_hits * avg_cost_per_query
    
    # Response time trend (hourly buckets)
    cursor = await conn.execute(
        """
        SELECT 
            strftime('%H:00', timestamp) as hour,
            AVG(response_time_ms) as avg_ms,
            COUNT(*) as count
        FROM query_events
        WHERE timestamp >= ? AND timestamp <= ?
        GROUP BY strftime('%H', timestamp)
        ORDER BY hour
        """,
        (start_date, end_date),
    )
    trend_rows = await cursor.fetchall()
    response_time_trend = [
        {"hour": row[0], "avg_ms": round(row[1] or 0), "count": row[2]}
        for row in trend_rows
    ]
    
    return AnalyticsOverview(
        period=period,
        total_queries=total_queries,
        total_feedback=total_feedback,
        positive_feedback=positive_feedback,
        negative_feedback=negative_feedback,
        feedback_score=feedback_score,
        avg_response_time_ms=avg_response_time,
        cache_hit_rate=cache_hit_rate,
        error_rate=error_rate,
        cache_hit_count=cache_hits,
        cache_miss_count=cache_miss_count,
        cache_savings_usd=round(cache_savings, 4),
        response_time_trend=response_time_trend,
        provider_distribution=provider_distribution,
        top_queries=top_queries,
        pending_review_count=pending_review,
        total_tokens_input=total_tokens_input,
        total_tokens_output=total_tokens_output,
        total_cost_usd=round(total_cost, 4),
        cost_by_model=cost_by_model,
    )


async def get_recent_queries(limit: int = 50) -> list[dict]:
    """Get recent query events."""
    db = await get_db()
    conn = await db.get_connection()
    
    cursor = await conn.execute(
        """
        SELECT id, timestamp, query, provider, model, response_time_ms, 
               sources_count, cache_hit, error
        FROM query_events
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )
    
    rows = await cursor.fetchall()
    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "query": row[2],
            "provider": row[3],
            "model": row[4],
            "response_time_ms": row[5],
            "sources_count": row[6],
            "cache_hit": bool(row[7]),
            "error": row[8],
        }
        for row in rows
    ]


async def get_pending_feedback(limit: int = 50) -> list[dict]:
    """Get feedback entries needing review."""
    db = await get_db()
    conn = await db.get_connection()
    
    cursor = await conn.execute(
        """
        SELECT f.id, f.timestamp, f.query_id, f.helpful, f.comment,
               q.query, q.provider
        FROM feedback f
        LEFT JOIN query_events q ON f.query_id = q.id
        WHERE f.reviewed = 0
        ORDER BY f.timestamp DESC
        LIMIT ?
        """,
        (limit,),
    )
    
    rows = await cursor.fetchall()
    return [
        {
            "id": row[0],
            "timestamp": row[1],
            "query_id": row[2],
            "helpful": bool(row[3]),
            "comment": row[4],
            "query": row[5],
            "provider": row[6],
        }
        for row in rows
    ]


async def mark_feedback_reviewed(feedback_id: str) -> bool:
    """Mark feedback as reviewed."""
    db = await get_db()
    conn = await db.get_connection()
    
    await conn.execute(
        "UPDATE feedback SET reviewed = 1 WHERE id = ?",
        (feedback_id,),
    )
    await conn.commit()
    return True
