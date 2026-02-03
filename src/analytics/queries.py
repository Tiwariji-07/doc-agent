"""
Dashboard query functions for analytics backed by Postgres.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.analytics.db import get_db
from src.analytics.models import AnalyticsOverview
from src.analytics.pricing import calculate_cost

logger = logging.getLogger(__name__)


def _get_date_filter(period: str) -> tuple[datetime, datetime]:
    """Get date range (UTC) for the requested period."""
    now = datetime.now(timezone.utc)

    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        start = now - timedelta(days=7)
    elif period == "month":
        start = now - timedelta(days=30)
    else:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    return start, now


async def get_overview_stats(period: str = "today") -> AnalyticsOverview:
    """Get dashboard overview statistics."""
    db = await get_db()
    start_date, end_date = _get_date_filter(period)

    async with db.connection() as conn:
        total_queries = await conn.fetchval(
            """
            SELECT COUNT(*)
            FROM query_events
            WHERE timestamp BETWEEN $1 AND $2
            """,
            start_date,
            end_date,
        )

        feedback_row = await conn.fetchrow(
            """
            SELECT 
                COUNT(*) AS total,
                SUM(CASE WHEN helpful THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN NOT helpful THEN 1 ELSE 0 END) AS negative
            FROM feedback
            WHERE timestamp BETWEEN $1 AND $2
            """,
            start_date,
            end_date,
        )
        if feedback_row:
            total_feedback = feedback_row["total"] or 0
            positive_feedback = feedback_row["positive"] or 0
            negative_feedback = feedback_row["negative"] or 0
        else:
            total_feedback = 0
            positive_feedback = 0
            negative_feedback = 0
        feedback_score = positive_feedback / max(total_feedback, 1)

        perf_row = await conn.fetchrow(
            """
            SELECT
                AVG(response_time_ms) AS avg_time,
                SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) AS cache_hits,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                COUNT(*) AS total
            FROM query_events
            WHERE timestamp BETWEEN $1 AND $2
            """,
            start_date,
            end_date,
        )

        if perf_row:
            avg_response_time = perf_row["avg_time"] or 0
            cache_hits = perf_row["cache_hits"] or 0
            errors = perf_row["errors"] or 0
            total_for_rates = perf_row["total"] or 0
        else:
            avg_response_time = 0
            cache_hits = 0
            errors = 0
            total_for_rates = 0
        cache_hit_rate = cache_hits / total_for_rates if total_for_rates else 0
        error_rate = errors / total_for_rates if total_for_rates else 0

        provider_rows = await conn.fetch(
            """
            SELECT provider, COUNT(*) AS count
            FROM query_events
            WHERE timestamp BETWEEN $1 AND $2
            GROUP BY provider
            """,
            start_date,
            end_date,
        )
        provider_distribution = {row["provider"]: row["count"] for row in provider_rows}

        top_query_rows = await conn.fetch(
            """
            SELECT query, COUNT(*) AS count
            FROM query_events
            WHERE timestamp BETWEEN $1 AND $2
            GROUP BY query
            ORDER BY count DESC
            LIMIT 10
            """,
            start_date,
            end_date,
        )
        top_queries = [{"query": row["query"], "count": row["count"]} for row in top_query_rows]

        pending_review = await conn.fetchval(
            "SELECT COUNT(*) FROM feedback WHERE helpful = FALSE AND reviewed = FALSE"
        ) or 0

        token_rows = await conn.fetch(
            """
            SELECT 
                SUM(COALESCE(tokens_input, 0)) AS total_input,
                SUM(COALESCE(tokens_output, 0)) AS total_output,
                COUNT(*) AS query_count,
                provider,
                model
            FROM query_events
            WHERE timestamp BETWEEN $1 AND $2
            GROUP BY provider, model
            """,
            start_date,
            end_date,
        )

        trend_rows = await conn.fetch(
            """
            SELECT 
                date_trunc('hour', timestamp) AS hour_bucket,
                AVG(response_time_ms) AS avg_ms,
                COUNT(*) AS count
            FROM query_events
            WHERE timestamp BETWEEN $1 AND $2
            GROUP BY hour_bucket
            ORDER BY hour_bucket
            """,
            start_date,
            end_date,
        )

    total_tokens_input = 0
    total_tokens_output = 0
    total_cost = 0.0
    cost_by_model: list[dict[str, Any]] = []

    for row in token_rows:
        input_tokens = row["total_input"] or 0
        output_tokens = row["total_output"] or 0
        query_count = row["query_count"] or 0
        provider = row["provider"]
        model = row["model"]

        total_tokens_input += input_tokens
        total_tokens_output += output_tokens

        model_cost = calculate_cost(provider, model, input_tokens, output_tokens)
        total_cost += model_cost

        if model_cost > 0 or query_count > 0:
            cost_by_model.append(
                {
                    "model": model,
                    "provider": provider,
                    "cost": round(model_cost, 4),
                    "queries": query_count,
                    "tokens_input": input_tokens,
                    "tokens_output": output_tokens,
                }
            )

    cost_by_model.sort(key=lambda x: x["cost"], reverse=True)

    cache_miss_count = total_for_rates - cache_hits if total_for_rates else 0
    avg_cost_per_query = (
        total_cost / cache_miss_count if cache_miss_count else 0
    )
    cache_savings = cache_hits * avg_cost_per_query

    response_time_trend = [
        {
            "hour": row["hour_bucket"].strftime("%H:00") if row["hour_bucket"] else "00:00",
            "avg_ms": round(row["avg_ms"] or 0),
            "count": row["count"],
        }
        for row in trend_rows
    ]

    return AnalyticsOverview(
        period=period,
        total_queries=total_queries or 0,
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
        pending_review_count=pending_review or 0,
        total_tokens_input=total_tokens_input,
        total_tokens_output=total_tokens_output,
        total_cost_usd=round(total_cost, 4),
        cost_by_model=cost_by_model,
    )


async def get_recent_queries(limit: int = 50) -> list[dict]:
    """Get recent query events."""
    db = await get_db()
    async with db.connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id, timestamp, query, provider, model, response_time_ms,
                   sources_count, cache_hit, error
            FROM query_events
            ORDER BY timestamp DESC
            LIMIT $1
            """,
            limit,
        )

    return [
        {
            "id": row["id"],
            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
            "query": row["query"],
            "provider": row["provider"],
            "model": row["model"],
            "response_time_ms": row["response_time_ms"],
            "sources_count": row["sources_count"],
            "cache_hit": bool(row["cache_hit"]),
            "error": row["error"],
        }
        for row in rows
    ]


async def get_pending_feedback(limit: int = 50) -> list[dict]:
    """Get feedback entries needing review."""
    db = await get_db()
    async with db.connection() as conn:
        rows = await conn.fetch(
            """
            SELECT f.id,
                   f.timestamp,
                   f.query_id,
                   f.helpful,
                   f.comment,
                   q.query,
                   q.provider
            FROM feedback f
            LEFT JOIN query_events q ON f.query_id = q.id
            WHERE f.reviewed = FALSE
            ORDER BY f.timestamp DESC
            LIMIT $1
            """,
            limit,
        )

    return [
        {
            "id": row["id"],
            "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
            "query_id": row["query_id"],
            "helpful": bool(row["helpful"]),
            "comment": row["comment"],
            "query": row["query"],
            "provider": row["provider"],
        }
        for row in rows
    ]


async def mark_feedback_reviewed(feedback_id: str) -> bool:
    """Mark feedback as reviewed."""
    db = await get_db()
    async with db.connection() as conn:
        result = await conn.execute(
            "UPDATE feedback SET reviewed = TRUE WHERE id = $1",
            feedback_id,
        )
    # asyncpg returns command tag e.g. 'UPDATE 1'
    return result.endswith("1")
