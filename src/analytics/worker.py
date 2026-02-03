"""
Background worker for flushing analytics events from Redis to Postgres.

Runs as an asyncio task, batch-inserting events periodically.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis

from src.config.settings import get_settings
from src.analytics.db import get_db
from src.analytics.models import QueryEvent, FeedbackEvent

logger = logging.getLogger(__name__)

# Worker task reference
_worker_task: Optional[asyncio.Task] = None


async def start_worker() -> None:
    """Start the background analytics worker."""
    global _worker_task
    settings = get_settings()
    
    if not settings.analytics_enabled:
        logger.info("Analytics disabled, worker not started")
        return
    
    if _worker_task is not None:
        logger.warning("Analytics worker already running")
        return
    
    _worker_task = asyncio.create_task(_worker_loop())
    logger.info("Analytics background worker started")


async def stop_worker() -> None:
    """Stop the background analytics worker."""
    global _worker_task
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
        logger.info("Analytics background worker stopped")


async def _worker_loop() -> None:
    """Main worker loop - flush events periodically."""
    settings = get_settings()
    redis_client = redis.from_url(settings.redis_url)
    
    while True:
        try:
            await _flush_events(redis_client, settings)
            await asyncio.sleep(settings.analytics_flush_interval)
        except asyncio.CancelledError:
            # Final flush before shutdown
            await _flush_events(redis_client, settings)
            break
        except Exception as e:
            logger.exception(f"Analytics worker error: {e}")
            await asyncio.sleep(5)  # Backoff on error


async def _flush_events(redis_client: redis.Redis, settings) -> None:
    """Flush events from Redis queues to Postgres."""
    # Flush query events
    await _flush_queue(
        redis_client,
        f"{settings.analytics_redis_key}:queries",
        _insert_query_events,
        settings.analytics_batch_size,
    )
    
    # Flush feedback events
    await _flush_queue(
        redis_client,
        f"{settings.analytics_redis_key}:feedback",
        _insert_feedback_events,
        settings.analytics_batch_size,
    )


async def _flush_queue(
    redis_client: redis.Redis,
    queue_key: str,
    insert_fn,
    batch_size: int,
) -> None:
    """Flush a single queue to database."""
    # Get batch of events
    events_raw = await redis_client.lrange(queue_key, -batch_size, -1)
    if not events_raw:
        return
    
    # Parse events
    events = []
    for raw in events_raw:
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid event JSON: {e}")
    
    if not events:
        return
    
    # Insert to database
    try:
        await insert_fn(events)
        # Remove processed events from queue
        await redis_client.ltrim(queue_key, 0, -(len(events_raw) + 1))
        logger.debug(f"Flushed {len(events)} events from {queue_key}")
    except Exception as e:
        logger.exception(f"Failed to insert events: {e}")


QUERY_UPSERT_SQL = """
INSERT INTO query_events (
    id, timestamp, query, session_id, provider, model,
    response_time_ms, tokens_input, tokens_output,
    sources_count, cache_hit, error
)
VALUES (
    $1, $2, $3, $4, $5, $6,
    $7, $8, $9,
    $10, $11, $12
)
ON CONFLICT (id) DO UPDATE SET
    timestamp = EXCLUDED.timestamp,
    query = EXCLUDED.query,
    session_id = EXCLUDED.session_id,
    provider = EXCLUDED.provider,
    model = EXCLUDED.model,
    response_time_ms = EXCLUDED.response_time_ms,
    tokens_input = EXCLUDED.tokens_input,
    tokens_output = EXCLUDED.tokens_output,
    sources_count = EXCLUDED.sources_count,
    cache_hit = EXCLUDED.cache_hit,
    error = EXCLUDED.error;
"""

FEEDBACK_UPSERT_SQL = """
INSERT INTO feedback (
    id, timestamp, query_id, helpful, comment, reviewed
)
VALUES ($1, $2, $3, $4, $5, FALSE)
ON CONFLICT (id) DO UPDATE SET
    timestamp = EXCLUDED.timestamp,
    query_id = EXCLUDED.query_id,
    helpful = EXCLUDED.helpful,
    comment = EXCLUDED.comment,
    reviewed = EXCLUDED.reviewed;
"""


def _parse_timestamp(value: str | None) -> datetime:
    """Parse ISO timestamp strings from queued events."""
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        logger.warning(f"Invalid timestamp format: {value}")
        return datetime.now(timezone.utc)


async def _insert_query_events(events: list[dict]) -> None:
    """Batch insert query events to Postgres."""
    if not events:
        return

    db = await get_db()
    async with db.connection() as conn:
        records = [
            (
                e["id"],
                _parse_timestamp(e.get("timestamp")),
                e["query"],
                e.get("session_id"),
                e["provider"],
                e["model"],
                e.get("response_time_ms"),
                e.get("tokens_input"),
                e.get("tokens_output"),
                e.get("sources_count", 0),
                bool(e.get("cache_hit")),
                e.get("error"),
            )
            for e in events
        ]
        async with conn.transaction():
            await conn.executemany(QUERY_UPSERT_SQL, records)


async def _insert_feedback_events(events: list[dict]) -> None:
    """Batch insert feedback events to Postgres."""
    if not events:
        return

    db = await get_db()
    async with db.connection() as conn:
        records = [
            (
                e["id"],
                _parse_timestamp(e.get("timestamp")),
                e["query_id"],
                bool(e["helpful"]),
                e.get("comment"),
            )
            for e in events
        ]
        async with conn.transaction():
            await conn.executemany(FEEDBACK_UPSERT_SQL, records)
