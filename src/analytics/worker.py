"""
Background worker for flushing analytics events from Redis to SQLite.

Runs as an asyncio task, batch-inserting events periodically.
"""

import asyncio
import json
import logging
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
    """Flush events from Redis to SQLite."""
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


async def _insert_query_events(events: list[dict]) -> None:
    """Batch insert query events to SQLite."""
    db = await get_db()
    conn = await db.get_connection()
    
    await conn.executemany(
        """
        INSERT OR REPLACE INTO query_events 
        (id, timestamp, query, session_id, provider, model, 
         response_time_ms, tokens_input, tokens_output, 
         sources_count, cache_hit, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                e["id"],
                e["timestamp"],
                e["query"],
                e.get("session_id"),
                e["provider"],
                e["model"],
                e.get("response_time_ms"),
                e.get("tokens_input"),
                e.get("tokens_output"),
                e.get("sources_count", 0),
                1 if e.get("cache_hit") else 0,
                e.get("error"),
            )
            for e in events
        ],
    )
    await conn.commit()


async def _insert_feedback_events(events: list[dict]) -> None:
    """Batch insert feedback events to SQLite."""
    db = await get_db()
    conn = await db.get_connection()
    
    await conn.executemany(
        """
        INSERT OR REPLACE INTO feedback 
        (id, timestamp, query_id, helpful, comment, reviewed)
        VALUES (?, ?, ?, ?, ?, 0)
        """,
        [
            (
                e["id"],
                e["timestamp"],
                e["query_id"],
                1 if e["helpful"] else 0,
                e.get("comment"),
            )
            for e in events
        ],
    )
    await conn.commit()
