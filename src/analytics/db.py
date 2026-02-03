"""
Postgres database setup for analytics events.

Creates a connection pool up front and ensures the schema exists.
"""

import asyncpg
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# SQL schema for analytics tables (Postgres compatible)
SCHEMA = """
CREATE TABLE IF NOT EXISTS query_events (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    query TEXT NOT NULL,
    session_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response_time_ms INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    sources_count INTEGER DEFAULT 0,
    cache_hit BOOLEAN DEFAULT FALSE,
    error TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    query_id TEXT NOT NULL,
    helpful BOOLEAN NOT NULL,
    comment TEXT,
    reviewed BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (query_id) REFERENCES query_events(id)
);

CREATE INDEX IF NOT EXISTS idx_query_events_timestamp ON query_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_events_provider ON query_events(provider);
CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_helpful ON feedback(helpful);
CREATE INDEX IF NOT EXISTS idx_feedback_reviewed ON feedback(reviewed);
"""


class AnalyticsDB:
    """Async Postgres database wrapper for analytics persistence."""

    _instance: Optional["AnalyticsDB"] = None

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def get_instance(cls) -> "AnalyticsDB":
        """Get or create singleton instance."""
        if cls._instance is None:
            settings = get_settings()
            cls._instance = cls(settings.analytics_db_url)
            await cls._instance.initialize()
        return cls._instance

    async def initialize(self) -> None:
        """Initialize connection pool and ensure schema exists."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.db_url, min_size=1, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(SCHEMA)
                await self._ensure_boolean_columns(conn)
            logger.info("Analytics database initialized")

    async def acquire(self) -> asyncpg.Connection:
        """Acquire a connection from the pool."""
        if self._pool is None:
            await self.initialize()
        return await self._pool.acquire()

    async def release(self, connection: asyncpg.Connection) -> None:
        """Release a connection back to the pool."""
        if self._pool:
            await self._pool.release(connection)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        """Context manager to acquire and release a connection."""
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)

    async def _ensure_boolean_columns(self, conn: asyncpg.Connection) -> None:
        """Normalize legacy integer columns to boolean."""
        conversions = [
            (
                "query_events",
                "cache_hit",
                "ALTER TABLE query_events "
                "ALTER COLUMN cache_hit TYPE BOOLEAN "
                "USING (CASE WHEN cache_hit IN (1, TRUE) THEN TRUE ELSE FALSE END)",
                "ALTER TABLE query_events ALTER COLUMN cache_hit SET DEFAULT FALSE",
            ),
            (
                "feedback",
                "helpful",
                "ALTER TABLE feedback "
                "ALTER COLUMN helpful TYPE BOOLEAN "
                "USING (CASE WHEN helpful IN (1, TRUE) THEN TRUE ELSE FALSE END)",
                "ALTER TABLE feedback ALTER COLUMN helpful SET DEFAULT FALSE",
            ),
            (
                "feedback",
                "reviewed",
                "ALTER TABLE feedback "
                "ALTER COLUMN reviewed TYPE BOOLEAN "
                "USING (CASE WHEN reviewed IN (1, TRUE) THEN TRUE ELSE FALSE END)",
                "ALTER TABLE feedback ALTER COLUMN reviewed SET DEFAULT FALSE",
            ),
        ]

        for table, column, alter_sql, default_sql in conversions:
            try:
                column_type = await conn.fetchval(
                    """
                    SELECT data_type
                    FROM information_schema.columns
                    WHERE table_name = $1 AND column_name = $2
                    """,
                    table,
                    column,
                )
                if column_type and column_type.lower() not in {"boolean", "bool"}:
                    await conn.execute(alter_sql)
                    await conn.execute(default_sql)
            except Exception as exc:
                logger.warning(
                    "Failed to normalize %s.%s column to boolean: %s",
                    table,
                    column,
                    exc,
                )

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Analytics database connection closed")


async def get_db() -> AnalyticsDB:
    """Get analytics database instance."""
    return await AnalyticsDB.get_instance()
