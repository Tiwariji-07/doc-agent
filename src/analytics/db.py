"""
SQLite database setup for analytics.

Uses aiosqlite for async operations.
"""

import aiosqlite
import logging
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# SQL schema for analytics tables
SCHEMA = """
CREATE TABLE IF NOT EXISTS query_events (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    session_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response_time_ms INTEGER,
    tokens_input INTEGER,
    tokens_output INTEGER,
    sources_count INTEGER DEFAULT 0,
    cache_hit INTEGER DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    query_id TEXT NOT NULL,
    helpful INTEGER NOT NULL,
    comment TEXT,
    reviewed INTEGER DEFAULT 0,
    FOREIGN KEY (query_id) REFERENCES query_events(id)
);

CREATE INDEX IF NOT EXISTS idx_query_events_timestamp ON query_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_query_events_provider ON query_events(provider);
CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_helpful ON feedback(helpful);
CREATE INDEX IF NOT EXISTS idx_feedback_reviewed ON feedback(reviewed);
"""


class AnalyticsDB:
    """Async SQLite database wrapper for analytics."""
    
    _instance: Optional["AnalyticsDB"] = None
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[aiosqlite.Connection] = None
    
    @classmethod
    async def get_instance(cls) -> "AnalyticsDB":
        """Get or create singleton instance."""
        if cls._instance is None:
            settings = get_settings()
            cls._instance = cls(settings.analytics_db_path)
            await cls._instance.initialize()
        return cls._instance
    
    async def initialize(self) -> None:
        """Initialize database and create tables."""
        # Ensure directory exists
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect and create schema
        self._connection = await aiosqlite.connect(self.db_path)
        await self._connection.executescript(SCHEMA)
        await self._connection.commit()
        logger.info(f"Analytics database initialized: {self.db_path}")
    
    async def get_connection(self) -> aiosqlite.Connection:
        """Get database connection."""
        if self._connection is None:
            await self.initialize()
        return self._connection
    
    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Analytics database connection closed")


async def get_db() -> AnalyticsDB:
    """Get analytics database instance."""
    return await AnalyticsDB.get_instance()
