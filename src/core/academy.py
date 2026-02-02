"""
Academy MCP client for fetching video recommendations.

Connects to the Academy MCP server using the Model Context Protocol
with streamable HTTP transport to get relevant videos based on the query.
"""

import asyncio
import json
import logging
import time
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

# Connection configuration
MCP_CONNECT_TIMEOUT = 10.0  # seconds
MCP_SESSION_MAX_AGE = 300  # 5 minutes - recreate session after this time
MCP_CALL_TIMEOUT = 15.0  # seconds - timeout for tool calls


class AcademyClient:
    """
    Client for WaveMaker Academy MCP server.

    Uses Model Context Protocol (MCP) with streamable HTTP transport
    to call the wm-academy-semantic-search tool.

    Features:
    - Automatic reconnection on timeout
    - Session age tracking to prevent stale connections
    - Configurable timeouts
    """

    def __init__(self):
        self.settings = get_settings()
        self._session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        self._context = None
        self._session_created_at: Optional[float] = None

    @property
    def is_configured(self) -> bool:
        """Check if Academy MCP is configured."""
        return bool(self.settings.academy_mcp_url)

    def _is_session_stale(self) -> bool:
        """Check if current session is stale and needs recreation."""
        if self._session is None or self._session_created_at is None:
            return True

        age = time.time() - self._session_created_at
        if age > MCP_SESSION_MAX_AGE:
            logger.debug(f"MCP session is stale (age: {age:.1f}s > {MCP_SESSION_MAX_AGE}s)")
            return True

        return False

    async def _close_session(self) -> None:
        """Close current MCP session and cleanup resources."""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing MCP session: {e}")
            finally:
                self._session = None

        if self._context:
            try:
                await self._context.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Error closing streamable HTTP context: {e}")
            finally:
                self._context = None
                self._read = None
                self._write = None

        self._session_created_at = None

    async def _ensure_session(self) -> ClientSession:
        """
        Get or create MCP session.

        Establishes a persistent session with the Academy MCP server
        using streamable HTTP transport. Automatically recreates stale sessions.

        Returns:
            ClientSession: Active MCP session

        Raises:
            ValueError: If Academy MCP URL is not configured
        """
        if not self.is_configured:
            raise ValueError("Academy MCP URL not configured")

        # Check if session needs recreation
        if self._is_session_stale():
            logger.debug("Recreating stale MCP session...")
            await self._close_session()

        # Create new session if needed
        if self._session is None:
            try:
                # Create streamable HTTP client with timeout
                logger.debug(f"Connecting to MCP server: {self.settings.academy_mcp_url}")
                self._context = streamablehttp_client(self.settings.academy_mcp_url)

                # Set connection timeout
                try:
                    self._read, self._write, _ = await asyncio.wait_for(
                        self._context.__aenter__(),
                        timeout=MCP_CONNECT_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"MCP connection timed out after {MCP_CONNECT_TIMEOUT}s")

                # Create and initialize MCP session
                self._session = ClientSession(self._read, self._write)
                await self._session.__aenter__()
                await self._session.initialize()

                self._session_created_at = time.time()
                logger.info(f"MCP session initialized with Academy server (timeout: {MCP_SESSION_MAX_AGE}s)")

            except Exception as e:
                logger.error(f"Failed to initialize MCP session: {e}")
                # Clean up partial state
                await self._close_session()
                raise

        return self._session

    async def search_videos(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant videos using MCP tool.

        Calls the wm-academy-semantic-search tool on the Academy MCP server
        to find videos related to the query.

        Args:
            query: Search query
            limit: Maximum number of videos to return (default: 3)

        Returns:
            List of video metadata dicts with keys:
            - id: Video ID
            - title: Video title
            - description: Video description
            - moduleName: Module name
            - code: Video code (e.g., CHAP_66)
            - link: Video URL
        """
        if not self.is_configured:
            logger.debug("Academy MCP not configured, skipping video search")
            return []

        # Retry logic for transient connection errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.debug(f"Searching Academy videos for query: {query[:50]}... (attempt {attempt + 1}/{max_retries})")
                session = await self._ensure_session()

                # Call the MCP tool with timeout
                logger.debug(f"Calling MCP tool: wm-academy-semantic-search with limit={limit}")
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(
                            name="wm-academy-semantic-search",
                            arguments={
                                "query": query,
                                "limit": limit,
                            },
                        ),
                        timeout=MCP_CALL_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"MCP tool call timed out after {MCP_CALL_TIMEOUT}s")

                # Parse MCP response
                # The tool returns a JSON string in result.content[0].text
                response_text = result.content[0].text
                logger.debug(f"MCP response received: {response_text[:200]}...")
                data = json.loads(response_text)

                # Extract videos from body array
                # Response format: {"headers": {}, "body": [...], "statusCodeValue": 200}
                videos = data.get("body", [])

                logger.info(f"Found {len(videos)} videos for query: {query[:50]}...")
                return videos

            except (TimeoutError, ConnectionError, asyncio.TimeoutError) as e:
                # Connection/timeout errors - retry with fresh session
                logger.warning(f"MCP connection error on attempt {attempt + 1}: {e}")
                await self._close_session()  # Force session recreation

                if attempt < max_retries - 1:
                    logger.debug(f"Retrying with new session...")
                    await asyncio.sleep(0.5)  # Brief delay before retry
                else:
                    logger.error(f"Academy MCP search failed after {max_retries} attempts")
                    return []

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Academy MCP response: {e}", exc_info=True)
                return []

            except Exception as e:
                logger.error(f"Academy MCP search failed ({type(e).__name__}): {e}", exc_info=True)
                return []

        return []

    async def close(self) -> None:
        """Close MCP session and cleanup resources."""
        await self._close_session()
        logger.debug("Academy MCP client closed")
