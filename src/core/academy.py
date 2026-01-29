"""
Academy MCP client for fetching video recommendations.

Connects to the Academy MCP server to get relevant videos
based on the query.
"""

import logging
from typing import Any, Optional

import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class AcademyClient:
    """
    Client for WaveMaker Academy MCP server.

    Currently fetches video metadata (title, URL, duration).
    Phase 2 will add transcript content for better integration.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=10.0,  # 10 second timeout
            )
        return self._client

    @property
    def is_configured(self) -> bool:
        """Check if Academy MCP is configured."""
        return bool(self.settings.academy_mcp_url)

    async def search_videos(
        self,
        query: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Search for relevant videos in Academy.

        Args:
            query: Search query
            limit: Maximum number of videos to return

        Returns:
            List of video metadata dicts
        """
        if not self.is_configured:
            logger.debug("Academy MCP not configured, skipping video search")
            return []

        try:
            client = await self._get_client()
            url = f"{self.settings.academy_mcp_url}/search"

            response = await client.post(
                url,
                json={
                    "query": query,
                    "limit": limit,
                },
            )
            response.raise_for_status()

            data = response.json()
            videos = data.get("results", [])

            logger.info(f"Found {len(videos)} videos for query: {query[:50]}...")
            return videos

        except httpx.TimeoutException:
            logger.warning("Academy MCP request timed out")
            return []
        except httpx.HTTPError as e:
            logger.warning(f"Academy MCP request failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"Academy MCP unexpected error: {e}")
            return []

    async def get_video_details(
        self,
        video_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get detailed information about a specific video.

        Args:
            video_id: Video identifier

        Returns:
            Video details including transcript (if available)
        """
        if not self.is_configured:
            return None

        try:
            client = await self._get_client()
            url = f"{self.settings.academy_mcp_url}/videos/{video_id}"

            response = await client.get(url)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.warning(f"Failed to get video details: {e}")
            return None

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
