"""
Response generator using Claude for answer synthesis.

Takes retrieved context and generates a helpful response
with proper citations and formatting.
"""

import logging
from typing import Any, AsyncGenerator, Optional

import anthropic

from src.api.models import Source, Video
from src.config.settings import get_settings
from src.core.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

# System prompt for the documentation assistant
SYSTEM_PROMPT = """You are the WaveMaker Documentation Assistant, an expert on the WaveMaker low-code development platform. Your job is to help developers understand and use WaveMaker effectively.

## Your Knowledge Source

You answer questions ONLY based on the provided documentation context. You do not have knowledge beyond what is given to you in each query.

## Citation Rules (CRITICAL)

1. ALWAYS cite your sources using numbered references: [1], [2], etc.
2. Place citations INLINE, immediately after the relevant information
3. Each citation number corresponds to a document in the context
4. If multiple sources support the same point, cite all: [1][2]
5. NEVER make up information not in the provided context

## When You Don't Know

If the provided context doesn't contain information to answer the question:
- Say: "I don't have specific information about this in the documentation."
- Suggest checking related topics or contacting support
- DO NOT hallucinate or guess

## Response Guidelines

1. Be concise but complete
2. Use markdown formatting for readability:
   - **Bold** for important terms
   - `code` for code/commands
   - Numbered lists for steps
   - Bullet points for features/options
3. Start with a direct answer, then elaborate if needed
4. For how-to questions, provide step-by-step instructions
5. Include code examples when they appear in the context

## Video Recommendations

You may have video metadata (titles only) without their content.
- DO NOT claim to know what's in the videos
- Recommend them as supplementary: "For a visual guide, see: [video title]"
- Only recommend videos whose titles clearly match the topic"""


class ResponseGenerator:
    """
    Generates responses using Claude with retrieved context.
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[anthropic.AsyncAnthropic] = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=self.settings.anthropic_api_key,
            )
        return self._client

    def _format_context(
        self,
        documents: list[RetrievedDocument],
        videos: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """
        Format retrieved documents and videos into context string.
        """
        parts = ["## Documentation Context\n"]

        # Add documents with numbered citations
        for i, doc in enumerate(documents, 1):
            section_info = f" > {doc.section}" if doc.section else ""
            parts.append(f"[{i}] {doc.title}{section_info}")
            parts.append(f"URL: {doc.url}")
            parts.append("---")
            parts.append(doc.content)
            parts.append("")

        # Add videos if available
        if videos:
            parts.append("\n## Related Videos (supplementary)\n")
            for video in videos:
                duration = f" ({video.get('duration', 'N/A')})" if video.get('duration') else ""
                parts.append(f"📺 \"{video.get('title', 'Video')}\" - {video.get('url', '')}{duration}")
            parts.append("")
            parts.append("(Note: Video titles are provided for recommendation. You do not have access to their content.)")

        return "\n".join(parts)

    def _build_user_prompt(
        self,
        query: str,
        documents: list[RetrievedDocument],
        videos: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Build the complete user prompt with context and query."""
        context = self._format_context(documents, videos)

        return f"""{context}

---

## Question

{query}

---

Please answer the question based on the documentation context above.
Remember to cite your sources using [1], [2], etc."""

    async def generate(
        self,
        query: str,
        documents: list[RetrievedDocument],
        videos: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Generate a complete response (non-streaming).

        Returns:
            Dict with answer, sources, and videos
        """
        client = self._get_client()
        user_prompt = self._build_user_prompt(query, documents, videos)

        response = await client.messages.create(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            temperature=self.settings.llm_temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.content[0].text

        # Build sources list
        sources = self._extract_sources(documents)

        # Build videos list
        video_list = self._extract_videos(videos) if videos else []

        return {
            "answer": answer,
            "sources": sources,
            "videos": video_list,
        }

    async def generate_stream(
        self,
        query: str,
        documents: list[RetrievedDocument],
        videos: Optional[list[dict[str, Any]]] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Generate a streaming response.

        Yields:
            Dict chunks with type and content
        """
        client = self._get_client()
        user_prompt = self._build_user_prompt(query, documents, videos)

        try:
            async with client.messages.stream(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            ) as stream:
                async for text in stream.text_stream:
                    yield {"type": "text", "content": text}

            # After text is complete, yield sources and videos
            sources = self._extract_sources(documents)
            yield {"type": "sources", "sources": [s.model_dump() for s in sources]}

            if videos:
                video_list = self._extract_videos(videos)
                yield {"type": "videos", "videos": [v.model_dump() for v in video_list]}

            yield {"type": "done", "cached": False}

        except Exception as e:
            logger.exception(f"Error during streaming generation: {e}")
            yield {"type": "error", "error": str(e)}

    def _extract_sources(self, documents: list[RetrievedDocument]) -> list[Source]:
        """Extract source citations from documents."""
        sources = []
        for i, doc in enumerate(documents, 1):
            sources.append(
                Source(
                    id=i,
                    title=doc.title,
                    url=doc.url,
                    section=doc.section,
                    relevance_score=doc.rrf_score,
                )
            )
        return sources

    def _extract_videos(self, videos: list[dict[str, Any]]) -> list[Video]:
        """Extract video objects from video data."""
        return [
            Video(
                title=v.get("title", "Video"),
                url=v.get("url", ""),
                duration=v.get("duration"),
            )
            for v in videos
        ]
