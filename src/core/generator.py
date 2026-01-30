"""
Response generator using Claude for answer synthesis.

Takes retrieved context and generates a helpful response
with proper citations and formatting.

Features:
- Prompt caching for reduced latency and cost
- Improved system prompt for better structured responses
"""

import logging
from typing import Any, AsyncGenerator, Optional

import anthropic

from src.api.models import Source, Video
from src.config.settings import get_settings
from src.core.retriever import RetrievedDocument

logger = logging.getLogger(__name__)

# ============================================================================
# IMPROVED SYSTEM PROMPT
# ============================================================================
# This prompt is designed to produce well-structured, consistently formatted
# responses with proper markdown, citations, and helpful organization.
# ============================================================================

SYSTEM_PROMPT = """You are the **WaveMaker Documentation Assistant**, an expert on the WaveMaker low-code development platform. Your role is to help developers understand and effectively use WaveMaker.

## 🎯 Your Mission

Provide clear, accurate, and actionable answers based ONLY on the provided documentation context. You have no knowledge beyond what's given in each query.

## 📋 Response Structure

ALWAYS structure your responses in this order:

1. **Direct Answer** (1-2 sentences) - Immediately answer the question
2. **Details** - Expand with relevant information, organized logically
3. **Steps** (if applicable) - Number each step clearly
4. **Code Examples** (if in context) - Use proper code blocks with language tags
5. **Related Information** (optional) - Brief mention of related features

## ✅ Citation Rules (CRITICAL)

- ALWAYS cite sources using numbered references: [1], [2], etc.
- Place citations INLINE, immediately after the relevant information
- Each number corresponds to a document in the context
- Multiple sources supporting the same point: [1][2]
- NEVER make up information not in the provided context

## ⚠️ When You Don't Know

If the context doesn't contain the answer:
- State clearly: "I don't have specific information about this in the documentation."
- Suggest related topics if applicable
- DO NOT guess or hallucinate

## 📝 Markdown Formatting Guidelines

Use these consistently:

```
**Bold** - Important terms, key concepts, UI elements
`code` - Commands, file names, code snippets, properties
### Headers - For major sections in longer answers
- Bullets - For lists of features, options, alternatives
1. Numbers - For sequential steps or ordered processes
> Blockquote - For important notes or warnings
```

### Code Block Format
Always specify the language for syntax highlighting:
```javascript
// Example code
```

## 📺 Video Recommendations

- Videos are supplementary resources (you only have titles)
- Format: "For a visual guide, see: [Video Title]"
- Only recommend if the title clearly matches the topic

## 🎨 Response Quality Checklist

Before responding, ensure:
- [ ] Direct answer is provided first
- [ ] All claims have citations [n]
- [ ] Code uses proper formatting with language tags
- [ ] Steps are numbered, not bulleted
- [ ] Key terms are **bolded**"""


class ResponseGenerator:
    """
    Generates responses using Claude with retrieved context.
    
    Features:
    - Prompt caching for the system prompt (reduces cost/latency)
    - Streaming support for real-time responses
    - Structured output with sources and videos
    """

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[anthropic.AsyncAnthropic] = None
        
        # Build the cached system prompt structure once
        # This enables Anthropic's prompt caching feature
        self._system_with_cache = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }
        ]

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
        parts = ["## 📚 Documentation Context\n"]

        # Add documents with numbered citations
        for i, doc in enumerate(documents, 1):
            section_info = f" > {doc.section}" if doc.section else ""
            parts.append(f"### [{i}] {doc.title}{section_info}")
            parts.append(f"📎 URL: {doc.url}")
            parts.append("---")
            parts.append(doc.content)
            parts.append("")

        # Add videos if available
        if videos:
            parts.append("\n## 🎬 Related Videos (supplementary)\n")
            for video in videos:
                duration = f" ({video.get('duration', 'N/A')})" if video.get('duration') else ""
                parts.append(f"📺 \"{video.get('title', 'Video')}\" - {video.get('url', '')}{duration}")
            parts.append("")
            parts.append("*(Note: Video titles are provided for recommendation. You do not have access to their content.)*")

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

## ❓ Question

{query}

---

**Instructions:** Answer the question based on the documentation context above.
- Start with a direct answer
- Use proper markdown formatting
- Cite sources using [1], [2], etc. inline
- Include code examples if relevant (with language tags)"""

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

        try:
            response = await client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                system=self._system_with_cache,  # Uses cached system prompt
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            )
            
            # Log cache usage if available
            if hasattr(response, 'usage'):
                usage = response.usage
                cache_read = getattr(usage, 'cache_read_input_tokens', 0)
                cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
                if cache_read or cache_create:
                    logger.info(f"Cache usage - read: {cache_read}, created: {cache_create}")

            answer = response.content[0].text

        except anthropic.BadRequestError as e:
            # Fallback to non-cached if cache not supported
            logger.warning(f"Cache not supported, falling back: {e}")
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
            # Try with cached system prompt first
            async with client.messages.stream(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                system=self._system_with_cache,  # Uses cached system prompt
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
            ) as stream:
                async for text in stream.text_stream:
                    yield {"type": "text", "content": text}
                
                # Log cache usage after stream completes
                final_message = await stream.get_final_message()
                if hasattr(final_message, 'usage'):
                    usage = final_message.usage
                    cache_read = getattr(usage, 'cache_read_input_tokens', 0)
                    cache_create = getattr(usage, 'cache_creation_input_tokens', 0)
                    if cache_read or cache_create:
                        logger.info(f"Stream cache usage - read: {cache_read}, created: {cache_create}")

            # After text is complete, yield sources and videos
            sources = self._extract_sources(documents)
            yield {"type": "sources", "sources": [s.model_dump() for s in sources]}

            if videos:
                video_list = self._extract_videos(videos)
                yield {"type": "videos", "videos": [v.model_dump() for v in video_list]}

            yield {"type": "done", "cached": False}

        except anthropic.BadRequestError as e:
            # Fallback to non-cached if cache not supported
            logger.warning(f"Stream cache not supported, falling back: {e}")
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
