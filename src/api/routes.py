"""
API routes for WaveMaker Docs Agent.
"""

import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    StreamChunk,
)
from src.config.settings import get_settings
from src.core.pipeline import QueryPipeline

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy-loaded pipeline instance
_pipeline: QueryPipeline | None = None


def get_pipeline() -> QueryPipeline:
    """Get or create the query pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = QueryPipeline()
    return _pipeline


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    Returns status of all components.
    """
    settings = get_settings()
    components = {}

    # Check Redis
    try:
        pipeline = get_pipeline()
        await pipeline.cache.ping()
        components["redis"] = "healthy"
    except Exception as e:
        components["redis"] = f"unhealthy: {str(e)}"

    # Check Qdrant
    try:
        pipeline = get_pipeline()
        await pipeline.retriever.health_check()
        components["qdrant"] = "healthy"
    except Exception as e:
        components["qdrant"] = f"unhealthy: {str(e)}"

    # Check Anthropic (just verify API key is set)
    if settings.anthropic_api_key:
        components["anthropic"] = "configured"
    else:
        components["anthropic"] = "not configured"

    # Overall status
    all_healthy = all(
        "healthy" in v or "configured" in v for v in components.values()
    )
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        components=components,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for querying documentation.
    Supports both streaming and non-streaming responses.
    """
    query_id = str(uuid.uuid4())
    logger.info(f"Chat request [{query_id}]: {request.query[:100]}...")

    try:
        pipeline = get_pipeline()

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_response(pipeline, request, query_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Return complete response with analytics
            start_time = time.time()
            result = await pipeline.query(
                query=request.query,
                include_sources=request.include_sources,
            )
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract token usage and cache status
            usage = result.get("usage", {})
            cache_hit = result.get("cached", False)
            
            # Track analytics (fire-and-forget)
            await _track_query(
                query_id, 
                request.query, 
                response_time_ms, 
                len(result.get("sources", [])),
                tokens_input=usage.get("input_tokens"),
                tokens_output=usage.get("output_tokens"),
                cache_hit=cache_hit,
            )
            
            # Add query_id to response
            result["query_id"] = query_id
            return ChatResponse(**result)

    except Exception as e:
        logger.exception(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(
    pipeline: QueryPipeline,
    request: ChatRequest,
    query_id: str,
) -> AsyncGenerator[str, None]:
    """
    Stream the response as Server-Sent Events.
    """
    start_time = time.time()
    sources_count = 0
    usage = {}
    cache_hit = False
    
    try:
        async for chunk in pipeline.query_stream(
            query=request.query,
            include_sources=request.include_sources,
        ):
            # Track sources count from sources chunk
            if chunk.get("type") == "sources":
                sources_count = len(chunk.get("sources", []))
            
            # Extract token usage and cache status from done chunk
            if chunk.get("type") == "done":
                chunk["query_id"] = query_id
                usage = chunk.get("usage", {})
                cache_hit = chunk.get("cached", False)
            
            # Format as SSE
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Track analytics after stream completes
        response_time_ms = int((time.time() - start_time) * 1000)
        await _track_query(
            query_id, 
            request.query, 
            response_time_ms, 
            sources_count,
            tokens_input=usage.get("input_tokens"),
            tokens_output=usage.get("output_tokens"),
            cache_hit=cache_hit,
        )

    except Exception as e:
        logger.exception(f"Error during streaming: {e}")
        error_chunk = StreamChunk(type="error", error=str(e))
        yield f"data: {error_chunk.model_dump_json()}\n\n"
        
        # Track error
        await _track_query(query_id, request.query, 0, 0, error=str(e))


async def _track_query(
    query_id: str,
    query: str,
    response_time_ms: int,
    sources_count: int,
    error: str | None = None,
    tokens_input: int | None = None,
    tokens_output: int | None = None,
    cache_hit: bool = False,
) -> None:
    """Track query analytics (fire-and-forget)."""
    settings = get_settings()
    if not settings.analytics_enabled:
        return
    
    try:
        from src.analytics import track_query, QueryEvent
        
        # Determine model based on provider
        if settings.ai_provider == "anthropic":
            model = settings.anthropic_model
        elif settings.ai_provider == "openai":
            model = settings.openai_model
        else:
            model = settings.ollama_model
        
        event = QueryEvent(
            id=query_id,
            query=query,
            provider=settings.ai_provider,
            model=model,
            response_time_ms=response_time_ms,
            sources_count=sources_count,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cache_hit=cache_hit,
            error=error,
        )
        await track_query(event)
    except Exception as e:
        # Never fail the main request due to analytics
        logger.warning(f"Failed to track analytics: {e}")


@router.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest) -> IndexResponse:
    """
    Trigger document indexing.
    This can be called manually or via webhook.
    """
    logger.info(f"Index request: force_reindex={request.force_reindex}")

    try:
        # Import here to avoid circular imports and startup cost
        from src.indexer.indexer import DocumentIndexer

        indexer = DocumentIndexer()
        result = await indexer.index(
            branch=request.branch,
            force=request.force_reindex,
        )

        return IndexResponse(**result)

    except Exception as e:
        logger.exception(f"Error during indexing: {e}")
        raise HTTPException(status_code=500, detail=str(e))
