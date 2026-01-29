"""
API routes for WaveMaker Docs Agent.
"""

import json
import logging
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
    logger.info(f"Chat request: {request.query[:100]}...")

    try:
        pipeline = get_pipeline()

        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_response(pipeline, request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Return complete response
            result = await pipeline.query(
                query=request.query,
                include_sources=request.include_sources,
            )
            return ChatResponse(**result)

    except Exception as e:
        logger.exception(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def stream_response(
    pipeline: QueryPipeline,
    request: ChatRequest,
) -> AsyncGenerator[str, None]:
    """
    Stream the response as Server-Sent Events.
    """
    try:
        async for chunk in pipeline.query_stream(
            query=request.query,
            include_sources=request.include_sources,
        ):
            # Format as SSE
            yield f"data: {json.dumps(chunk)}\n\n"

    except Exception as e:
        logger.exception(f"Error during streaming: {e}")
        error_chunk = StreamChunk(type="error", error=str(e))
        yield f"data: {error_chunk.model_dump_json()}\n\n"


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
