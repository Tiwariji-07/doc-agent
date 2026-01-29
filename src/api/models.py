"""
Pydantic models for API request/response schemas.
"""

from typing import Optional

from pydantic import BaseModel, Field


# === Request Models ===


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question about WaveMaker documentation",
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream the response",
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source citations in response",
    )


# === Response Models ===


class Source(BaseModel):
    """A source citation from the documentation."""

    id: int = Field(..., description="Citation number (1-indexed)")
    title: str = Field(..., description="Document/section title")
    url: str = Field(..., description="URL to the documentation page")
    section: Optional[str] = Field(None, description="Section within the document")
    relevance_score: Optional[float] = Field(
        None, description="Relevance score from reranking"
    )


class Video(BaseModel):
    """A recommended video from WaveMaker Academy."""

    title: str = Field(..., description="Video title")
    url: str = Field(..., description="Video URL")
    duration: Optional[str] = Field(None, description="Video duration (e.g., '12:45')")


class ChatResponse(BaseModel):
    """Response model for chat endpoint (non-streaming)."""

    answer: str = Field(..., description="The generated answer with inline citations")
    sources: list[Source] = Field(
        default_factory=list,
        description="List of source citations",
    )
    videos: list[Video] = Field(
        default_factory=list,
        description="Recommended Academy videos",
    )
    cached: bool = Field(
        default=False,
        description="Whether this response was served from cache",
    )


class StreamChunk(BaseModel):
    """A chunk of streamed response."""

    type: str = Field(
        ...,
        description="Type of chunk: 'text', 'sources', 'videos', 'done', 'error'",
    )
    content: Optional[str] = Field(None, description="Text content for 'text' chunks")
    sources: Optional[list[Source]] = Field(
        None, description="Sources for 'sources' chunk"
    )
    videos: Optional[list[Video]] = Field(None, description="Videos for 'videos' chunk")
    error: Optional[str] = Field(None, description="Error message for 'error' chunk")
    cached: Optional[bool] = Field(
        None, description="Cache status for 'done' chunk"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: dict[str, str] = Field(
        default_factory=dict,
        description="Status of individual components",
    )


class IndexRequest(BaseModel):
    """Request model for triggering document indexing."""

    force_reindex: bool = Field(
        default=False,
        description="Force full reindex even if no changes detected",
    )
    branch: Optional[str] = Field(
        None,
        description="Git branch to index (defaults to configured branch)",
    )


class IndexResponse(BaseModel):
    """Response model for indexing operation."""

    status: str = Field(..., description="Indexing status")
    documents_processed: int = Field(
        default=0, description="Number of documents processed"
    )
    chunks_created: int = Field(default=0, description="Number of chunks created")
    duration_seconds: float = Field(
        default=0.0, description="Time taken for indexing"
    )
    errors: list[str] = Field(
        default_factory=list, description="Any errors encountered"
    )
