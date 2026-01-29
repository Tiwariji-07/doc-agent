"""
Main RAG pipeline orchestrator.

Ties together all layers:
1. Cache check
2. Query embedding
3. Hybrid retrieval
4. Reranking
5. Response generation
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

import numpy as np

from src.core.academy import AcademyClient
from src.core.cache import SemanticCache
from src.core.embedder import Embedder
from src.core.generator import ResponseGenerator
from src.core.reranker import Reranker
from src.core.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Main RAG pipeline for answering documentation queries.

    Flow:
    1. Check cache (exact match, then semantic)
    2. Generate query embeddings (dense + sparse)
    3. Retrieve documents from Qdrant (hybrid search)
    4. Fetch Academy videos (parallel with retrieval)
    5. Rerank results with cross-encoder
    6. Generate response with Claude
    7. Cache the response
    """

    def __init__(self):
        self.cache = SemanticCache()
        self.embedder = Embedder()
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.generator = ResponseGenerator()
        self.academy = AcademyClient()

    async def query(
        self,
        query: str,
        include_sources: bool = True,
    ) -> dict[str, Any]:
        """
        Process a query and return complete response.

        Args:
            query: User's question
            include_sources: Whether to include source citations

        Returns:
            Dict with answer, sources, videos, cached flag
        """
        # Layer 1: Check exact cache
        cached = await self.cache.get_exact(query)
        if cached:
            cached["cached"] = True
            return cached

        # Layer 2: Generate embeddings
        dense_vector = self.embedder.embed_query(query)
        sparse_vector = self.embedder.generate_sparse_vector(query)

        # Check semantic cache
        semantic_cached = await self.cache.get_semantic(query, dense_vector)
        if semantic_cached:
            semantic_cached["cached"] = True
            return semantic_cached

        # Layer 3: Retrieve documents
        # Note: Academy MCP disabled until server is ready
        documents = await self.retriever.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        videos = []  # Empty until Academy MCP is ready

        # Layer 4: Rerank documents
        if documents and self.reranker.should_rerank(documents):
            documents = await self.reranker.rerank(query, documents)
        elif documents:
            # Just take top_k without reranking
            documents = documents[:self.settings.rerank_top_k]

        # Layer 5: Generate response
        if not documents:
            response = {
                "answer": "I don't have information about this topic in the documentation. Please try rephrasing your question or check the WaveMaker documentation directly.",
                "sources": [],
                "videos": [],
                "cached": False,
            }
        else:
            result = await self.generator.generate(
                query=query,
                documents=documents if include_sources else documents[:3],
                videos=videos,
            )
            response = {
                "answer": result["answer"],
                "sources": [s.model_dump() for s in result["sources"]] if include_sources else [],
                "videos": [v.model_dump() for v in result["videos"]],
                "cached": False,
            }

        # Cache the response
        await self.cache.set_exact(query, response)
        await self.cache.set_semantic(query, dense_vector, response)

        return response

    async def query_stream(
        self,
        query: str,
        include_sources: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Process a query and stream the response.

        Yields:
            Dict chunks with type and content
        """
        # Layer 1: Check exact cache
        cached = await self.cache.get_exact(query)
        if cached:
            # Stream cached response
            yield {"type": "text", "content": cached.get("answer", "")}
            if include_sources and cached.get("sources"):
                yield {"type": "sources", "sources": cached["sources"]}
            if cached.get("videos"):
                yield {"type": "videos", "videos": cached["videos"]}
            yield {"type": "done", "cached": True}
            return

        # Layer 2: Generate embeddings
        dense_vector = self.embedder.embed_query(query)
        sparse_vector = self.embedder.generate_sparse_vector(query)

        # Check semantic cache
        semantic_cached = await self.cache.get_semantic(query, dense_vector)
        if semantic_cached:
            yield {"type": "text", "content": semantic_cached.get("answer", "")}
            if include_sources and semantic_cached.get("sources"):
                yield {"type": "sources", "sources": semantic_cached["sources"]}
            if semantic_cached.get("videos"):
                yield {"type": "videos", "videos": semantic_cached["videos"]}
            yield {"type": "done", "cached": True}
            return

        # Layer 3: Retrieve documents
        # Note: Academy MCP disabled until server is ready
        documents = await self.retriever.search(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        videos = []  # Empty until Academy MCP is ready

        # Layer 4: Rerank documents
        if documents and self.reranker.should_rerank(documents):
            documents = await self.reranker.rerank(query, documents)
        elif documents:
            # Just take top_k without reranking
            documents = documents[:self.settings.rerank_top_k]

        # Layer 5: Stream response
        if not documents:
            yield {
                "type": "text",
                "content": "I don't have information about this topic in the documentation. Please try rephrasing your question or check the WaveMaker documentation directly.",
            }
            yield {"type": "sources", "sources": []}
            yield {"type": "videos", "videos": []}
            yield {"type": "done", "cached": False}
            return

        # Collect full response for caching
        full_answer = ""
        sources_data = []
        videos_data = []

        async for chunk in self.generator.generate_stream(
            query=query,
            documents=documents if include_sources else documents[:3],
            videos=videos,
        ):
            yield chunk

            # Collect for caching
            if chunk.get("type") == "text":
                full_answer += chunk.get("content", "")
            elif chunk.get("type") == "sources":
                sources_data = chunk.get("sources", [])
            elif chunk.get("type") == "videos":
                videos_data = chunk.get("videos", [])

        # Cache the complete response
        response = {
            "answer": full_answer,
            "sources": sources_data,
            "videos": videos_data,
            "cached": False,
        }
        await self.cache.set_exact(query, response)
        await self.cache.set_semantic(query, dense_vector, response)

    async def close(self) -> None:
        """Clean up resources."""
        await self.cache.close()
        await self.retriever.close()
        await self.academy.close()
