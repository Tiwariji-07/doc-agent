"""
Embedder module for generating dense and sparse vectors.

Uses sentence-transformers for dense embeddings (BGE-base)
and basic BM25-style tokenization for sparse vectors.
"""

import logging
import re
from collections import Counter
from functools import lru_cache
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates dense and sparse embeddings for queries and documents.
    """

    def __init__(self):
        self.settings = get_settings()
        self._model: Optional[SentenceTransformer] = None
        self._stopwords = self._get_stopwords()

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device="cpu",  # Use CPU for production stability
            )
            logger.info("Embedding model loaded successfully")
        return self._model

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate dense embedding for a query.
        BGE models require a prefix for better retrieval.
        Runs in a thread pool to avoid blocking the event loop.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        def _compute():
            model = self._get_model()
            prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
            embedding = model.encode(
                prefixed_query,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.array(embedding, dtype=np.float32)

        return await loop.run_in_executor(None, _compute)

    async def embed_document(self, document: str) -> np.ndarray:
        """
        Generate dense embedding for a document.
        Runs in a thread pool.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        def _compute():
            model = self._get_model()
            embedding = model.encode(
                document,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.array(embedding, dtype=np.float32)

        return await loop.run_in_executor(None, _compute)

    async def embed_documents_batch(
        self,
        documents: list[str],
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        """
        Generate embeddings for multiple documents in batches.
        Runs in a thread pool.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        def _compute():
            model = self._get_model()
            embeddings = model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=True,
                batch_size=batch_size,
            )
            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        return await loop.run_in_executor(None, _compute)

    def generate_sparse_vector(self, text: str) -> dict[int, float]:
        """
        Generate BM25-style sparse vector from text.
        Returns dict of {token_index: weight} using hash for indices.
        FAST enough to remain CPU-bound sync, but can be offloaded if needed.
        """
        # Tokenize
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        # Count term frequencies
        term_counts = Counter(tokens)
        total_terms = len(tokens)

        # Calculate weights (simplified TF-IDF)
        sparse_vector = {}
        for term, count in term_counts.items():
            # Skip stopwords
            if term in self._stopwords:
                continue
            # Simple TF weighting with saturation
            tf = count / total_terms
            # Terms with higher counts get diminishing returns
            weight = min(1.0, tf * 5)  # Cap at 1.0
            if weight > 0.1:  # Only include significant terms
                # Use deterministic hash (hashlib instead of Python's hash())
                # Python's hash() is randomized per session (PYTHONHASHSEED)
                import hashlib
                token_hash = hashlib.md5(term.encode()).hexdigest()
                token_idx = int(token_hash[:8], 16) % 100000  # First 8 hex chars
                sparse_vector[token_idx] = round(weight, 3)

        return sparse_vector

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, alphanumeric only."""
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r"\b[a-z0-9]+\b", text)
        return tokens

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_stopwords() -> set[str]:
        """Get English stopwords."""
        return {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
            "who", "when", "where", "why", "how", "all", "each", "every", "both",
            "few", "more", "most", "other", "some", "such", "no", "not", "only",
            "same", "so", "than", "too", "very", "just", "also", "now", "here",
            "there", "then", "if", "about", "into", "through", "during", "before",
            "after", "above", "below", "up", "down", "out", "off", "over", "under",
            "again", "further", "once", "any", "your", "our", "my", "his", "her",
        }

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
