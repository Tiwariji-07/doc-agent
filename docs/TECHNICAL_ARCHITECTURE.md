# WaveMaker Docs Agent - Technical Architecture for AI Engineers

A deep dive into the design decisions, model choices, and implementation details of our RAG system.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Document Processing Pipeline](#document-processing-pipeline)
3. [Embedding Strategy](#embedding-strategy)
4. [Vector Database Design](#vector-database-design)
5. [Retrieval Strategy](#retrieval-strategy)
6. [Reranking Layer](#reranking-layer)
7. [Caching Architecture](#caching-architecture)
8. [LLM Generation](#llm-generation)
9. [Performance Optimizations](#performance-optimizations)
10. [Evaluation & Metrics](#evaluation--metrics)

---

## System Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INDEXING PIPELINE                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │  Clone   │→ │  Parse   │→ │  Chunk   │→ │  Embed   │→ │  Store   │  │
│  │  Repo    │   │  Markdown│   │  Docs    │   │  Chunks  │   │  Qdrant  │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                               QUERY PIPELINE                                 │
│                                                                              │
│   Query → Cache Check → Embed → Retrieve → Rerank → Generate → Stream       │
│              ↓             ↓        ↓          ↓         ↓                   │
│           Redis       BGE-base   Qdrant     Jina      Claude                 │
│                       (768-dim)  (HNSW)     API       (Sonnet)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```python
# Simplified flow
class RAGPipeline:
    async def query(self, query: str) -> AsyncGenerator[dict, None]:
        # L1: Cache
        if cached := await self.cache.get_semantic(query):
            yield cached; return
        
        # L2: Embed
        query_vector = self.embedder.embed(query)  # BGE → 768-dim
        
        # L3: Retrieve
        docs = await self.retriever.search(query_vector)  # Top 30
        
        # L4: Rerank
        docs = await self.reranker.rerank(query, docs)  # Top 5
        
        # L5: Generate
        async for chunk in self.generator.stream(query, docs):
            yield chunk
```

---

## Document Processing Pipeline

### Chunking Strategy

**Challenge:** Documents vary from 100 to 10,000+ tokens. LLMs have context limits. How do we split effectively?

**Our approach: Semantic-aware hierarchical chunking**

```python
# Configuration
CHUNK_MIN_TOKENS = 100   # Avoid tiny, context-less chunks
CHUNK_MAX_TOKENS = 512   # Stay within model context windows
CHUNK_TARGET_TOKENS = 350  # Optimal for retrieval quality

# Algorithm
def chunk_document(content: str, metadata: dict) -> list[Chunk]:
    chunks = []
    
    # 1. Split by headers (preserve semantic units)
    sections = split_by_headers(content)  # H1 > H2 > H3
    
    # 2. For each section, apply token-aware splitting
    for section in sections:
        if token_count(section) <= CHUNK_MAX_TOKENS:
            chunks.append(section)
        else:
            # Split on paragraph boundaries, respecting min/max
            chunks.extend(split_paragraphs(section))
    
    # 3. Sliding window overlap for context continuity
    chunks = add_overlap(chunks, overlap_tokens=50)
    
    return chunks
```

**Why these parameters?**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `min_tokens` | 100 | Smaller chunks lack context, hurt retrieval quality |
| `max_tokens` | 512 | BERT-based models (like BGE) trained on 512 tokens max |
| `target_tokens` | 350 | Sweet spot: enough context, room for overlap |
| `overlap` | 50 | Prevents losing context at chunk boundaries |

### Metadata Extraction

Each chunk carries rich metadata for filtering and citation:

```python
@dataclass
class ChunkMetadata:
    # Document identification
    doc_id: str           # Unique document hash
    file_path: str        # learn/app-development/widgets/button.md
    
    # Content hierarchy
    title: str            # "Button Widget"
    section: Optional[str]  # "Properties > Styling"
    parent_sections: list[str]  # ["Button Widget", "Properties"]
    
    # URL construction
    url_slug: str         # button-widget
    url_hash: Optional[str]  # #styling
    
    # Chunk position
    chunk_index: int      # 0, 1, 2, ... for ordering
    total_chunks: int     # Total chunks in document
```

---

## Embedding Strategy

### Model Selection: BAAI/bge-base-en-v1.5

**Why this model?**

| Criterion | BGE-base | OpenAI ada-002 | Cohere embed-v3 |
|-----------|----------|----------------|-----------------|
| **Quality (MTEB)** | 63.55 | 61.0 | 64.5 |
| **Dimensions** | 768 | 1536 | 1024 |
| **Cost** | Free (local) | $0.0001/1K tokens | $0.0001/1K tokens |
| **Latency** | ~50ms (local) | ~200ms (API) | ~150ms (API) |
| **Privacy** | Full control | Data sent to OpenAI | Data sent to Cohere |

**Decision:** BGE-base offers best quality/cost tradeoff for our use case (technical English docs).

### Embedding Configuration

```python
class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(
            "BAAI/bge-base-en-v1.5",
            device="cpu",  # MPS/CUDA for GPU acceleration
        )
        self.instruction = "Represent this sentence for searching relevant passages:"
    
    def embed_query(self, query: str) -> list[float]:
        # BGE requires instruction prefix for queries (not documents)
        prefixed = f"{self.instruction} {query}"
        return self.model.encode(prefixed).tolist()
    
    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        # Documents don't need instruction prefix
        return self.model.encode(docs).tolist()
```

### Dimensionality Trade-offs

```
768 dimensions (BGE-base):
├── Memory: 768 * 4 bytes = 3KB per vector
├── 10,000 docs = ~30MB vector storage
├── HNSW index overhead: ~3x = ~90MB total
└── Search latency: O(log n) with HNSW

1536 dimensions (OpenAI):
├── Memory: 1536 * 4 bytes = 6KB per vector  
├── 10,000 docs = ~60MB vector storage
├── More nuanced semantic space
└── Diminishing returns for our corpus size
```

---

## Vector Database Design

### Qdrant Configuration

```python
# Collection schema
collection_config = {
    "vectors": {
        "dense": {
            "size": 768,
            "distance": "Cosine",  # Normalized similarity
        }
    },
    "optimizers_config": {
        "default_segment_number": 2,  # For small collections
        "indexing_threshold": 10000,  # Start indexing at 10k docs
    },
    "hnsw_config": {
        "m": 16,              # Connections per node
        "ef_construct": 100,  # Build-time accuracy
        "full_scan_threshold": 1000,  # Below this, brute force
    }
}
```

### HNSW Index Parameters

**m (connections per layer):**
```
m=4:  Faster insert, less memory, lower recall
m=16: Good balance (our choice)
m=64: Higher recall, slower builds, more memory
```

**ef_construct (build-time beam width):**
```
ef_construct=50:  Faster indexing, lower quality
ef_construct=100: Good quality (our choice)
ef_construct=500: High quality, slow builds
```

**ef (search-time beam width):**
```python
# Higher ef = slower but more accurate search
search_params = {
    "exact": False,
    "hnsw_ef": 128,  # Search-time quality
}
```

### Payload Design

```python
# What we store alongside vectors
payload = {
    # For filtering
    "source": "docs",  # docs | academy (future)
    "category": "widgets",
    
    # For reconstruction
    "content": "The Button widget allows...",
    "title": "Button Widget",
    "url": "https://docs.wavemaker.com/...",
    "section": "Properties",
    
    # For ranking
    "chunk_index": 0,
    "doc_importance": 0.8,  # Based on page views (future)
}
```

---

## Retrieval Strategy

### Hybrid Search (Dense + Sparse)

**Why hybrid?**

| Query Type | Dense (Semantic) | Sparse (Lexical) | Best For |
|------------|-----------------|------------------|----------|
| "How to create API" | ✅ Understands intent | ❌ Misses synonyms | Dense |
| "AIRA" (abbreviation) | ❌ No semantic meaning | ✅ Exact match | Sparse |
| "Configure DB_HOST" | ❌ Looks like noise | ✅ Token match | Sparse |

**Implementation:**

```python
async def hybrid_search(
    query: str,
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    top_k: int = 30,
) -> list[Document]:
    # Parallel search
    dense_results = await qdrant.search(
        vector=dense_vector,
        limit=top_k,
        with_payload=True,
    )
    
    sparse_results = await qdrant.search(
        vector=SparseVector(sparse_vector),
        limit=top_k,
        with_payload=True,
    )
    
    # Reciprocal Rank Fusion
    return reciprocal_rank_fusion(dense_results, sparse_results, k=60)
```

### Reciprocal Rank Fusion (RRF)

**Formula:**
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where:
- `k` = 60 (constant, prevents high ranks from dominating)
- `rank_i(d)` = rank of document `d` in result list `i`

**Example:**
```python
def reciprocal_rank_fusion(
    results_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    scores = defaultdict(float)
    
    for results in results_lists:
        for rank, doc in enumerate(results, 1):
            scores[doc.id] += 1.0 / (k + rank)
    
    # Sort by RRF score
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [get_doc(doc_id) for doc_id, score in fused]

# Example calculation:
# Doc A: Rank 1 in dense, Rank 5 in sparse
# RRF(A) = 1/(60+1) + 1/(60+5) = 0.0164 + 0.0154 = 0.0318

# Doc B: Rank 3 in dense, Rank 2 in sparse  
# RRF(B) = 1/(60+3) + 1/(60+2) = 0.0159 + 0.0161 = 0.0320

# Result: Doc B > Doc A (despite A ranking higher in dense)
```

### Why k=60?

```
k=1:   Top ranks dominate heavily
k=60:  Standard choice, balanced weighting (our choice)
k=1000: Nearly uniform weighting across all ranks
```

---

## Reranking Layer

### Cross-Encoder vs Bi-Encoder

```
BI-ENCODER (Embedding Search):
┌─────────┐     ┌─────────┐
│  Query  │     │   Doc   │
└────┬────┘     └────┬────┘
     │               │
     ▼               ▼
┌─────────┐     ┌─────────┐
│ Encoder │     │ Encoder │   ← Same model, separate encoding
└────┬────┘     └────┬────┘
     │               │
     ▼               ▼
  [0.2, 0.8]    [0.3, 0.7]   ← Compare with cosine similarity
     │               │
     └───────────────┘
           ↓
     similarity = 0.95

Pros: Fast (encode once, compare many)
Cons: Approximate (no cross-attention)


CROSS-ENCODER (Reranking):
┌─────────────────────────┐
│   [CLS] Query [SEP] Doc │   ← Concatenated input
└───────────┬─────────────┘
            │
            ▼
      ┌───────────┐
      │ Encoder   │   ← Full attention between Q and D
      └─────┬─────┘
            │
            ▼
    relevance_score = 0.82

Pros: Precise (full cross-attention)
Cons: Slow (O(n) for n docs)
```

### Jina Reranker API

**Why Jina over local cross-encoder?**

| Factor | Local (ms-marco-MiniLM) | Jina API |
|--------|------------------------|----------|
| **Apple Silicon** | MLX causes NaN scores | ✅ Works |
| **Quality** | Good | Excellent (multilingual) |
| **Latency** | ~500ms (30 docs) | ~500ms (30 docs) |
| **Cost** | Free | ~$0.0001/call |

**Implementation:**

```python
async def rerank_jina(
    query: str,
    documents: list[str],
    top_n: int = 5,
) -> list[tuple[int, float]]:
    response = await httpx.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": [d[:2000] for d in documents],  # Truncate
        },
    )
    
    return [
        (r["index"], r["relevance_score"])
        for r in response.json()["results"]
    ]
```

### Reranking Threshold

```python
def should_rerank(documents: list[Document]) -> bool:
    """Skip reranking if top result is already very confident."""
    if not documents:
        return False
    
    top_rrf_score = documents[0].rrf_score
    
    # High RRF score = dense and sparse agree strongly
    if top_rrf_score > 0.04:  # ~top 25 in both lists
        logger.debug(f"Skipping rerank: confident (RRF={top_rrf_score:.4f})")
        return False
    
    return True
```

---

## Caching Architecture

### Three-Tier Cache Design

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CACHE LAYER                                      │
│                                                                               │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐        │
│  │  TIER 1: EXACT    │  │  TIER 2: SEMANTIC │  │  TIER 3: EMBEDDING│        │
│  │                   │  │                   │  │                   │        │
│  │  hash(query)→resp │  │  vector:resp pairs│  │  text→embedding   │        │
│  │                   │  │  sim(q,cached)>θ  │  │                   │        │
│  │  O(1) lookup      │  │  Linear scan      │  │  Skip re-compute  │        │
│  │  100% match only  │  │  95%+ similarity  │  │  Reuse embeddings │        │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘        │
│                                                                               │
│  Key: exact:{sha256}     Key: semantic:{sha256}   Key: embed:{sha256}        │
│  TTL: 1 hour             TTL: 1 hour              TTL: 24 hours              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Exact Match Cache

```python
async def get_exact(self, query: str) -> Optional[dict]:
    """O(1) lookup using query hash."""
    normalized = " ".join(query.lower().strip().split())
    key = f"exact:{sha256(normalized)[:32]}"
    return await redis.get(key)
```

### Tier 2: Semantic Cache

```python
async def get_semantic(
    self,
    query: str,
    query_embedding: np.ndarray,
) -> Optional[dict]:
    """Find semantically similar cached query."""
    threshold = 0.95
    
    # Scan cached embeddings (limited to 100 for performance)
    for cached in await redis.keys("semantic:*")[:100]:
        cached_embedding = cached["embedding"]
        similarity = cosine_similarity(query_embedding, cached_embedding)
        
        if similarity >= threshold:
            return cached["response"]
    
    return None
```

### Tier 3: Embedding Cache

```python
async def get_embedding(self, text: str) -> Optional[np.ndarray]:
    """Reuse previously computed embeddings."""
    key = f"embed:{sha256(text)[:32]}"
    cached = await redis.get(key)
    if cached:
        return np.array(json.loads(cached))
    return None

async def set_embedding(self, text: str, embedding: np.ndarray) -> None:
    """Cache embedding with 24x longer TTL (embeddings rarely change)."""
    await redis.setex(
        key=f"embed:{sha256(text)[:32]}",
        ttl=self.ttl * 24,  # 24 hours instead of 1 hour
        value=embedding.tolist(),
    )
```

### Cache Hit Priority

```
Query: "How to create REST API?"
         │
         ▼
┌─────────────────────────┐
│  Tier 1: Exact Match?   │ ← Hash lookup, O(1)
│  exact:{hash} exists?   │
└───────────┬─────────────┘
            │ Miss
            ▼
┌─────────────────────────┐
│  Tier 2: Semantic?      │ ← Vector similarity scan
│  Any cache sim > 0.95?  │
└───────────┬─────────────┘
            │ Miss
            ▼
┌─────────────────────────┐
│  Tier 3: Embedding?     │ ← Avoid re-computing embedding
│  embed:{hash} exists?   │
└───────────┬─────────────┘
            │ Miss/Hit
            ▼
      Continue to retrieval
      (with cached or new embedding)
```

### Why 0.95 Threshold?

```
Threshold = 0.90: Too loose
  "How to create API" ≈ "How to delete API" (0.91)  ← Wrong!
  
Threshold = 0.95: Good balance (our choice)
  "How to create API" ≈ "How do I create an API?" (0.96)  ← Same intent
  "How to create API" ≈ "Creating APIs in WaveMaker" (0.94)  ← Below threshold
  
Threshold = 0.99: Too strict
  "How to create API" ≈ "How to create API?" (0.995)  ← Only punctuation diff
```

---

## LLM Generation

### System Prompt Design

```python
SYSTEM_PROMPT = """You are an expert WaveMaker documentation assistant.
Your role is to help developers understand and use WaveMaker effectively.

## Guidelines

1. **Answer from context only** - Use only the provided documentation excerpts
2. **Cite sources** - Reference documents as [1], [2], etc.
3. **Be precise** - Give specific steps, code examples when relevant
4. **Acknowledge limitations** - If context is insufficient, say so
5. **Format clearly** - Use markdown for readability

## Response Format

- Start with a direct answer
- Provide step-by-step instructions if applicable
- Include code snippets with proper formatting
- End with source citations
"""
```

### Context Formatting

```python
def format_context(documents: list[Document]) -> str:
    """Format documents for LLM consumption."""
    parts = ["## Documentation Context\n"]
    
    for i, doc in enumerate(documents, 1):
        parts.append(f"""
### [{i}] {doc.title}
**Source:** {doc.url}
{f"**Section:** {doc.section}" if doc.section else ""}

{doc.content}

---
""")
    
    return "\n".join(parts)
```

### Streaming Implementation

```python
async def generate_stream(
    query: str,
    documents: list[Document],
) -> AsyncGenerator[dict, None]:
    client = anthropic.AsyncAnthropic()
    
    async with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        temperature=0.2,  # Low for factual answers
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": format_prompt(query, documents)}
        ],
    ) as stream:
        async for text in stream.text_stream:
            yield {"type": "text", "content": text}
    
    # After streaming, yield sources
    yield {
        "type": "sources",
        "sources": [doc.to_source() for doc in documents]
    }
    yield {"type": "done", "cached": False}
```

### Temperature Selection

```
temperature = 0.0: Deterministic, may miss nuance
temperature = 0.2: Slight variation, mostly factual (our choice)
temperature = 0.7: Creative, may hallucinate
temperature = 1.0: Very creative, unreliable for docs
```

---

## Performance Optimizations

### Lazy Model Loading

```python
class Embedder:
    """Lazy-load model on first use, not app startup."""
    
    def __init__(self):
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            logger.info("Loading embedding model...")
            self._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return self._model
```

### Connection Pooling

```python
# Reuse HTTP connections
class RerankerClient:
    def __init__(self):
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )
        return self._client
```

### Batch Processing (Indexing)

```python
async def index_documents(chunks: list[Chunk], batch_size: int = 100):
    """Batch embed and upsert for efficiency."""
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        # Batch embed (much faster than one-by-one)
        vectors = embedder.encode([c.content for c in batch])
        
        # Batch upsert
        points = [
            PointStruct(id=c.id, vector=v, payload=c.metadata)
            for c, v in zip(batch, vectors)
        ]
        await qdrant.upsert(collection, points)
```

---

## Evaluation & Metrics

### Key Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Recall@K** | % of relevant docs in top K results | > 90% at K=30 |
| **MRR** | Mean Reciprocal Rank of first relevant | > 0.7 |
| **NDCG@5** | Normalized DCG for ranked results | > 0.8 |
| **Latency P95** | 95th percentile response time | < 5s |
| **Cache Hit Rate** | % queries served from cache | > 40% |

### Evaluation Dataset

```python
# eval/test_queries.json
{
    "queries": [
        {
            "query": "How to create a new page in WaveMaker?",
            "relevant_docs": ["pages-overview.md", "create-page.md"],
            "ideal_answer_keywords": ["Pages", "New", "Markup"]
        },
        # ... more test cases
    ]
}
```

### A/B Testing Configuration

```python
# For experimenting with model changes
EXPERIMENT_CONFIG = {
    "baseline": {
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "reranker": "jina",
        "top_k_retrieve": 30,
        "top_k_rerank": 5,
    },
    "treatment": {
        "embedding_model": "intfloat/e5-large-v2",
        "reranker": "jina",
        "top_k_retrieve": 50,
        "top_k_rerank": 7,
    },
    "traffic_split": 0.1,  # 10% to treatment
}
```

---

## Future Improvements

### Planned Enhancements

1. **Query Understanding**
   - Intent classification (how-to vs. conceptual vs. troubleshooting)
   - Query expansion with synonyms

2. **Hybrid Retrieval v2**
   - Add sparse embeddings (SPLADE)
   - Multi-vector retrieval (ColBERT)

3. **Adaptive Reranking**
   - Skip reranking for high-confidence retrievals
   - Use lighter model for simple queries

4. **Response Quality**
   - Add answer validation (check citations exist)
   - Implement confidence scoring

5. **Observability**
   - Trace each layer with OpenTelemetry
   - Log retrieval quality metrics

---

## References

- [BGE Embeddings Paper](https://arxiv.org/abs/2309.07597)
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Cross-Encoders for Reranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
