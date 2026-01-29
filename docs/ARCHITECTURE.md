# WaveMaker Docs Agent - Architecture Guide

A simple explanation of how our AI documentation assistant works, layer by layer.

---

## 🎯 The Problem We're Solving

**User asks:** "How do I create a REST API in WaveMaker?"

**Challenge:** We have 500+ documentation pages. How do we:
1. Find the 3-5 most relevant pages out of 500+?
2. Give the AI only those pages (LLMs have token limits)?
3. Generate an accurate, cited answer?

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                   │
│                 "How to create a REST API?"                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: CACHE                                                      │
│  ┌──────────────┐    ┌──────────────┐                               │
│  │ Exact Match  │ OR │  Semantic    │  ← "Same question asked       │
│  │ Cache        │    │  Cache       │     before? Return cached!"   │
│  └──────────────┘    └──────────────┘                               │
└─────────────────────────────────────────────────────────────────────┘
                                │ Cache Miss
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: EMBEDDINGS                                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  "How to create REST API" → [0.23, -0.45, 0.89, ...]         │   │
│  │                              (768-dimensional vector)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  Model: BAAI/bge-base-en-v1.5 (runs locally, free)                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: RETRIEVAL                                                  │
│  ┌─────────────────┐                                                │
│  │  Qdrant Cloud   │  ← Vector similarity search                    │
│  │  (Vector DB)    │  ← Returns top 30 similar docs                 │
│  └─────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────┘
                                │ 30 documents
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 4: RERANKING                                                  │
│  ┌─────────────────┐                                                │
│  │   Jina Rerank   │  ← Precise relevance scoring                   │
│  │   (API)         │  ← 30 docs → 5 best docs                       │
│  └─────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────┘
                                │ 5 documents
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 5: GENERATION                                                 │
│  ┌─────────────────┐                                                │
│  │  Claude LLM     │  ← Reads 5 docs + question                     │
│  │  (Anthropic)    │  ← Generates answer with citations [1][2]      │
│  └─────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         RESPONSE                                     │
│  "To create a REST API in WaveMaker, navigate to APIs → REST →     │
│   New [1]. Define your endpoints and methods [2]..."                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Layer-by-Layer Explanation

### Layer 1: Cache (Redis)

**What it does:** Remembers previous answers to avoid re-processing.

**Why we need it:**
- LLM calls cost money ($0.003-0.01 per query)
- Same questions get asked repeatedly
- Faster response time (~50ms vs ~3s)

**Two types of caching:**

| Type | How it Works | Example |
|------|-------------|---------|
| **Exact Cache** | Hash of question → answer | "What is AIRA?" matches "What is AIRA?" |
| **Semantic Cache** | Similar meaning → answer | "What is AIRA?" matches "Tell me about AIRA" |

**Example:**
```
User 1: "What is AIRA?" → Processed, cached
User 2: "Tell me about AIRA" → Semantic match (95% similar) → Return cached
```

**Technology:** Redis (fast in-memory database)

---

### Layer 2: Embeddings (BAAI/bge-base-en-v1.5)

**What it does:** Converts text into numbers that capture meaning.

**Why we need it:**
- Computers can't understand "meaning" directly
- Numbers allow mathematical similarity comparison
- Similar meanings → similar numbers

**How it works:**
```
"How to create REST API"     → [0.23, -0.45, 0.89, ... 768 numbers]
"Creating RESTful services"  → [0.21, -0.43, 0.91, ... 768 numbers]
                                     ↑
                              Very similar vectors! (cosine similarity ~0.95)

"What is database security?" → [0.67, 0.12, -0.34, ... 768 numbers]
                                     ↑
                              Very different vector (cosine similarity ~0.3)
```

**Why bge-base-en-v1.5?**
- High quality for technical docs
- Runs locally (no API cost)
- Fast (768-dim is good balance of quality/speed)

---

### Layer 3: Retrieval (Qdrant Vector Database)

**What it does:** Finds the 30 most similar documents from 500+ indexed docs.

**Why we need it:**
- Can't send all 500 docs to LLM (too expensive, hits token limits)
- Vector search is fast (milliseconds for 10,000+ docs)
- Gets "approximately right" candidates quickly

**How it works:**
```
Query Vector: [0.23, -0.45, 0.89, ...]
                    ↓
              Qdrant Search
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ Results (sorted by similarity):                             │
│  1. "REST API Development" (score: 0.94)                    │
│  2. "API Endpoints Guide" (score: 0.91)                     │
│  3. "Creating Services" (score: 0.88)                       │
│  ... 27 more docs ...                                       │
└─────────────────────────────────────────────────────────────┘
```

**Why Qdrant Cloud?**
- Managed service (no server maintenance)
- Fast vector search at scale
- Free tier available for development

**Why 30 documents?**
- More candidates = better chance of finding best matches
- Will be filtered down by reranking next

---

### Layer 4: Reranking (Jina AI)

**What it does:** Precisely re-scores 30 docs to find the 5 best.

**Why we need it:**
- Embedding search is "approximate" - fast but not perfectly accurate
- Cross-encoder reranking is slow but much more accurate
- We only have 5 slots to give to the LLM

**The difference:**

| Method | Speed | Accuracy | Use Case |
|--------|-------|----------|----------|
| **Embedding Search** | ~10ms for 10K docs | Good | Initial filtering |
| **Cross-Encoder Rerank** | ~500ms for 30 docs | Excellent | Final selection |

**Example:**
```
Before Reranking (30 docs, embedding scores):
  1. "REST API Development" (0.94) ← Looks best
  2. "API Endpoints Guide" (0.91)
  3. "Creating Services" (0.88)
  4. "Database REST API" (0.85)  ← Actually most relevant!
  ...

After Reranking (Jina scores):
  1. "Database REST API" (0.82)  ← Jina found this is best match
  2. "REST API Development" (0.76)
  3. "API Endpoints Guide" (0.71)
  4. "Creating Services" (0.45)
  5. "Import APIs" (0.42)
```

**Why Jina AI?**
- High quality multilingual reranker
- API-based (no local GPU needed)
- Works around MLX issues on Apple Silicon

---

### Layer 5: Generation (Claude LLM)

**What it does:** Reads the 5 documents and generates a helpful answer.

**Why we need it:**
- Humans don't want to read 5 documents
- LLM synthesizes information into clear answer
- Adds structure, examples, and citations

**The prompt structure:**
```
SYSTEM: You are a WaveMaker documentation expert. Answer using 
        the provided context and cite sources using [1], [2], etc.

CONTEXT:
[1] REST API Development - "To create a REST API, go to..."
[2] API Endpoints Guide - "You can define GET, POST methods..."
[3] Creating Services - "Services in WaveMaker include..."
[4] Database REST API - "Database APIs auto-generate CRUD..."
[5] Import APIs - "External APIs can be imported..."

QUESTION: How do I create a REST API in WaveMaker?
```

**Response:**
```
To create a REST API in WaveMaker, follow these steps:

1. Navigate to **APIs → REST → New** [1]
2. Define your endpoint path and HTTP method (GET, POST, etc.) [2]
3. For database entities, WaveMaker auto-generates CRUD APIs [4]

The API will be available at `/rest/your-endpoint` [1][2].
```

**Why Claude?**
- Excellent at following instructions
- High quality, accurate responses
- Good at citing sources

---

## 🔄 Complete Example Flow

**User Query:** "How do I add authentication to my WaveMaker app?"

```
1. CACHE CHECK
   ├─ Exact match? No
   └─ Semantic match? No (new question)

2. EMBEDDING
   └─ Query → [0.12, 0.89, -0.34, ...] (768 dims)

3. RETRIEVAL (Qdrant)
   └─ Returns 30 docs about security, authentication, users...

4. RERANKING (Jina)
   ├─ Input: 30 documents
   └─ Output: Top 5 most relevant
      1. "WaveMaker Security Configuration" (0.89)
      2. "OpenID Authentication" (0.82)
      3. "SAML SSO Setup" (0.78)
      4. "Role-Based Access Control" (0.71)
      5. "Login Page Customization" (0.65)

5. GENERATION (Claude)
   └─ Reads 5 docs, generates:
      
      "To add authentication to your WaveMaker app:
      
      1. Go to **Security** settings [1]
      2. Choose a provider:
         - OpenID/OAuth for Google, Auth0 [2]
         - SAML for enterprise SSO [3]
      3. Configure roles for access control [4]
      
      The login page is auto-generated but customizable [5]."

6. CACHE STORE
   └─ Save response for future similar questions
```

---

## 📊 Cost & Performance Summary

| Layer | Cost | Latency | Notes |
|-------|------|---------|-------|
| Cache | Free | ~5ms | Redis (local or managed) |
| Embedding | Free | ~100ms | Runs locally |
| Retrieval | ~$0 | ~200ms | Qdrant free tier |
| Reranking | ~$0.0001 | ~500ms | Jina API |
| Generation | ~$0.005 | ~2-3s | Claude API |
| **Total** | **~$0.005/query** | **~3-4s** | (cached: free, ~50ms) |

---

## 🛠️ Why This Architecture?

### The RAG Pattern (Retrieval-Augmented Generation)

Our architecture follows the **RAG pattern**:

```
Traditional LLM:  Question → LLM → Answer (might hallucinate)

RAG:              Question → Retrieve Docs → LLM + Docs → Grounded Answer
```

**Benefits:**
1. **Accurate** - LLM only answers from real documentation
2. **Up-to-date** - Re-index docs when content changes
3. **Verifiable** - Citations point to source documents
4. **Cost-effective** - Only process relevant docs, not entire corpus

### Why Not Just Use LLM's Knowledge?

| Approach | Pros | Cons |
|----------|------|------|
| **LLM Knowledge** | Simple | Outdated, may hallucinate, no citations |
| **RAG (our approach)** | Accurate, current, cited | More complex |

---

## 🧩 Component Choices Summary

| Component | Choice | Why |
|-----------|--------|-----|
| **Cache** | Redis | Fast, simple, semantic search support |
| **Embedding** | bge-base-en-v1.5 | Free, local, high quality |
| **Vector DB** | Qdrant Cloud | Managed, fast, free tier |
| **Reranker** | Jina AI | API-based (avoids local GPU issues) |
| **LLM** | Claude | High quality, follows instructions well |
| **API** | FastAPI | Async, fast, streaming support |

---

## 🎓 Key Takeaways for Developers

1. **Embeddings are dimensionality reduction** 
   - Text → Fixed-size numbers that capture meaning

2. **Two-stage retrieval is optimal**
   - Fast approximate search first (vectors)
   - Slow precise rerank second (cross-encoder)

3. **Caching saves money and time**
   - Most questions are repeats or similar

4. **LLMs need context**
   - Don't ask LLM to know everything
   - Give it relevant documents to read

5. **Citations build trust**
   - Users can verify answers against sources
