# WaveMaker Docs Agent

AI-powered documentation assistant for WaveMaker, built with a 5-layer RAG architecture.

## Features

- 🔍 **Hybrid Search**: Combines dense (semantic) and sparse (keyword) search with RRF fusion
- 📚 **Smart Chunking**: Semantically splits documentation by headers
- 🚀 **3-Tier Caching**: Redis-based caching for fast responses
- 🎯 **Cross-Encoder Reranking**: Improves relevance of search results
- 💬 **Streaming Responses**: Real-time answer generation with Claude
- 📖 **Inline Citations**: Every answer includes source references

## Architecture

```
User Query
     │
     ▼
┌─────────────┐     ┌─────────────┐
│  FastAPI    │────▶│  Pipeline   │
│  /api/chat  │     │  Orchestr.  │
└─────────────┘     └──────┬──────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
┌─────────┐         ┌─────────────┐       ┌───────────┐
│  Redis  │         │   Qdrant    │       │  Claude   │
│  Cache  │         │   Search    │       │  Sonnet   │
└─────────┘         └─────────────┘       └───────────┘
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your values:
# - ANTHROPIC_API_KEY
# - QDRANT_URL
# - QDRANT_API_KEY
# - REDIS_URL (optional, defaults to localhost)
```

### 3. Index Documentation

```bash
# Index WaveMaker docs from GitHub
python scripts/index_docs.py

# Or force full reindex
python scripts/index_docs.py --force

# Or use local docs
python scripts/index_docs.py --local /path/to/wavemaker/docs
```

### 4. Start the Server

```bash
# Development mode
uvicorn src.main:app --reload --port 8000

# Or production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 5. Test the API

```bash
# Health check
curl http://localhost:8000/api/health

# Ask a question (streaming)
curl -N -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I create a REST API in WaveMaker?"}'

# Ask a question (non-streaming)
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a Live Variable?", "stream": false}'
```

## API Endpoints

### POST /api/chat

Ask a question about WaveMaker documentation.

**Request:**
```json
{
  "query": "How do I create a REST API?",
  "stream": true,
  "include_sources": true
}
```

**Response (streaming):**
```
data: {"type": "text", "content": "To create a REST API..."}
data: {"type": "text", "content": " in WaveMaker..."}
data: {"type": "sources", "sources": [...]}
data: {"type": "done", "cached": false}
```

**Response (non-streaming):**
```json
{
  "answer": "To create a REST API in WaveMaker...",
  "sources": [
    {"id": 1, "title": "REST Variables", "url": "..."}
  ],
  "videos": [],
  "cached": false
}
```

### GET /api/health

Check service health status.

### POST /api/index

Trigger document re-indexing.

```json
{
  "force_reindex": false,
  "branch": "release-12"
}
```

## Project Structure

```
docs-agent/
├── src/
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Pydantic schemas
│   ├── core/
│   │   ├── pipeline.py      # Main RAG orchestrator
│   │   ├── cache.py         # Redis caching
│   │   ├── embedder.py      # Embedding generation
│   │   ├── retriever.py     # Qdrant hybrid search
│   │   ├── reranker.py      # Cross-encoder reranking
│   │   ├── generator.py     # Claude response generation
│   │   └── academy.py       # Academy MCP client
│   ├── indexer/
│   │   ├── parser.py        # Markdown parsing
│   │   ├── chunker.py       # Semantic chunking
│   │   └── indexer.py       # Document indexing
│   └── config/
│       └── settings.py      # Configuration
├── scripts/
│   └── index_docs.py        # Indexing CLI
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

All configuration is done via environment variables. See `.env.example` for all options.

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | Required |
| `QDRANT_URL` | Qdrant Cloud URL | Required |
| `QDRANT_API_KEY` | Qdrant API key | Required |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |
| `DOCS_BRANCH` | Git branch to index | `release-12` |
| `LLM_MODEL` | Claude model to use | `claude-sonnet-4-5-20250929` |
| `LLM_TEMPERATURE` | Generation temperature | `0.2` |

## Development

```bash
# Run tests
pytest

# Run with debug logging
DEBUG=true uvicorn src.main:app --reload
```

## Roadmap

- [ ] Academy video transcript integration (Phase 2)
- [ ] Conversation history/memory
- [ ] GitHub Actions for auto-indexing
- [ ] Docusaurus chat widget component
- [ ] Query analytics dashboard

## License

MIT
