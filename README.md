# Drive RAG System

## Problem

Organizations accumulate thousands of documents in Google Drive. Finding information requires manually searching through hundreds of files, often taking hours and yielding incomplete results. Traditional keyword search fails when documents use different terminology, and even when relevant documents are found, users must read through multiple files and manually synthesize information.

## Solution

This system enables natural language questions over document collections, returning accurate answers with verifiable source citations. It addresses three fundamental limitations of traditional search:

1. **Semantic understanding**: Finds documents even when they use different terminology than the query
2. **Information synthesis**: Combines information from multiple documents into coherent answers
3. **Source verification**: Every answer includes specific citations with document names and page references

## How It Works

The system implements a retrieval-augmented generation pipeline:

**Ingestion**: Documents are parsed, segmented into semantic chunks, converted to vector embeddings, and indexed for both semantic and keyword search.

**Query Processing**: User questions are expanded into multiple variations, searched using both vector similarity and keyword matching, reranked for relevance, and synthesized into answers with source citations.

**Key Design Decisions**:
- Hybrid retrieval combines semantic search with traditional keyword matching
- Two-stage retrieval (broad recall, then precision reranking) optimizes accuracy
- Structured prompts enforce source citations and prevent hallucination
- Asynchronous ingestion handles large document collections

## Documentation

- **[SETUP.md](./SETUP.md)** - Installation and configuration guide
- **[LM_STUDIO.md](./LM_STUDIO.md)** - LM Studio setup guide (beginner-friendly GUI for local LLMs)
- **[GEMINI.md](./GEMINI.md)** - Google Gemini API setup guide
- **[ITERATIVE_RAG.md](./ITERATIVE_RAG.md)** - Agentic RAG with iterative search
- **[ARCHITECTURE.md](./ARCHITECTURE.md)** - Design decisions and rationale

## Technology Stack

**Backend**: FastAPI with async request handling, PostgreSQL with pgvector for vector operations, Celery and Redis for background job processing

**RAG Components**: LangChain framework with HuggingFace embeddings (multilingual-e5-large) for local embedding generation, hybrid search combining vector similarity with BM25 full-text search, BGE cross-encoder for reranking, flexible LLM support (Ollama, OpenAI-compatible APIs like LM Studio, or Gemini) for answer generation via LangChain providers

**Fully Open Source**: No API keys required - runs entirely on local infrastructure with open source models

**LLM Options**: 
- **Ollama** (default): Free, open-source local models (Mistral, Llama, etc.)
- **LM Studio**: OpenAI-compatible API for running local models with a GUI
- **Gemini**: Google's Gemini API (requires API key from AI Studio)
- **Other OpenAI-compatible APIs**: Any service that implements the OpenAI API standard

**Evaluation**: Ragas framework for quality metrics (faithfulness, recall, precision, relevance)

## Quick Start

See [SETUP.md](./SETUP.md) for detailed installation instructions.

```bash
# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Start services
docker-compose up -d

# Verify
curl http://localhost:8000/healthz

## Usage Examples

**List available documents:**
```bash
python scripts/list_drive_files.py --folder-id YOUR_FOLDER_ID
```

**Selective ingestion:**
```bash
python scripts/list_drive_files.py --folder-id YOUR_FOLDER_ID --format csv > files.csv
# Edit files.csv to keep only desired files
python scripts/ingest_from_csv.py --csv files.csv
```

**Query via API:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "your question here", "multi_query": true}'
```

**Search without answer generation:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "search terms", "k": 20}'
```

## Prerequisites

- Docker and Docker Compose
- Google Drive Service Account credentials (for document ingestion only)
- LLM provider (choose one):
  - **Ollama** (default): Free, open-source, install from https://ollama.ai
  - **LM Studio**: User-friendly GUI for local models, download from https://lmstudio.ai
  - Any OpenAI-compatible API endpoint

No paid API services required. For detailed setup instructions, see [SETUP.md](./SETUP.md).

## Project Structure

```
app/
├── ingest/         # Google Drive integration
├── parse/          # PDF and Google Docs extraction
├── chunking/       # Semantic text segmentation
├── index/          # Vector and full-text indexing
├── retrieval/      # Hybrid search implementation
├── rerank/         # Cross-encoder reranking
├── generate/       # LLM answer generation
└── eval/           # Quality evaluation
```

## License

MIT License
