# System Architecture

## Design Philosophy

This system is built entirely with open source components, requiring no proprietary API keys or paid services. It prioritizes three objectives:

1. **Answer accuracy**: Responses must be grounded in source documents with verifiable citations
2. **Retrieval quality**: Search must balance semantic understanding with keyword precision
3. **Open source principles**: Full transparency, no vendor lock-in, runs on local infrastructure

## Core Architectural Decisions

### Fully Open Source Stack

**Decision**: Use only open source models and libraries - no proprietary APIs

**Rationale**: Eliminates ongoing costs, ensures data privacy (no external API calls), provides full control over model behavior, and enables offline operation. Makes the system truly free to run and fully transparent.

**Implementation**:
- **Embeddings**: HuggingFace embeddings via LangChain (multilingual-e5-large) - local execution, 1024-dim vectors
- **Reranking**: BGE cross-encoder (BAAI/bge-reranker-v2-m3) - local execution, multilingual
- **LLM**: Ollama, LM Studio, or Gemini via LangChain providers - flexible provider support
- **Database**: PostgreSQL with pgvector extension
- **Framework**: LangChain for embeddings, text splitting, LLM providers, and document loaders
- **Ingestion only**: Google Drive API (for document access, not ML)

**Trade-offs**: Higher computational requirements (requires GPU for good performance) in exchange for zero ongoing costs, complete data privacy, and no rate limits.

### Hybrid Retrieval Strategy

**Decision**: Combine vector similarity search with BM25 keyword search using Reciprocal Rank Fusion

**Rationale**: Vector embeddings capture semantic similarity (understanding "timeline" and "schedule" are related) but miss exact keyword matches. BM25 excels at keyword matching but ignores semantic relationships. Combining both approaches addresses their individual weaknesses.

**Trade-offs**: Increased query complexity and latency in exchange for significantly improved recall across diverse query types.

### Cross-Encoder Reranking

**Decision**: Apply BGE cross-encoder model (local) to rerank initial retrieval results before LLM generation

**Rationale**: Initial retrieval casts a wide net (50 candidates) for high recall. Cross-encoders provide more accurate relevance scoring but are computationally expensive. Two-stage retrieval optimizes the speed-accuracy trade-off.

**Implementation**: BGE-reranker-v2-m3 (multilingual, open source) runs locally and reduces 50 candidates to 8 high-confidence results without any API calls.

**Why BGE over Cohere**: Free, runs locally, supports 100+ languages, competitive quality, no rate limits or API costs.

### Semantic Chunking

**Decision**: Segment documents into approximately 400-token chunks with 60-token overlap, preserving document structure

**Rationale**: Chunk size balances context completeness with retrieval precision. Overlap ensures concepts spanning chunk boundaries remain retrievable. Structure preservation (page numbers, headings) enables specific citations.

**Trade-offs**: Larger chunks increase embedding and storage costs but reduce the risk of fragmenting coherent information.

### Asynchronous Ingestion

**Decision**: Process document ingestion as background jobs with progress tracking

**Rationale**: Large-scale document processing blocks API responses. Background processing with job status endpoints maintains API responsiveness while enabling long-running operations.

**Implementation**: Celery workers consume ingestion jobs from Redis queue, updating PostgreSQL job status table.

### Source Citation Enforcement

**Decision**: Structure LLM prompts to require source citations and validate responses post-generation

**Rationale**: LLMs naturally hallucinate information and citations. Explicit prompt engineering combined with structured metadata reduces hallucination. Post-generation validation confirms all cited sources exist in the context.

**Trade-offs**: More complex prompting and validation logic, but essential for trustworthy answers in enterprise contexts.

## System Components

### Ingestion Pipeline

```
Google Drive API → Document Parser → Semantic Chunker → Embedding Generator → Database Indexer
```

**Design considerations:**
- Idempotency: Multiple ingestion runs for the same document produce identical results
- Error isolation: Individual document failures do not halt batch processing
- Metadata preservation: File paths, page numbers, and modification timestamps enable citation and change detection

### Query Pipeline

```
User Query → Query Expander → Hybrid Retriever → Cross-Encoder Reranker → LLM Generator → Response Validator
```

**Design considerations:**
- Dynamic top-k: Query complexity heuristics determine optimal context size (4-16 sources)
- Query expansion: Multi-query generation and HyDE improve recall for ambiguous queries
- Prompt engineering: System prompts explicitly forbid hallucination and require citations

## Data Architecture

### Database Schema Design

**Core entities:**
- `documents`: File metadata and Drive links
- `chunks`: Text segments with structural locators (page/heading)
- `embeddings`: Vector representations with HNSW indexing
- `documents_fts`: Full-text search vectors for BM25

**Design rationale:**
- Separation of concerns: Chunks can be re-embedded without re-parsing documents
- Indexing strategy: HNSW for vector search (approximate nearest neighbors), GIN for full-text search
- Foreign key cascades: Deleting a document removes all associated chunks, embeddings, and search vectors

### Embedding Strategy

**Decision**: Use LangChain's HuggingFaceEmbeddings wrapper with multilingual-e5-large model (1024-dimensional embeddings)

**Rationale**: Excellent multilingual support (including Finnish), open source, runs locally on CPU or GPU, no API costs. The 1024-dim vectors capture nuanced semantic relationships while remaining computationally tractable. LangChain provides standardized interface and ecosystem integration.

**Alternatives considered**:
- all-MiniLM-L6-v2: Faster but English-focused, 384-dim
- BGE-large-en: English-only but excellent quality
- OpenAI/Vertex AI: Ruled out to maintain fully open source architecture

**Why multilingual-e5-large with LangChain**: Best balance of quality, multilingual support, local execution, and ecosystem compatibility. Supports Finnish and 100+ languages.

## API Design

### RESTful Endpoints

**`POST /ask`**: Question answering with full RAG pipeline

**`POST /search`**: Retrieval-only endpoint for debugging and alternative interfaces

**`POST /ingest/start`**: Asynchronous document ingestion

**`GET /ingest/status/{job_id}`**: Job progress polling

**Design considerations:**
- Separation of concerns: `/search` enables retrieval quality evaluation independent of LLM generation
- Asynchronous operations: Long-running ingestion jobs return immediately with tracking ID
- Error responses: Structured error messages with actionable guidance

## Scalability Approach

### Performance Characteristics

Query latency is dominated by reranking (200-300ms with GPU) and LLM generation (400-1000ms depending on Ollama model). Retrieval operations (hybrid search) complete within 30-40ms.

Ingestion throughput is constrained by local GPU capacity for embedding generation. With GPU: ~50-100 docs/min. Without GPU: ~5-10 docs/min.

**Hardware recommendations**:
- Development: CPU-only works but slow
- Production: NVIDIA GPU (8GB+ VRAM) for acceptable performance
- Embeddings: Batch processing amortizes GPU overhead

### Horizontal Scaling

The architecture supports horizontal scaling through:
- Stateless API servers (FastAPI instances behind load balancer)
- Worker pool scaling (Celery workers process ingestion jobs independently)
- Database read replicas (query load distribution)

Vector search performance degrades logarithmically with document count (HNSW indexing), maintaining acceptable latency beyond 100,000 documents.

## Quality Assurance

### Evaluation Framework

The system uses Ragas for automated quality evaluation across four dimensions:

1. **Faithfulness**: Do answers accurately reflect source content?
2. **Context Recall**: Does retrieval find all relevant information?
3. **Context Precision**: Are retrieved chunks actually relevant?
4. **Answer Relevancy**: Do answers address the original question?

### Error Handling Strategy

**Graceful degradation:**
- PDF parsing failures trigger fallback to alternative parser
- API rate limits trigger exponential backoff and retry
- LLM generation failures return retrieval results without synthesis

**Logging and observability:**
- Structured logging at component boundaries
- Job status tracking for asynchronous operations
- Health check endpoints for monitoring integration

## Security Architecture

### Authentication and Authorization

**Google Drive access:** Service Account with scoped, read-only permissions. Service account email must be explicitly granted access to target folders.

**API security:** Production deployments should implement API key authentication and rate limiting. Current implementation assumes trusted network (suitable for internal tools).

### Data Privacy

**Stored data:** Document text and metadata only. No user-identifying information.

**Secret management:** Environment variables and volume-mounted secrets. Production should use managed secret services (Google Secret Manager, AWS Secrets Manager).

## Technology Selection Rationale

### FastAPI
- Native async/await support for high concurrency
- Automatic OpenAPI documentation generation
- Type-validated request/response models

### PostgreSQL + pgvector
- Mature relational database with ACID guarantees
- Native vector operations eliminate separate vector database
- Full-text search (tsvector) enables hybrid retrieval in single database

### LangChain
- Unified interface for embeddings, LLMs, and document loaders
- Standardized prompt templating and message formatting
- Rich ecosystem of integrations and tools
- Simplifies provider switching (Ollama, OpenAI, Gemini)

### Celery + Redis
- Proven distributed task queue
- Job status persistence enables progress tracking
- Worker pool elasticity for load management

## Alternative Approaches Considered

### Vector-only retrieval
**Rejected because:** Semantic search misses queries requiring exact keyword matches (names, IDs, specific terminology)

### Single-stage retrieval (no reranking)
**Rejected because:** Initial retrieval optimizes for recall. Reranking improves precision. Two-stage approach significantly improves answer quality.

### Synchronous ingestion
**Rejected because:** Large document collections take minutes to hours to process. Asynchronous processing with progress tracking provides better user experience.

### Separate vector database
**Rejected because:** pgvector provides sufficient performance while simplifying infrastructure. Separate vector databases add operational complexity without clear benefit at target scale.

## Deployment Architecture

### Development Environment
- Docker Compose orchestrates all services locally
- Volume mounts enable code changes without container rebuilds
- Separate services for database, cache, API, and workers

### Production Considerations
- Managed PostgreSQL (AWS RDS, Google Cloud SQL) for reliability
- Container orchestration (Cloud Run, ECS, or Kubernetes) for scaling
- Secret management services for credential storage
- Monitoring and alerting (Prometheus, Grafana, or cloud-native tools)

## Future Architecture Evolution

### Incremental Updates
**Challenge:** Currently requires full reindexing to capture document changes

**Solution:** Implement Google Drive webhooks (changes.watch API) to detect modifications and trigger selective reindexing

### Multimodal Support
**Challenge:** Tables, charts, and images are ignored during text extraction

**Solution:** Integrate OCR and table extraction to expand content coverage

### Permission-Aware Filtering
**Challenge:** All users see results from all indexed documents

**Solution:** Store file permissions and filter results based on user identity (requires OAuth implementation)

## Conclusion

The architecture balances multiple objectives: answer accuracy through hybrid retrieval and reranking, system reliability through asynchronous processing and error handling, and operational simplicity through consolidated infrastructure. Key decisions prioritize correctness over raw performance, reflecting the system's enterprise use case where answer accuracy is paramount.
