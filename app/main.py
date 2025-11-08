"""FastAPI application with endpoints."""
import logging
import time
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import psycopg
from app.config import settings
from app.retrieval.hybrid import HybridRetriever
from app.rerank.bge import BGEReranker
from app.generate.llm import LLMService
from app.tasks import ingest_folder_task
from app.agents.iterative_rag import IterativeRAGAgent

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Drive RAG API",
    description="RAG system for Google Drive documents",
    version="1.0.0"
)

# Convert SQLAlchemy URL to psycopg format
db_url = settings.db_url.replace('postgresql+psycopg://', 'postgresql://')

# Lazy-load services (initialize on first use to avoid startup crashes)
_retriever = None
_reranker = None
_llm_service = None
_loading = False


def get_retriever():
    """Get or initialize retriever."""
    global _retriever
    if _retriever is None:
        logger.info("Loading retriever...")
        _retriever = HybridRetriever(db_url)
        logger.info("✓ Retriever loaded")
    return _retriever


def get_reranker():
    """Get or initialize BGE reranker."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading BGE reranker: {settings.reranker_model}")
        _reranker = BGEReranker(model_name=settings.reranker_model)
        logger.info("✓ Reranker loaded")
    return _reranker


def get_llm_service():
    """Get or initialize LLM service."""
    global _llm_service
    if _llm_service is None:
        logger.info("Loading LLM service...")
        _llm_service = LLMService()
        logger.info("✓ LLM service loaded")
    return _llm_service


def calculate_dynamic_top_k(query: str) -> int:
    """
    Calculate optimal top_k based on query complexity.
    
    Factors:
    - Query length (longer = more context needed)
    - Question words (multiple questions = more sources)
    - Comparative/list keywords (needs more examples)
    - Search/find keywords (exhaustive search needed)
    
    Returns:
        int: Recommended top_k value (4-20)
    """
    query_lower = query.lower()
    
    # Base top_k
    top_k = 6
    
    # CRITICAL: Check for exhaustive search keywords first
    exhaustive_keywords = [
        'etsi', 'hae', 'löydä', 'kaikki', 'kaikkia',
        'search', 'find', 'all', 'every', 'each',
        'listaa', 'luettele', 'kerro kaikki',
        'mitkä kaikki', 'mitä kaikkea'
    ]
    if any(keyword in query_lower for keyword in exhaustive_keywords):
        top_k = 20  # Maximum sources for exhaustive search
        logger.info(f"Exhaustive search detected - using top_k={top_k}")
    
    # Factor 1: Query length
    words = query.split()
    if len(words) > 20:
        top_k += 3
    elif len(words) > 10:
        top_k += 2
    elif len(words) < 5:
        top_k = max(top_k - 1, 4)
    
    # Factor 2: Multiple questions
    question_markers = ['?', 'ja', 'sekä', 'myös', 'lisäksi', 'and', 'also']
    question_count = sum(1 for marker in question_markers if marker in query_lower)
    if question_count > 2:
        top_k += 3
    elif question_count > 1:
        top_k += 2
    
    # Factor 3: Comparative/list queries (need more examples)
    comparative_keywords = [
        'vertaa', 'vertaile', 'ero', 'erot', 'eroa',
        'compare', 'difference', 
        'yhteenveto', 'summary', 'kokonaiskuva', 'overview'
    ]
    if any(keyword in query_lower for keyword in comparative_keywords):
        top_k += 4
    
    # Factor 4: Specific detail queries (need less context)
    specific_keywords = [
        'mikä on', 'kuka on', 'milloin', 'missä',
        'what is', 'who is', 'when', 'where',
        'määrittele', 'define'
    ]
    if any(keyword in query_lower for keyword in specific_keywords):
        top_k = max(top_k - 2, 4)
    
    # Clamp between 4 and 20
    return max(4, min(20, top_k))


@app.on_event("startup")
async def startup_event():
    """
    Preload models on startup in background.
    This makes the first request faster.
    """
    global _loading
    if _loading:
        return
    
    _loading = True
    logger.info("🚀 Starting model preload...")
    
    import asyncio
    
    async def preload_models():
        try:
            # Load in order of importance
            logger.info("📥 Loading retriever (embeddings)...")
            get_retriever()
            
            logger.info("📥 Loading reranker...")
            get_reranker()
            
            logger.info("📥 Loading LLM service...")
            get_llm_service()
            
            logger.info("✅ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error preloading models: {e}")
            logger.info("Models will be loaded on first request instead.")
    
    # Run in background
    asyncio.create_task(preload_models())


# Request/Response Models
class IngestRequest(BaseModel):
    root_folder_id: str
    full_reindex: bool = False


class IngestResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    state: str
    processed: int
    indexed: int
    errors: list


class AskRequest(BaseModel):
    query: str
    multi_query: bool = True
    hyde: bool = False
    top_k: int = None  # None = auto-detect based on query complexity


class Source(BaseModel):
    file_name: str
    link: str
    locator: str
    chunk_id: str
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int


class SearchRequest(BaseModel):
    query: str
    k: int = 20
    # If true, perform document-level/broad search (aggregate chunks into documents)
    document_level: bool = False
    # When document_level=True, control how many chunks to consider and how many documents to return
    max_chunks: int = 1000
    top_docs: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str


# Endpoints
@app.post("/ingest/start", response_model=IngestResponse)
async def start_ingest(request: IngestRequest):
    """
    Start background job to ingest documents from Google Drive folder.
    """
    try:
        # Create job record
        job_id = str(uuid.uuid4())
        
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO ingest_jobs (id, root_folder_id, state, full_reindex)
                    VALUES (%s, %s, 'pending', %s)
                """, (job_id, request.root_folder_id, request.full_reindex))
                conn.commit()
        
        # Start background task
        ingest_folder_task.delay(job_id, request.root_folder_id, request.full_reindex)
        
        return IngestResponse(job_id=job_id)
        
    except Exception as e:
        logger.error(f"Error starting ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/status/{job_id}", response_model=JobStatus)
async def get_ingest_status(job_id: str):
    """
    Get status of an ingest job.
    """
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT state, processed, indexed, errors
                    FROM ingest_jobs
                    WHERE id = %s
                """, (job_id,))
                
                result = cur.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="Job not found")
                
                return JobStatus(
                    state=result[0],
                    processed=result[1],
                    indexed=result[2],
                    errors=result[3] or []
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a question and get an answer with source citations.
    """
    start_time = time.time()
    
    try:
        # Get services
        llm_service = get_llm_service()
        retriever = get_retriever()
        reranker = get_reranker()
        
        # Calculate dynamic top_k if not provided
        if request.top_k is None:
            top_k = calculate_dynamic_top_k(request.query)
            logger.info(f"Auto-calculated top_k={top_k} for query: {request.query[:50]}...")
        else:
            top_k = request.top_k
        
        # Detect if exhaustive search is needed
        query_lower = request.query.lower()
        exhaustive_keywords = [
            'etsi', 'hae', 'löydä', 'kaikki', 'kaikkia',
            'search', 'find', 'all', 'every', 'each',
            'listaa', 'luettele', 'kerro kaikki',
            'mitkä kaikki', 'mitä kaikkea'
        ]
        is_exhaustive = any(keyword in query_lower for keyword in exhaustive_keywords)
        
        # Increase candidate retrieval for exhaustive searches
        num_candidates = 100 if is_exhaustive else settings.topk_candidates
        if is_exhaustive:
            logger.info(f"Exhaustive search mode: retrieving {num_candidates} candidates")
        
        # Handle multi-query expansion
        if request.multi_query and settings.enable_multi_query:
            queries = llm_service.generate_multi_queries(request.query)
            logger.info(f"Generated {len(queries)} query variations")
        else:
            queries = [request.query]
        
        # Handle HyDE if enabled
        if request.hyde and settings.enable_hyde:
            hyde_doc = llm_service.generate_hyde(request.query)
            queries.append(hyde_doc)
        
        # Retrieve candidates from all queries
        all_candidates = []
        for query in queries:
            candidates = retriever.search(query, num_candidates)
            all_candidates.extend(candidates)
        
        # Deduplicate by chunk_id
        seen = set()
        unique_candidates = []
        for cand in all_candidates:
            if cand['chunk_id'] not in seen:
                seen.add(cand['chunk_id'])
                unique_candidates.append(cand)
        
        logger.info(f"Retrieved {len(unique_candidates)} unique candidates for reranking")
        
        # Rerank candidates
        if unique_candidates:
            reranked = reranker.rerank(
                request.query,
                unique_candidates,
                top_k
            )
        else:
            reranked = []
        
        if not reranked:
            return AskResponse(
                answer="En löytänyt relevanttia tietoa annetusta kysymyksestä.",
                sources=[],
                latency_ms=int((time.time() - start_time) * 1000)
            )
        
        # Generate answer with sources
        result = llm_service.generate_answer(request.query, reranked)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return AskResponse(
            answer=result['answer'],
            sources=[Source(**s) for s in result['sources']],
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-iterative")
async def ask_iterative(request: AskRequest):
    """
    Iterative Agentic RAG - searches until satisfied with result quality.
    
    The agent:
    1. Performs initial comprehensive search
    2. Assesses if information is complete
    3. Identifies gaps and generates refined queries
    4. Continues iterating until confident or max iterations (default 5)
    5. Returns comprehensive answer with ALL relevant sources
    
    Best for: "etsi kaikki", "hae kattavasti", "kerro kaikki mitä tiedät"
    """
    start_time = time.time()
    
    try:
        # Get services
        llm_service = get_llm_service()
        retriever = get_retriever()
        reranker = get_reranker()
        
        # Initialize iterative agent
        agent = IterativeRAGAgent(
            retriever=retriever,
            reranker=reranker,
            llm_service=llm_service,
            max_iterations=5,  # Can be made configurable
            confidence_threshold=0.85,
            max_sources=100
        )
        
        # Run iterative search
        logger.info(f"Starting iterative RAG for: {request.query}")
        result = agent.search_iteratively(
            original_query=request.query,
            initial_candidates=100  # Start with comprehensive search
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "answer": result['answer'],
            "sources": result['sources'],
            "latency_ms": latency_ms,
            "iterations": result['iterations'],
            "total_sources": result['total_sources'],
            "total_iterations": result['total_iterations'],
            "final_confidence": result['final_confidence']
        }
        
    except Exception as e:
        logger.error(f"Error in iterative ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research")
async def deep_research(request: AskRequest):
    """
    Deep research endpoint - iterative multi-step analysis.
    
    1. Decomposes question into sub-questions
    2. Answers each sub-question independently
    3. Synthesizes comprehensive final answer
    """
    start_time = time.time()
    
    try:
        # Get services
        llm_service = get_llm_service()
        retriever = get_retriever()
        reranker = get_reranker()
        
        query = request.query
        top_k = request.top_k or calculate_dynamic_top_k(query)
        
        logger.info(f"Starting deep research for: {query}")
        
        # Step 1: Decompose into sub-questions
        decompose_prompt = f"""Olet tutkimusassistentti. Analysoi seuraava kysymys ja jaa se 3-5 tarkentavaan alikysymykseen, 
joihin vastaamalla saat kattavan vastauksen alkuperäiseen kysymykseen.

Alkuperäinen kysymys: {query}

Palauta JSON-muodossa:
{{"sub_questions": ["kysymys1", "kysymys2", "kysymys3"]}}

Palauta VAIN JSON, ei muuta tekstiä."""
        
        decomposition_result = llm_service.generate(decompose_prompt)
        
        # Parse sub-questions
        import json
        try:
            # Try to find JSON in response
            start_idx = decomposition_result.find('{')
            end_idx = decomposition_result.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = decomposition_result[start_idx:end_idx]
                sub_questions_data = json.loads(json_str)
                sub_questions = sub_questions_data.get("sub_questions", [])
            else:
                raise ValueError("No JSON found")
        except:
            # Fallback: extract questions from text
            sub_questions = [q.strip().strip('"').strip("'") for q in decomposition_result.split('\n') if q.strip() and ('?' in q or len(q.split()) > 3)]
            sub_questions = [q for q in sub_questions if not q.startswith('{') and not q.startswith('[')]
        
        if not sub_questions:
            sub_questions = [query]  # Fallback to original
        
        # Limit to 5 sub-questions
        sub_questions = sub_questions[:5]
        logger.info(f"Generated {len(sub_questions)} sub-questions: {sub_questions}")
        
        # Step 2: Answer each sub-question
        sub_answers = []
        all_sources = {}  # Deduplicate sources by chunk_id
        
        for i, sub_q in enumerate(sub_questions, 1):
            logger.info(f"Researching sub-question {i}/{len(sub_questions)}: {sub_q}")
            
            # Retrieve documents
            candidates = retriever.search(sub_q, settings.topk_candidates)
            
            if not candidates:
                sub_answers.append({
                    "question": sub_q,
                    "answer": "Ei löytynyt relevanttia tietoa tähän kysymykseen.",
                    "source_ids": []
                })
                continue
            
            # Rerank to get best sources
            reranked = reranker.rerank(sub_q, candidates, min(5, top_k))
            
            if not reranked:
                sub_answers.append({
                    "question": sub_q,
                    "answer": "Ei löytynyt relevanttia tietoa tähän kysymykseen.",
                    "source_ids": []
                })
                continue
            
            # Format context
            context_parts = []
            source_ids = []
            
            for doc in reranked:
                file_name = doc.get('file_name', 'Unknown')
                context_parts.append(f"[{file_name}]: {doc['text']}")
                
                # Store source
                chunk_id = doc['chunk_id']
                if chunk_id not in all_sources:
                    snippet = doc['text'][:150]
                    if len(doc['text']) > 150:
                        snippet += "..."
                    
                    all_sources[chunk_id] = {
                        "file_name": file_name,
                        "link": doc.get('drive_link', ''),
                        "locator": doc.get('page_or_heading', 'N/A'),
                        "chunk_id": chunk_id,
                        "snippet": snippet
                    }
                source_ids.append(chunk_id)
            
            context = "\n\n".join(context_parts)
            
            # Generate answer for sub-question
            answer_prompt = f"""Vastaa seuraavaan kysymykseen käyttäen annettua kontekstia. Ole tarkka ja ytimekäs.

Konteksti:
{context}

Kysymys: {sub_q}

Vastaus (2-4 virkettä):"""
            
            answer = llm_service.generate(answer_prompt)
            
            sub_answers.append({
                "question": sub_q,
                "answer": answer.strip(),
                "source_ids": source_ids
            })
        
        # Step 3: Synthesize final comprehensive answer
        # Build context with source information
        synthesis_parts = []
        for i, sa in enumerate(sub_answers, 1):
            # Get file names for this sub-answer
            source_files = []
            for source_id in sa.get('source_ids', []):
                if source_id in all_sources:
                    source_files.append(all_sources[source_id]['file_name'])
            
            # Remove duplicates while preserving order
            unique_sources = []
            for sf in source_files:
                if sf not in unique_sources:
                    unique_sources.append(sf)
            
            sources_str = ", ".join(unique_sources[:3])  # Show max 3 sources
            synthesis_parts.append(
                f"Kysymys {i}: {sa['question']}\n"
                f"Vastaus: {sa['answer']}\n"
                f"Lähteet: {sources_str}"
            )
        
        synthesis_context = "\n\n".join(synthesis_parts)
        
        synthesis_prompt = f"""Olet tutkija joka syntetisoi tietoa kattavaksi raportiksi.

Alkuperäinen tutkimuskysymys: {query}

Tutkimustulokset alikysymyksistä:
{synthesis_context}

Luo kattava, hyvin jäsennelty synteesivastaus joka:
1. Vastaa suoraan alkuperäiseen tutkimuskysymykseen
2. Integroi kaiken oleellisen tiedon alivastauksistayhtenäiseksi kokonaisuudeksi
3. On selkeä, loogisesti etenevä ja helppolukuinen
4. Käyttää väliotsikoita tarvittaessa
5. Mainitsee lähteet SUORAAN TEKSTISSÄ muodossa: "...tiedon mukaan (Tiedostonimi.pdf)..."
   - ÄLÄ käytä []-merkintöjä vaan pelkkää tiedostonimeä sulkeissa
   - Mainitse lähde sen tekstin yhteydessä mistä tieto on peräisin

Synteesivastaus:"""
        
        final_answer = llm_service.generate(synthesis_prompt)
        
        # Format response
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Convert sources dict to list
        sources_list = list(all_sources.values())
        
        return {
            "query": query,
            "answer": final_answer.strip(),
            "research_steps": sub_answers,
            "sources": sources_list,
            "num_sub_questions": len(sub_questions),
            "latency_ms": latency_ms
        }
        
    except Exception as e:
        logger.error(f"Deep research error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_documents(request: SearchRequest):
    """
    Search documents without generating an answer.
    """
    try:
        retriever = get_retriever()
        # If caller wants a document-level broad search, use the document_search helper
        if getattr(request, 'document_level', False):
            docs = retriever.document_search(
                request.query,
                max_chunks=request.max_chunks,
                top_docs=request.top_docs
            )
            return {"documents": docs}

        results = retriever.search(request.query, request.k)
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reindex/{file_id}")
async def reindex_document(file_id: str):
    """
    Reindex a single document by file_id.
    
    Note: Currently not implemented. Use /ingest/start with full_reindex=true instead.
    """
    raise HTTPException(status_code=501, detail="Not implemented. Use /ingest/start with full_reindex=true instead.")


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check database connection
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        
        return HealthResponse(status="healthy", version="1.0.0")
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/metrics")
async def get_metrics():
    """
    Get application metrics.
    
    Returns basic system statistics. For production use, integrate with Prometheus.
    """
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Get document counts
                cur.execute("SELECT COUNT(*) FROM documents")
                doc_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM chunks")
                chunk_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM embeddings")
                embedding_count = cur.fetchone()[0]
        
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "embeddings": embedding_count,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/")
async def root():
    """
    Root endpoint.
    """
    return {
        "service": "Drive RAG API",
        "version": "1.0.0",
        "endpoints": [
            "/ingest/start",
            "/ingest/status/{job_id}",
            "/ask",
            "/ask-iterative - Agentic RAG with iterative search",
            "/research - Deep research with sub-questions",
            "/search",
            "/healthz"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
