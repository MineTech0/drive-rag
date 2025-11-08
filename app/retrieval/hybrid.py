"""Hybrid retrieval combining vector search and BM25."""
import logging
from typing import List, Dict, Optional
import psycopg
from app.config import settings
from app.index.pgvector import EmbeddingService

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines vector similarity search with BM25 full-text search."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.embedding_service = EmbeddingService()
    
    def search(
        self,
        query: str,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Hybrid search combining vector and BM25 results.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of search results with RRF scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_texts([query])[0]
        
        # Perform vector search
        vector_results = self._vector_search(query_embedding, top_k * 2)
        
        # Perform BM25 search
        bm25_results = self._bm25_search(query, top_k * 2)
        
        # Combine with Reciprocal Rank Fusion
        combined_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            top_k
        )
        
        # Fetch full chunk details
        results_with_details = self._fetch_chunk_details(combined_results)
        
        return results_with_details
    
    def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Dict]:
        """
        Perform vector similarity search using pgvector.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            
        Returns:
            List of results with chunk_id and similarity score
        """
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Cosine similarity search with pgvector
                    cur.execute("""
                        SELECT 
                            e.chunk_id,
                            1 - (e.embedding <=> %s::vector) AS similarity
                        FROM embeddings e
                        ORDER BY e.embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, top_k))
                    
                    results = []
                    for row in cur.fetchall():
                        results.append({
                            'chunk_id': str(row[0]),
                            'vector_score': float(row[1])
                        })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict]:
        """
        Perform BM25 full-text search using PostgreSQL tsvector.
        
        Args:
            query: Search query string
            top_k: Number of results
            
        Returns:
            List of results with chunk_id and BM25 score
        """
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Full-text search with ranking
                    # Use plainto_tsquery with 'simple' dictionary to be language-agnostic
                    # (avoids relying on english stemmer which can miss non-English texts)
                    cur.execute("""
                        SELECT 
                            fts.chunk_id,
                            ts_rank(fts.tsv, query) AS bm25_score
                        FROM documents_fts fts,
                             plainto_tsquery('simple', %s) query
                        WHERE fts.tsv @@ query
                        ORDER BY bm25_score DESC
                        LIMIT %s
                    """, (query, top_k))
                    
                    results = []
                    for row in cur.fetchall():
                        results.append({
                            'chunk_id': str(row[0]),
                            'bm25_score': float(row[1])
                        })
                    
                    return results
                    
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []
    
    def _prepare_query_for_tsquery(self, query: str) -> str:
        """Prepare query string for PostgreSQL tsquery."""
        # Simple preparation: split words and join with &
        words = query.lower().split()
        # Filter out common stop words and join
        filtered_words = [w for w in words if len(w) > 2]
        return ' & '.join(filtered_words) if filtered_words else query
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        top_k: int,
        k: int = 60
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            top_k: Number of top results to return
            k: RRF constant (default: 60)
            
        Returns:
            Fused and sorted results
        """
        rrf_scores = {}
        
        # Add vector search scores
        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)
        
        # Add BM25 scores
        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result['chunk_id']
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank)
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            {'chunk_id': chunk_id, 'rrf_score': score}
            for chunk_id, score in sorted_results
        ]
    
    def _fetch_chunk_details(
        self,
        results: List[Dict]
    ) -> List[Dict]:
        """
        Fetch full details for chunks.
        
        Args:
            results: List with chunk_ids
            
        Returns:
            Results with full chunk and document details
        """
        if not results:
            return []
        
        try:
            chunk_ids = [r['chunk_id'] for r in results]
            
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Fetch chunks with document metadata
                    placeholders = ','.join(['%s'] * len(chunk_ids))
                    cur.execute(f"""
                        SELECT 
                            c.id, c.text, c.chunk_index, c.page_or_heading,
                            d.id, d.name, d.path, d.drive_link, d.mime_type
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE c.id::text IN ({placeholders})
                    """, chunk_ids)
                    
                    chunk_details = {}
                    for row in cur.fetchall():
                        chunk_details[str(row[0])] = {
                            'chunk_id': str(row[0]),
                            'text': row[1],
                            'chunk_index': row[2],
                            'page_or_heading': row[3],
                            'document_id': str(row[4]),
                            'file_name': row[5],
                            'file_path': row[6],
                            'drive_link': row[7],
                            'mime_type': row[8]
                        }
            
            # Merge scores with details
            detailed_results = []
            for result in results:
                chunk_id = result['chunk_id']
                if chunk_id in chunk_details:
                    detailed_result = chunk_details[chunk_id].copy()
                    detailed_result['score'] = result.get('rrf_score', 0)
                    detailed_results.append(detailed_result)
            
            return detailed_results
            
        except Exception as e:
            logger.error(f"Error fetching chunk details: {e}")
            return []

    def document_search(self, query: str, max_chunks: int = 1000, top_docs: int = 0) -> List[Dict]:
        """
        Perform a broader document-level search.

        This method retrieves a larger set of matching chunks (up to max_chunks),
        groups results by document, and returns documents sorted by the best
        matching chunk score. If top_docs is 0, all matching documents are returned.

        Args:
            query: Search query
            max_chunks: Maximum number of chunks to retrieve for scoring
            top_docs: Limit number of documents to return (0 = no limit)

        Returns:
            List of documents with aggregated scores and metadata
        """
        # Get many candidate chunks from both vector and bm25
        try:
            query_embedding = self.embedding_service.embed_texts([query])[0]
        except Exception:
            query_embedding = None

        vector_results = []
        if query_embedding is not None:
            vector_results = self._vector_search(query_embedding, max_chunks)

        bm25_results = self._bm25_search(query, max_chunks)

        # Combine results but do not truncate by top_k; just fuse all candidates
        combined = self._reciprocal_rank_fusion(vector_results, bm25_results, top_k=max_chunks)

        # Fetch details for chunks
        chunk_details = self._fetch_chunk_details(combined)

        # Aggregate by document
        docs = {}
        for c in chunk_details:
            doc_id = c.get('document_id')
            if not doc_id:
                continue
            score = c.get('score', 0)
            if doc_id not in docs:
                docs[doc_id] = {
                    'document_id': doc_id,
                    'file_name': c.get('file_name'),
                    'drive_link': c.get('drive_link'),
                    'best_score': score,
                    'matched_chunks': [c]
                }
            else:
                docs[doc_id]['matched_chunks'].append(c)
                if score > docs[doc_id]['best_score']:
                    docs[doc_id]['best_score'] = score

        # Sort documents by best_score desc
        sorted_docs = sorted(docs.values(), key=lambda x: x['best_score'], reverse=True)

        if top_docs and top_docs > 0:
            sorted_docs = sorted_docs[:top_docs]

        return sorted_docs
