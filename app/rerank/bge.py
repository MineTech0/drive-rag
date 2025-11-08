"""BGE reranking service using sentence-transformers."""
import logging
from typing import List, Dict
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class BGEReranker:
    """Rerank search results using BGE cross-encoder model."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        Initialize BGE reranker.
        
        Args:
            model_name: Hugging Face model name
                - BAAI/bge-reranker-v2-m3 (multilingual, recommended)
                - BAAI/bge-reranker-large
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (lightweight)
        """
        try:
            self.model = CrossEncoder(model_name, max_length=512)
            logger.info(f"Loaded BGE reranker: {model_name}")
        except Exception as e:
            logger.error(f"Error loading BGE reranker: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 8
    ) -> List[Dict]:
        """
        Rerank documents using BGE cross-encoder.
        
        Args:
            query: Search query
            documents: List of document dictionaries with 'text' field
            top_k: Number of top results to return
            
        Returns:
            Reranked documents with relevance scores
        """
        if not documents:
            return []
        
        try:
            # Prepare query-document pairs
            pairs = [[query, doc['text']] for doc in documents]
            
            # Get reranking scores
            scores = self.model.predict(pairs)
            
            # Sort documents by score
            scored_docs = []
            for idx, score in enumerate(scores):
                doc = documents[idx].copy()
                doc['rerank_score'] = float(score)
                scored_docs.append(doc)
            
            # Sort by score descending
            scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Return top-k
            reranked = scored_docs[:top_k]
            
            logger.info(f"Reranked {len(documents)} documents to top {len(reranked)} using BGE")
            return reranked
            
        except Exception as e:
            logger.error(f"Error in BGE reranking: {e}")
            # Fallback: return top_k documents without reranking
            return documents[:top_k]
