"""Embedding generation and pgvector indexing using LangChain."""
import logging
from typing import List, Dict
import psycopg
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using LangChain's HuggingFaceEmbeddings."""
    
    def __init__(self):
        self.model = settings.embedding_model
        logger.info(f"Loading embedding model: {self.model}")
        
        # Initialize LangChain HuggingFaceEmbeddings
        self.model_instance = HuggingFaceEmbeddings(
            model_name=self.model,
            model_kwargs={'device': 'cpu'},  # Will auto-detect GPU
            encode_kwargs={'normalize_embeddings': False, 'batch_size': 32}
        )
        
        # Get dimension from config (LangChain doesn't expose it directly)
        self.dimension = settings.embedding_dimension
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use LangChain's embed_documents method
            embeddings = self.model_instance.embed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


class PgVectorIndexer:
    """Handles indexing to PostgreSQL with pgvector and BM25."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.embedding_service = EmbeddingService()
    
    def index_chunks(
        self,
        document_id: str,
        chunks: List[Dict]
    ) -> int:
        """
        Index document chunks with embeddings and BM25.
        
        Args:
            document_id: UUID of the parent document
            chunks: List of chunk dictionaries
            
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0
        
        try:
            # Generate embeddings for all chunks
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_service.embed_texts(texts)
            
            # Connect to database
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    indexed_count = 0
                    
                    for chunk, embedding in zip(chunks, embeddings):
                        # Insert chunk
                        cur.execute("""
                            INSERT INTO chunks 
                            (document_id, chunk_index, text, start_offset, end_offset, 
                             page_or_heading, token_count)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, (
                            document_id,
                            chunk['chunk_index'],
                            chunk['text'],
                            chunk.get('start_offset', 0),
                            chunk.get('end_offset', 0),
                            chunk.get('page_or_heading', ''),
                            chunk.get('token_count', 0)
                        ))
                        
                        chunk_id = cur.fetchone()[0]
                        
                        # Insert embedding
                        cur.execute("""
                            INSERT INTO embeddings (chunk_id, embedding, model, dim)
                            VALUES (%s, %s, %s, %s)
                        """, (
                            chunk_id,
                            embedding,
                            self.embedding_service.model,
                            self.embedding_service.dimension
                        ))
                        
                        # Insert BM25 full-text search
                        cur.execute("""
                            INSERT INTO documents_fts (chunk_id, tsv)
                            VALUES (%s, to_tsvector('english', %s))
                        """, (chunk_id, chunk['text']))
                        
                        indexed_count += 1
                    
                    conn.commit()
                    logger.info(f"Indexed {indexed_count} chunks for document {document_id}")
                    return indexed_count
                    
        except Exception as e:
            logger.error(f"Error indexing chunks: {e}")
            raise
    
    def upsert_document(self, file_metadata: Dict) -> str:
        """
        Insert or update document metadata.
        
        Args:
            file_metadata: Dictionary with document metadata
            
        Returns:
            Document UUID
        """
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Check if document exists
                    cur.execute(
                        "SELECT id FROM documents WHERE file_id = %s",
                        (file_metadata['file_id'],)
                    )
                    result = cur.fetchone()
                    
                    if result:
                        # Update existing document
                        doc_id = result[0]
                        cur.execute("""
                            UPDATE documents 
                            SET name = %s, path = %s, mime_type = %s, 
                                revision = %s, modified_time = %s, 
                                drive_link = %s, content_sha256 = %s
                            WHERE id = %s
                        """, (
                            file_metadata['name'],
                            file_metadata.get('path', ''),
                            file_metadata['mime_type'],
                            file_metadata.get('revision', ''),
                            file_metadata.get('modified_time'),
                            file_metadata['drive_link'],
                            file_metadata.get('content_sha256', ''),
                            doc_id
                        ))
                        
                        # Delete old chunks (cascade will handle embeddings and fts)
                        cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
                    else:
                        # Insert new document
                        cur.execute("""
                            INSERT INTO documents 
                            (file_id, name, path, mime_type, revision, modified_time, 
                             drive_link, content_sha256)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                        """, (
                            file_metadata['file_id'],
                            file_metadata['name'],
                            file_metadata.get('path', ''),
                            file_metadata['mime_type'],
                            file_metadata.get('revision', ''),
                            file_metadata.get('modified_time'),
                            file_metadata['drive_link'],
                            file_metadata.get('content_sha256', '')
                        ))
                        doc_id = cur.fetchone()[0]
                    
                    conn.commit()
                    return str(doc_id)
                    
        except Exception as e:
            logger.error(f"Error upserting document: {e}")
            raise
