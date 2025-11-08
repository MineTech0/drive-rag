"""Celery tasks for background processing."""
import logging
from celery import Celery
from app.config import settings
from app.ingest.drive import DriveClient
from app.parse.pdf import parse_pdf
from app.parse.docs import parse_google_doc
from app.chunking.semantic import SemanticChunker
from app.index.pgvector import PgVectorIndexer
import psycopg

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    'drive_rag',
    broker=settings.redis_url,
    backend=settings.redis_url
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)


@celery_app.task(bind=True)
def ingest_folder_task(self, job_id: str, root_folder_id: str, full_reindex: bool = False):
    """
    Background task to ingest documents from Google Drive folder.
    
    Args:
        job_id: Ingest job UUID
        root_folder_id: Google Drive folder ID
        full_reindex: Whether to reindex existing documents
    """
    try:
        # Update job status to running
        with psycopg.connect(settings.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_jobs SET state = 'running' WHERE id = %s",
                    (job_id,)
                )
                conn.commit()
        
        # Initialize services
        drive_client = DriveClient(settings.google_application_credentials)
        chunker = SemanticChunker(
            max_tokens=settings.max_chunk_tokens,
            overlap_tokens=settings.chunk_overlap
        )
        indexer = PgVectorIndexer(settings.db_url)
        
        # List all files
        logger.info(f"Listing files from folder {root_folder_id}")
        files = drive_client.list_files_recursive(root_folder_id)
        
        processed = 0
        indexed = 0
        errors = []
        
        for file_meta in files:
            try:
                # Update progress
                self.update_state(
                    state='PROGRESS',
                    meta={'processed': processed, 'total': len(files)}
                )
                
                # Parse document based on type
                if file_meta['mime_type'] == 'application/pdf':
                    content_bytes = drive_client.download_file(file_meta['file_id'])
                    text = parse_pdf(content_bytes)
                elif file_meta['mime_type'] == 'application/vnd.google-apps.document':
                    text = drive_client.export_document(file_meta['file_id'])
                    text = parse_google_doc(text)
                else:
                    logger.warning(f"Unsupported mime type: {file_meta['mime_type']}")
                    continue
                
                if not text or not text.strip():
                    logger.warning(f"Empty text for file {file_meta['name']}")
                    errors.append({
                        'file_id': file_meta['file_id'],
                        'error': 'Empty text after parsing'
                    })
                    processed += 1
                    continue
                
                # Compute content hash
                content_hash = drive_client.compute_content_hash(text)
                file_meta['content_sha256'] = content_hash
                
                # Chunk text
                chunks = chunker.chunk_text(text, metadata={
                    'file_id': file_meta['file_id'],
                    'file_name': file_meta['name']
                })
                
                # Index document and chunks
                doc_id = indexer.upsert_document(file_meta)
                chunk_count = indexer.index_chunks(doc_id, chunks)
                
                processed += 1
                indexed += chunk_count
                
                logger.info(f"Indexed {file_meta['name']}: {chunk_count} chunks")
                
            except Exception as e:
                logger.error(f"Error processing file {file_meta.get('name', 'unknown')}: {e}")
                errors.append({
                    'file_id': file_meta.get('file_id', 'unknown'),
                    'error': str(e)
                })
                processed += 1
                continue
        
        # Update job status to completed
        with psycopg.connect(settings.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE ingest_jobs 
                    SET state = 'completed', processed = %s, indexed = %s, errors = %s
                    WHERE id = %s
                """, (processed, indexed, errors, job_id))
                conn.commit()
        
        return {
            'job_id': job_id,
            'state': 'completed',
            'processed': processed,
            'indexed': indexed,
            'errors': errors
        }
        
    except Exception as e:
        logger.error(f"Fatal error in ingest task: {e}")
        
        # Update job status to failed
        with psycopg.connect(settings.db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE ingest_jobs 
                    SET state = 'failed', errors = %s
                    WHERE id = %s
                """, ([{'error': str(e)}], job_id))
                conn.commit()
        
        raise
