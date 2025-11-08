"""PDF parsing using LangChain PDF loaders."""
import io
import logging
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def parse_pdf(content: bytes, filename: str = "temp.pdf") -> str:
    """
    Extract text from PDF content using LangChain loaders.
    
    Args:
        content: PDF file content as bytes
        filename: Optional filename for better error messages
        
    Returns:
        Extracted text as string
    """
    try:
        # Save content to temporary file-like object
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Try PyPDFLoader first (uses pypdf)
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            if not documents or not any(doc.page_content.strip() for doc in documents):
                # If PyPDF returns empty, try PDFMiner
                logger.info("PyPDF returned empty text, trying PDFMiner fallback")
                loader = PDFMinerLoader(tmp_path)
                documents = loader.load()
            
            # Combine all pages
            full_text = "\n\n".join(doc.page_content for doc in documents if doc.page_content.strip())
            return full_text
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"Error parsing PDF with LangChain: {e}")
        raise


def extract_pdf_metadata(content: bytes) -> dict:
    """
    Extract metadata from PDF using LangChain.
    
    Args:
        content: PDF file content as bytes
        
    Returns:
        Dictionary with metadata
    """
    try:
        import tempfile
        import os
        from pypdf import PdfReader
        
        # Still use pypdf directly for metadata extraction
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        
        metadata = {
            'num_pages': len(reader.pages),
            'title': '',
            'author': '',
            'subject': '',
            'creator': ''
        }
        
        if reader.metadata:
            metadata['title'] = reader.metadata.get('/Title', '')
            metadata['author'] = reader.metadata.get('/Author', '')
            metadata['subject'] = reader.metadata.get('/Subject', '')
            metadata['creator'] = reader.metadata.get('/Creator', '')
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting PDF metadata: {e}")
        return {'num_pages': 0}
