"""PDF parsing using pypdf with pymupdf fallback."""
import io
import logging
from typing import Optional
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def parse_pdf(content: bytes) -> str:
    """
    Extract text from PDF content.
    
    Args:
        content: PDF file content as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        # Try pypdf first
        pdf_file = io.BytesIO(content)
        reader = PdfReader(pdf_file)
        
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(text)
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num}: {e}")
                continue
        
        full_text = "\n\n".join(text_parts)
        
        # If pypdf returns empty text, try pymupdf
        if not full_text.strip():
            logger.info("pypdf returned empty text, trying pymupdf fallback")
            full_text = parse_pdf_with_pymupdf(content)
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error parsing PDF with pypdf: {e}")
        # Try fallback
        try:
            return parse_pdf_with_pymupdf(content)
        except Exception as fallback_error:
            logger.error(f"Fallback pymupdf also failed: {fallback_error}")
            raise


def parse_pdf_with_pymupdf(content: bytes) -> str:
    """
    Fallback PDF parser using pymupdf (fitz).
    
    Args:
        content: PDF file content as bytes
        
    Returns:
        Extracted text as string
    """
    try:
        import fitz  # PyMuPDF
        
        pdf_file = io.BytesIO(content)
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text and text.strip():
                text_parts.append(text)
        
        doc.close()
        return "\n\n".join(text_parts)
        
    except ImportError:
        logger.error("pymupdf not installed, cannot use fallback parser")
        return ""
    except Exception as e:
        logger.error(f"Error parsing PDF with pymupdf: {e}")
        return ""


def extract_pdf_metadata(content: bytes) -> dict:
    """
    Extract metadata from PDF.
    
    Args:
        content: PDF file content as bytes
        
    Returns:
        Dictionary with metadata
    """
    try:
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
