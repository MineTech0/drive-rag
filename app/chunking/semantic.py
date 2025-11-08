"""Semantic text chunking using LangChain's text splitters."""
import logging
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunks text using LangChain's RecursiveCharacterTextSplitter."""
    
    def __init__(
        self,
        max_tokens: int = 400,
        overlap_tokens: int = 60,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize chunker with LangChain text splitter.
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of overlapping tokens between chunks
            encoding_name: Tiktoken encoding name
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load encoding {encoding_name}: {e}. Using default.")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        # Convert tokens to approximate character count (rough estimate: 1 token ≈ 4 chars)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_tokens * 4,  # Approximate character count
            chunk_overlap=overlap_tokens * 4,
            length_function=lambda text: len(self.encoding.encode(text)),  # Use token count
            separators=["\n\n", "\n", ". ", " ", ""],  # Hierarchical splitting
            keep_separator=True
        )
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Split text into chunks using LangChain splitter.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Use LangChain's text splitter
        text_chunks = self.splitter.split_text(text)
        
        chunks = []
        current_offset = 0
        
        for index, chunk_text in enumerate(text_chunks):
            token_count = len(self.encoding.encode(chunk_text))
            
            chunk = {
                'chunk_index': index,
                'text': chunk_text,
                'start_offset': current_offset,
                'end_offset': current_offset + len(chunk_text),
                'token_count': token_count
            }
            
            if metadata:
                chunk.update(metadata)
            
            chunks.append(chunk)
            current_offset += len(chunk_text)
        
        logger.info(f"Created {len(chunks)} chunks from text using LangChain splitter")
        return chunks
