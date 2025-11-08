"""Semantic text chunking with overlap and metadata preservation."""
import logging
from typing import List, Dict, Optional
import tiktoken

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Chunks text semantically with token-based splitting."""
    
    def __init__(
        self,
        max_tokens: int = 400,
        overlap_tokens: int = 60,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize chunker.
        
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
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # First, try to split by paragraphs
        paragraphs = self._split_by_paragraphs(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_offset = 0
        
        for para in paragraphs:
            para_tokens = self.encoding.encode(para)
            para_token_count = len(para_tokens)
            
            # If single paragraph exceeds max_tokens, split it further
            if para_token_count > self.max_tokens:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text,
                        len(chunks),
                        current_offset,
                        current_offset + len(chunk_text),
                        metadata
                    ))
                    current_chunk = []
                    current_tokens = 0
                    current_offset += len(chunk_text)
                
                # Split long paragraph by sentences
                sub_chunks = self._split_long_text(para, para_tokens)
                for sub_chunk in sub_chunks:
                    chunks.append(self._create_chunk(
                        sub_chunk,
                        len(chunks),
                        current_offset,
                        current_offset + len(sub_chunk),
                        metadata
                    ))
                    current_offset += len(sub_chunk)
                continue
            
            # Check if adding this paragraph exceeds limit
            if current_tokens + para_token_count > self.max_tokens and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text,
                    len(chunks),
                    current_offset,
                    current_offset + len(chunk_text),
                    metadata
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_offset += len(chunk_text) - len(overlap_text)
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_tokens = len(self.encoding.encode("\n\n".join(current_chunk)))
            else:
                current_chunk.append(para)
                current_tokens += para_token_count
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text,
                len(chunks),
                current_offset,
                current_offset + len(chunk_text),
                metadata
            ))
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph breaks."""
        # Split by double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If no paragraphs found, try single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        return paragraphs
    
    def _split_long_text(self, text: str, tokens: List[int]) -> List[str]:
        """Split text that exceeds max_tokens by sentences or fixed token windows."""
        chunks = []
        
        # Try to split by sentences
        sentences = self._split_by_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Start new chunk with overlap
                overlap = current_chunk[-1:] if current_chunk else []
                current_chunk = overlap + [sentence]
                current_tokens = len(self.encoding.encode(" ".join(current_chunk)))
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        import re
        # Split by common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, chunks: List[str]) -> str:
        """Get overlap text from end of current chunks."""
        if not chunks:
            return ""
        
        # Take last paragraph or portion for overlap
        overlap_text = chunks[-1]
        overlap_tokens = len(self.encoding.encode(overlap_text))
        
        if overlap_tokens <= self.overlap_tokens:
            return overlap_text
        
        # Truncate to overlap size
        tokens = self.encoding.encode(overlap_text)
        overlap_tokens_list = tokens[-self.overlap_tokens:]
        return self.encoding.decode(overlap_tokens_list)
    
    def _create_chunk(
        self,
        text: str,
        index: int,
        start_offset: int,
        end_offset: int,
        metadata: Optional[Dict]
    ) -> Dict:
        """Create chunk dictionary with metadata."""
        token_count = len(self.encoding.encode(text))
        
        chunk = {
            'chunk_index': index,
            'text': text,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'token_count': token_count
        }
        
        if metadata:
            chunk.update(metadata)
        
        return chunk
