"""LLM generation service with Ollama integration."""
import logging
import requests
from typing import List, Dict
from app.config import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Olet yritysdatan RAG-avustaja. Vastaa vain toimitetun kontekstin pohjalta.

TÄRKEÄT SÄÄNNÖT:
1. Jos vastaus ei ilmene kontekstista, kerro "En löytänyt varmaa vastausta annetusta materiaalista."
2. Lisää AINA lähteet muodossa: [Tiedostonimi] (linkki, sivu/otsikko)
3. ÄLÄ keksi linkkejä tai sisältöä
4. Käytä vain annettuja dokumentteja vastauksessasi
5. Jos jokin asia on epäselvä, mainitse se rehellisesti

KONTEKSTI:
{context}

Vastaa kysymykseen suomeksi ja ammattimaisesti."""


MULTI_QUERY_PROMPT = """Laadi 3–5 toisistaan merkityksellisesti eroavaa hakulauseketta,
jotka auttavat löytämään relevantin sisällön kysymykseen: "{user_query}".

Palauta vain hakulausekkeet, yksi per rivi, ilman numerointia tai muuta tekstiä."""


HYDE_PROMPT = """Laadi tiivis hypoteettinen vastaus kysymykseen "{user_query}".
Käytä sitä dokumenttihakua ohjaavana "pseudo-dokumenttina".

Vastaus:"""


class LLMService:
    """Service for LLM generation using Ollama (local open-source models)."""
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        logger.info(f"Initialized Ollama with model: {self.model} at {self.base_url}")
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": max_tokens
                    }
                },
                timeout=120
            )
            
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            return "Virhe vastauksen generoinnissa. Varmista että Ollama on käynnissä."
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict]
    ) -> Dict:
        """
        Generate answer with source citations.
        
        Args:
            query: User query
            context_chunks: List of relevant chunks with metadata
            
        Returns:
            Dictionary with answer and sources
        """
        # Format context
        context_text = self._format_context(context_chunks)
        
        # Create prompt
        prompt = SYSTEM_PROMPT.format(context=context_text)
        prompt += f"\n\nKYSYMYS: {query}\n\nVASTAUS:"
        
        # Generate response
        answer_text = self.generate(prompt)
        
        # Extract sources
        sources = self._extract_sources(context_chunks)
        
        return {
            "answer": answer_text,
            "sources": sources
        }
    
    def generate_multi_queries(self, query: str) -> List[str]:
        """
        Generate multiple query variations.
        
        Args:
            query: Original user query
            
        Returns:
            List of query variations
        """
        prompt = MULTI_QUERY_PROMPT.format(user_query=query)
        response = self.generate(prompt)
        
        # Parse queries
        queries = [q.strip() for q in response.split('\n') if q.strip()]
        # Add original query
        queries.insert(0, query)
        
        return queries[:5]  # Limit to 5 queries
    
    def generate_hyde(self, query: str) -> str:
        """
        Generate hypothetical document (HyDE).
        
        Args:
            query: User query
            
        Returns:
            Hypothetical answer
        """
        prompt = HYDE_PROMPT.format(user_query=query)
        return self.generate(prompt)
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format context chunks for prompt."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            file_name = chunk.get('file_name', 'Unknown')
            source_info = f"[{file_name}]"
            if chunk.get('page_or_heading'):
                source_info += f" ({chunk['page_or_heading']})"
            
            context_parts.append(
                f"DOKUMENTTI: {file_name} {source_info}\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Extract source information from chunks."""
        sources = []
        
        for chunk in chunks:
            # Create snippet (first 200 chars)
            snippet = chunk['text'][:200]
            if len(chunk['text']) > 200:
                snippet += "..."
            
            source = {
                "file_name": chunk.get('file_name', 'Unknown'),
                "link": chunk.get('drive_link', ''),
                "locator": chunk.get('page_or_heading', 'N/A'),
                "chunk_id": chunk.get('chunk_id', ''),
                "snippet": snippet
            }
            sources.append(source)
        
        return sources
