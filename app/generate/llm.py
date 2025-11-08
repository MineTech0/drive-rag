"""LLM generation service using LangChain."""
import logging
from typing import List, Dict, Optional
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
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
    """Service for LLM generation using LangChain."""
    
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        else:
            self._init_ollama()
    
    def _init_ollama(self):
        """Initialize Ollama provider using LangChain."""
        self.model = settings.ollama_model
        self.llm = Ollama(
            model=self.model,
            base_url=settings.ollama_base_url,
            temperature=0.3,
        )
        logger.info(f"Initialized LangChain Ollama with model: {self.model} at {settings.ollama_base_url}")
    
    def _init_openai(self):
        """Initialize OpenAI-compatible provider using LangChain."""
        self.model = settings.openai_model
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            temperature=0.3,
        )
        logger.info(f"Initialized LangChain ChatOpenAI with model: {self.model} at {settings.openai_api_base}")
    
    def _init_gemini(self):
        """Initialize Google Gemini provider using LangChain."""
        self.model = settings.gemini_model
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=settings.gemini_api_key,
            temperature=0.3,
        )
        logger.info(f"Initialized LangChain ChatGoogleGenerativeAI with model: {self.model}")
    
    def generate(self, prompt: str, max_tokens: int = 1000, system_message: Optional[str] = None) -> str:
        """
        Generate text using the configured LLM provider via LangChain.
        
        Args:
            prompt: The prompt to generate from
            max_tokens: Maximum tokens to generate
            system_message: Optional system message
            
        Returns:
            Generated text
        """
        try:
            if system_message and self.provider in ["openai", "gemini"]:
                # Use ChatPromptTemplate for chat models
                chat_prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(system_message),
                    HumanMessagePromptTemplate.from_template("{prompt}")
                ])
                formatted_prompt = chat_prompt.format_messages(prompt=prompt)
                response = self.llm.invoke(formatted_prompt)
                # Extract content from AIMessage
                return response.content.strip() if hasattr(response, 'content') else str(response).strip()
            else:
                # Simple invoke for Ollama or when no system message
                full_prompt = f"{system_message}\n\n{prompt}" if system_message else prompt
                response = self.llm.invoke(full_prompt)
                return response.strip() if isinstance(response, str) else response.content.strip()
                
        except Exception as e:
            logger.error(f"Error generating with LangChain {self.provider}: {e}")
            return f"Virhe vastauksen generoinnissa: {str(e)}"
    
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
        
        # For OpenAI-style APIs and Gemini, use system message properly
        if self.provider in ["openai", "gemini"]:
            system_message = SYSTEM_PROMPT.format(context=context_text)
            user_prompt = f"KYSYMYS: {query}\n\nVASTAUS:"
            answer_text = self.generate(user_prompt, system_message=system_message)
        else:
            # For Ollama, combine everything into one prompt
            prompt = SYSTEM_PROMPT.format(context=context_text)
            prompt += f"\n\nKYSYMYS: {query}\n\nVASTAUS:"
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
