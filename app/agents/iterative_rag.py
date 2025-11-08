"""Iterative Agentic RAG that searches until satisfied."""
import logging
from typing import List, Dict, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchIteration:
    """Represents one search iteration."""
    iteration: int
    query: str
    num_results: int
    assessment: str
    confidence: float
    missing_info: List[str]


class IterativeRAGAgent:
    """
    Agentic RAG that iteratively searches and refines until satisfied.
    
    The agent:
    1. Performs initial search
    2. Assesses if information is complete
    3. Identifies gaps and generates new queries
    4. Continues until confident or max iterations reached
    """
    
    def __init__(
        self,
        retriever,
        reranker,
        llm_service,
        max_iterations: int = 5,
        confidence_threshold: float = 0.85,
        max_sources: int = 100
    ):
        """
        Initialize the iterative RAG agent.
        
        Args:
            retriever: Hybrid retriever instance
            reranker: BGE reranker instance
            llm_service: LLM service for generation
            max_iterations: Maximum search iterations
            confidence_threshold: Stop when confidence >= this
            max_sources: Maximum total sources to accumulate
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm_service = llm_service
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.max_sources = max_sources
    
    def search_iteratively(
        self,
        original_query: str,
        initial_candidates: int = 100
    ) -> Dict:
        """
        Perform iterative search until satisfied or max iterations.
        
        Args:
            original_query: User's original question
            initial_candidates: Number of candidates per search
            
        Returns:
            Dict with final answer, sources, and iteration history
        """
        logger.info(f"Starting iterative RAG for: {original_query}")
        
        # Track state
        all_sources: Dict[str, Dict] = {}  # chunk_id -> source info
        iterations: List[SearchIteration] = []
        current_query = original_query
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n=== Iteration {iteration}/{self.max_iterations} ===")
            logger.info(f"Query: {current_query}")
            
            # Search with current query
            candidates = self.retriever.search(current_query, initial_candidates)
            
            # Deduplicate and add new sources
            new_sources_count = 0
            for cand in candidates:
                chunk_id = cand['chunk_id']
                if chunk_id not in all_sources:
                    all_sources[chunk_id] = cand
                    new_sources_count += 1
            
            logger.info(f"Found {new_sources_count} new sources (total: {len(all_sources)})")
            
            # Rerank all accumulated sources
            all_sources_list = list(all_sources.values())
            reranked = self.reranker.rerank(
                original_query,
                all_sources_list,
                min(self.max_sources, len(all_sources_list))
            )
            
            # Assess if we have enough information
            assessment_result = self._assess_completeness(
                original_query,
                reranked,
                iteration
            )
            
            iterations.append(SearchIteration(
                iteration=iteration,
                query=current_query,
                num_results=len(reranked),
                assessment=assessment_result['assessment'],
                confidence=assessment_result['confidence'],
                missing_info=assessment_result['missing_info']
            ))
            
            logger.info(f"Confidence: {assessment_result['confidence']:.2f}")
            logger.info(f"Assessment: {assessment_result['assessment']}")
            
            # Check if satisfied
            if assessment_result['confidence'] >= self.confidence_threshold:
                logger.info(f"✓ Satisfied with confidence {assessment_result['confidence']:.2f}")
                break
            
            # Check if we've hit source limit
            if len(all_sources) >= self.max_sources:
                logger.info(f"✓ Reached maximum sources ({self.max_sources})")
                break
            
            # Check if this is the last iteration
            if iteration >= self.max_iterations:
                logger.info(f"✓ Reached maximum iterations ({self.max_iterations})")
                break
            
            # Generate new query to fill gaps
            if assessment_result['missing_info']:
                current_query = self._generate_followup_query(
                    original_query,
                    assessment_result['missing_info']
                )
                logger.info(f"Generated follow-up query: {current_query}")
            else:
                # No specific gaps identified, try broader search
                current_query = self._broaden_query(original_query, iteration)
                logger.info(f"Broadening search: {current_query}")
        
        # Generate final comprehensive answer
        logger.info(f"\n=== Generating final answer with {len(reranked)} sources ===")
        
        final_answer = self._generate_comprehensive_answer(
            original_query,
            reranked,
            iterations
        )
        
        return {
            'answer': final_answer,
            'sources': self._format_sources(reranked),
            'iterations': [
                {
                    'iteration': it.iteration,
                    'query': it.query,
                    'num_results': it.num_results,
                    'assessment': it.assessment,
                    'confidence': it.confidence,
                    'missing_info': it.missing_info
                }
                for it in iterations
            ],
            'total_sources': len(reranked),
            'total_iterations': len(iterations),
            'final_confidence': iterations[-1].confidence if iterations else 0.0
        }
    
    def _assess_completeness(
        self,
        query: str,
        sources: List[Dict],
        iteration: int
    ) -> Dict:
        """
        Assess if current sources provide complete answer.
        
        Returns:
            Dict with assessment, confidence (0-1), and missing_info list
        """
        if not sources:
            return {
                'assessment': 'Ei lähteitä löytynyt',
                'confidence': 0.0,
                'missing_info': ['Tarvitaan relevantteja lähteitä']
            }
        
        # Build context from sources
        context_parts = []
        for i, src in enumerate(sources[:15], 1):  # Use top 15 for assessment
            context_parts.append(f"[Lähde {i}] {src.get('file_name', 'Unknown')}: {src['text'][:200]}")
        
        context = "\n\n".join(context_parts)
        
        # Ask LLM to assess completeness
        assessment_prompt = f"""Arvioi voivatko nämä lähteet vastata kysymykseen kattavasti.

Alkuperäinen kysymys: {query}

Löydetyt lähteet ({len(sources)} kpl):
{context}

Analysoi:
1. Voidaanko kysymykseen vastata näiden lähteiden perusteella?
2. Mitä tietoa mahdollisesti puuttuu?
3. Kuinka varma olet että vastaus on kattava? (0-100%)

Vastaa JSON-muodossa:
{{
    "can_answer": true/false,
    "confidence": 0-100,
    "missing_info": ["puuttuva asia 1", "puuttuva asia 2"],
    "reasoning": "lyhyt perustelu"
}}

Palauta VAIN JSON:"""
        
        try:
            response = self.llm_service.generate(assessment_prompt)
            
            # Parse JSON
            import json
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                result = json.loads(json_str)
                
                confidence = result.get('confidence', 50) / 100.0
                missing_info = result.get('missing_info', [])
                reasoning = result.get('reasoning', 'Ei perustelua')
                
                return {
                    'assessment': reasoning,
                    'confidence': confidence,
                    'missing_info': missing_info if isinstance(missing_info, list) else []
                }
        except Exception as e:
            logger.warning(f"Failed to parse assessment JSON: {e}")
        
        # Fallback: simple heuristic
        confidence = min(0.5 + (len(sources) / 40), 0.9)
        
        return {
            'assessment': f'Löydettiin {len(sources)} lähdettä',
            'confidence': confidence,
            'missing_info': []
        }
    
    def _generate_followup_query(
        self,
        original_query: str,
        missing_info: List[str]
    ) -> str:
        """
        Generate a follow-up query to find missing information.
        
        Args:
            original_query: Original user query
            missing_info: List of missing information aspects
            
        Returns:
            New query string
        """
        if not missing_info:
            return original_query
        
        missing_str = ", ".join(missing_info[:3])  # Use top 3
        
        prompt = f"""Luo uusi hakukysely löytääksesi puuttuvan tiedon.

Alkuperäinen kysymys: {original_query}

Puuttuva tieto: {missing_str}

Luo tarkka hakukysely joka löytäisi tämän puuttuvan tiedon.
Palauta VAIN hakukysely, ei muuta tekstiä:"""
        
        try:
            new_query = self.llm_service.generate(prompt, max_tokens=100)
            return new_query.strip().strip('"').strip("'")
        except:
            # Fallback
            return f"{original_query} {missing_info[0]}"
    
    def _broaden_query(self, original_query: str, iteration: int) -> str:
        """
        Broaden the query for wider search.
        
        Args:
            original_query: Original query
            iteration: Current iteration number
            
        Returns:
            Broadened query
        """
        # Strategy: add broadening keywords
        broadening_patterns = [
            f"kaikki tiedot {original_query}",
            f"kattava selvitys {original_query}",
            f"laaja haku {original_query}",
            f"eri näkökulmat {original_query}"
        ]
        
        idx = (iteration - 1) % len(broadening_patterns)
        return broadening_patterns[idx]
    
    def _generate_comprehensive_answer(
        self,
        query: str,
        sources: List[Dict],
        iterations: List[SearchIteration]
    ) -> str:
        """
        Generate comprehensive final answer using all sources.
        
        Args:
            query: Original query
            sources: All reranked sources
            iterations: History of search iterations
            
        Returns:
            Final answer string
        """
        # Build rich context
        context_parts = []
        for i, src in enumerate(sources, 1):
            file_name = src.get('file_name', 'Unknown')
            page = src.get('page_or_heading', 'N/A')
            text = src['text']
            context_parts.append(f"[Lähde {i}: {file_name}, {page}]\n{text}\n")
        
        context = "\n".join(context_parts)
        
        # Create system prompt emphasizing comprehensiveness
        system_prompt = f"""Olet asiantuntija-analyytikko joka laatii kattavia raportteja.

Olet suorittanut {len(iterations)} hakukierrosta ja löytänyt {len(sources)} relevanttia lähdettä.

TÄRKEÄT OHJEET:
1. Vastaa KATTAVASTI käyttäen KAIKKIA relevantteja lähteitä
2. Älä jätä pois yhtään tärkeää tietoa - käy läpi kaikki lähteet
3. Mainitse KAIKKI eri ajankohdat, henkilöt, tapahtumat jne.
4. Ryhmittele tieto loogisesti (esim. aikajärjestyksessä tai teemoittain)
5. Lisää lähdeviitteet SUORAAN tekstiin: "...tiedon mukaan (Tiedosto.pdf, sivu 5)..."
6. Jos jokin tieto puuttuu, mainitse se selkeästi
7. Jos tietoa on paljon, käytä väliotsikoita ja luetteloita

LÄHTEET ({len(sources)} kpl):
{context}

Anna ERITTÄIN KATTAVA ja YKSITYISKOHTAINEN vastaus."""

        user_prompt = f"""KYSYMYS: {query}

Anna kattava, hyvin jäsennelty vastaus joka sisältää KAIKEN oleellisen tiedon lähteistä.

VASTAUS:"""
        
        try:
            answer = self.llm_service.generate(
                user_prompt,
                max_tokens=2000,
                system_message=system_prompt
            )
            return answer.strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Virhe vastausta luodessa: {str(e)}"
    
    def _format_sources(self, sources: List[Dict]) -> List[Dict]:
        """Format sources for API response."""
        formatted = []
        for src in sources:
            snippet = src['text'][:200]
            if len(src['text']) > 200:
                snippet += "..."
            
            formatted.append({
                'file_name': src.get('file_name', 'Unknown'),
                'link': src.get('drive_link', ''),
                'locator': src.get('page_or_heading', 'N/A'),
                'chunk_id': src['chunk_id'],
                'snippet': snippet
            })
        
        return formatted
