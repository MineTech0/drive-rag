"""Test script to verify LLM provider configuration."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.generate.llm import LLMService
from app.config import settings


def test_llm_generation():
    """Test basic LLM generation."""
    print(f"\n{'='*60}")
    print(f"Testing LLM Provider: {settings.llm_provider.upper()}")
    print(f"{'='*60}\n")
    
    if settings.llm_provider == "ollama":
        print(f"Ollama URL: {settings.ollama_base_url}")
        print(f"Model: {settings.ollama_model}")
    else:
        print(f"API Base: {settings.openai_api_base}")
        print(f"Model: {settings.openai_model}")
    
    print("\nInitializing LLM service...")
    try:
        llm = LLMService()
        print("✓ LLM service initialized successfully\n")
    except Exception as e:
        print(f"✗ Failed to initialize LLM service: {e}")
        return False
    
    # Test simple generation
    print("Testing simple text generation...")
    prompt = "Vastaa yhdellä lauseella: Mikä on Suomen pääkaupunki?"
    
    try:
        response = llm.generate(prompt, max_tokens=100)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}\n")
        
        if response and len(response) > 0:
            print("✓ Basic generation works!")
        else:
            print("✗ Got empty response")
            return False
            
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False
    
    # Test with context
    print("\n" + "="*60)
    print("Testing RAG-style generation with context...")
    print("="*60 + "\n")
    
    test_context = [
        {
            "text": "Projektimme aikataulu on seuraava: Vaihe 1 alkaa tammikuussa ja kestää 3 kuukautta. Vaihe 2 alkaa huhtikuussa.",
            "file_name": "projektisuunnitelma.pdf",
            "page_or_heading": "Aikataulu",
            "drive_link": "https://drive.google.com/file/d/example",
            "chunk_id": "1"
        }
    ]
    
    test_query = "Milloin projektin vaihe 1 alkaa?"
    
    try:
        result = llm.generate_answer(test_query, test_context)
        
        print(f"Query: {test_query}\n")
        print(f"Answer: {result['answer']}\n")
        print(f"Sources ({len(result['sources'])} found):")
        for src in result['sources']:
            print(f"  - {src['file_name']} ({src['locator']})")
        
        print("\n✓ RAG-style generation works!")
        
    except Exception as e:
        print(f"✗ RAG generation failed: {e}")
        return False
    
    # Test multi-query generation if enabled
    if settings.enable_multi_query:
        print("\n" + "="*60)
        print("Testing multi-query generation...")
        print("="*60 + "\n")
        
        try:
            queries = llm.generate_multi_queries("Mikä on projektin budjetti?")
            print("Generated query variations:")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")
            print("\n✓ Multi-query generation works!")
        except Exception as e:
            print(f"✗ Multi-query generation failed: {e}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    success = test_llm_generation()
    sys.exit(0 if success else 1)
