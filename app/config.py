"""Configuration management using Pydantic settings."""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Google API
    google_application_credentials: str = "/secrets/sa.json"
    root_folder_id: str
    
    # Database
    db_url: str
    
    # Embedding (local sentence-transformers only)
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024
    
    # LLM (Ollama for local open-source models)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"  # or llama3.1, phi3, etc.
    
    # Reranker (BGE only)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    
    # Chunking
    max_chunk_tokens: int = 400
    chunk_overlap: int = 60
    
    # Retrieval
    topk_candidates: int = 50
    topk_context: int = 8
    enable_hyde: bool = False
    enable_multi_query: bool = True
    multi_query_count: int = 3
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
