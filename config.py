"""
Configuration module for RAG system
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for RAG system"""
    
    # Database Configuration (SQLite)
    DB_FILE = os.getenv('DB_FILE', 'rag_database.db')
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '384'))
    
    # Chunking Configuration
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '500'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))
    
    # Retrieval Configuration
    TOP_K = int(os.getenv('TOP_K', '5'))
    
    # Ollama Configuration
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
    OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '90'))
    
    # RAG Agent Configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    ENABLE_WEB_SEARCH = os.getenv('ENABLE_WEB_SEARCH', 'True').lower() == 'true'
    ENABLE_ANSWER_GRADING = os.getenv('ENABLE_ANSWER_GRADING', 'True').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        # SQLite requires no configuration validation
        return True
