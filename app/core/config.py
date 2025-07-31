from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).parent.parent.parent

class Settings(BaseSettings):
    # Application settings
    APP_NAME: str = "RAG System Backend"
    APP_VERSION: str = "0.1.0"

    # Vector DB settings (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "document_chunks"

    # Embedding model settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 outputs 384 dimensions

    # SQL DB settings (SQLite for simplicity, can be PostgreSQL)
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/metadata.db" # Using absolute path

    # Chunking strategies default values
    DEFAULT_FIXED_CHUNK_SIZE: int = 1000
    DEFAULT_FIXED_CHUNK_OVERLAP: int = 200
    DEFAULT_RECURSIVE_CHUNK_SIZE: int = 1000
    DEFAULT_RECURSIVE_CHUNK_OVERLAP: int = 200
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TEXT_SPLITTER_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]
    
    # Redis settings for chat memory
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_TTL: int = 3600 * 24 * 7  # 7 days
    
    # Email settings
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = "your-email@gmail.com"
    SMTP_PASSWORD: str = "your-password-or-app-password"
    SENDER_EMAIL: str = "your-email@gmail.com"
    
    # LLM settings
    LLM_PROVIDER: str = "gemini"  # Options: "gemini", "openai", etc.
    LLM_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    LLM_MODEL: str = "gemini-2.5-flash-lite"  # Gemini model (gemini-pro, gemini-pro-vision)
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 1000
    
    # RAG settings
    MAX_CHUNKS_PER_QUERY: int = 5
    MAX_CONVERSATION_HISTORY: int = 10  # Number of messages to keep in context

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()

# Create data directory if it doesn't exist
os.makedirs(BASE_DIR / "data", exist_ok=True) # Use BASE_DIR directly here
