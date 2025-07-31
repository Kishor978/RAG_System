from fastapi import FastAPI
from app.api import ingestion,  conversation, evaluation
from app.core.config import settings, BASE_DIR
from app.database.connection import create_db_and_tables
from app.services.vector_db_manager import qdrant_manager

import os
import redis

# Create the 'data' directory if it doesn't exist
os.makedirs(BASE_DIR / "data", exist_ok=True)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Backend for Document Ingestion and Conversational RAG."
)

@app.on_event("startup")
async def startup_event():
    print("Starting up application...")
    # Ensure database tables are created on startup
    create_db_and_tables()
    print("Database tables ensured.")
    
    # Initialize Qdrant client (already done in vector_db_manager.py,
    # but this ensures the collection is created if not exists on startup)
    qdrant_manager._ensure_collection_exists() # Call to ensure collection on startup
    print(f"Collection '{qdrant_manager.collection_name}' ensured in Qdrant.")
    
    # Test Redis connection
    try:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB
        )
        redis_client.ping()
        print("Redis connection successful")
    except Exception as e:
        print(f"Warning: Redis connection failed - {e}")
        print("Chat memory functionality may not work properly")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the RAG System Backend!",
        "documentation": "/docs",
        "version": settings.APP_VERSION
    }

app.include_router(ingestion.router, prefix="/documents", tags=["Document Ingestion"])
app.include_router(conversation.router, prefix="/conversation", tags=["Conversational RAG"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["RAG Evaluation"])
