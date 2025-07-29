from fastapi import FastAPI
from app.api import ingestion
from app.core.config import settings,BASE_DIR
from app.database.connection import create_db_and_tables
from app.services.vector_db_manager import qdrant_manager

import os

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

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG System Backend!"}

app.include_router(ingestion.router, prefix="/documents", tags=["Document Ingestion"])