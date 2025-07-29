from pydantic import BaseModel
from typing import List, Optional

class IngestedDocument(BaseModel):
    document_id: str
    filename: str
    chunking_strategy: str
    num_chunks: int
    storage_path: Optional[str] = None # Path where the raw file might be stored
    timestamp: float # Unix timestamp

    class Config:
        from_attributes = True # Or from_orm = True for Pydantic V1

class DocumentChunk(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    # embedding: List[float] # If you ever wanted to return embeddings directly
    timestamp: str # ISO format string for consistency with datetime.isoformat()