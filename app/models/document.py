from pydantic import BaseModel, Field
from typing import List, Optional
import uuid

class IngestedDocument(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the ingested document.")
    filename: str
    chunking_strategy: str
    num_chunks: int
    storage_path: Optional[str] = None # Path to original file if saved
    timestamp: float = Field(default_factory=lambda: uuid.uuid1().time_low, description="Unix timestamp of ingestion.")

class DocumentChunk(BaseModel):
    document_id: str
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the chunk.")
    chunk_text: str
    chunk_index: int
    embedding: List[float]
    metadata: dict = {} # Additional metadata for the chunk (e.g., page number)