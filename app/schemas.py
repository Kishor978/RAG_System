from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[float] = None

class ConversationSchema(BaseModel):
    conversation_id: str
    messages: List[Message]
    metadata: Dict[str, Any] = {}

class QueryResultSchema(BaseModel):
    document_id: str
    chunk_id: str
    chunk_index: int
    content: str
    score: float
    filename: Optional[str] = None

class BookingSchema(BaseModel):
    booking_id: str
    name: str
    email: str
    date: str
    time: str
    created_at: datetime

class EvaluationMetric(BaseModel):
    """Schema for storing evaluation metrics."""
    chunking_method: str = Field(..., description="The chunking method used (fixed_size or recursive_character)")
    similarity_algorithm: str = Field(..., description="The similarity algorithm used (e.g., cosine, dot_product)")
    accuracy: float = Field(..., description="Accuracy of the RAG system")
    precision: float = Field(..., description="Precision of retrieved documents")
    recall: float = Field(..., description="Recall of relevant documents")
    f1_score: float = Field(..., description="F1 score (harmonic mean of precision and recall)")
    latency: float = Field(..., description="Query latency in milliseconds")
    
class EvaluationReport(BaseModel):
    """Schema for a complete evaluation report."""
    metrics: List[EvaluationMetric]
    best_combination: Dict[str, str] = Field(
        ..., description="Best chunking method and similarity algorithm combination"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None
