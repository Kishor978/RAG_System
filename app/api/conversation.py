from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from app.database.connection import get_db
from app.services.document_processor import document_processor
from app.services.vector_db_manager import qdrant_manager
from app.services.rag_manager import RAGManager
from app.services.memory_manager import RedisMemoryManager
from app.services.booking_manager import BookingManager
from app.services.email_service import EmailService, MockEmailService
from app.services.llm_service import LLMService
from app.core.config import settings

router = APIRouter()

# Initialize services
memory_manager = RedisMemoryManager(
    redis_host=settings.REDIS_HOST,
    redis_port=settings.REDIS_PORT,
    redis_db=settings.REDIS_DB,
    ttl=settings.REDIS_TTL
)

# Use MockEmailService for development/testing
email_service = MockEmailService()
# In production, you'd use the real EmailService:
# email_service = EmailService(
#     smtp_server=settings.SMTP_SERVER,
#     smtp_port=settings.SMTP_PORT,
#     smtp_username=settings.SMTP_USERNAME,
#     smtp_password=settings.SMTP_PASSWORD,
#     sender_email=settings.SENDER_EMAIL
# )

# Initialize LLM service
llm_service = LLMService(
    provider=settings.LLM_PROVIDER,
    api_key=settings.LLM_API_KEY
)

# Initialize RAG manager
rag_manager = RAGManager(
    document_processor=document_processor,
    vector_db_manager=qdrant_manager,
    memory_manager=memory_manager,
    llm_service=llm_service
)

# Request/response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query or message")
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")

class BookingDetails(BaseModel):
    name: str = Field(..., description="Full name of the person booking")
    email: str = Field(..., description="Email address for confirmation")
    date: str = Field(..., description="Requested date (YYYY/MM/DD)")
    time: str = Field(..., description="Requested time (HH:MM AM/PM)")

class QueryResponse(BaseModel):
    response: str = Field(..., description="Response to user query")
    conversation_id: str = Field(..., description="Conversation ID (new or existing)")
    is_booking_related: bool = Field(False, description="Whether the query was related to booking")
    booking_info: Optional[Dict[str, Any]] = Field(None, description="Extracted booking information")
    booking_complete: Optional[bool] = Field(None, description="Whether booking information is complete")

@router.post("/chat", response_model=QueryResponse)
async def chat(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Process a conversational query using the RAG system.
    Handles multi-turn conversations and booking requests.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty."
        )
    
    try:
        # Process the query using our RAG manager
        result = rag_manager.process_query(
            query=request.query,
            conversation_id=request.conversation_id
        )
        
        # Check if this is a booking-related query
        is_booking_related = "booking_info" in result
        booking_complete = result.get("booking_complete", False)
        
        # If booking is complete, process it
        if is_booking_related and booking_complete:
            booking_manager = BookingManager(db, email_service)
            booking_result = booking_manager.process_booking(result["booking_info"])
            
            if booking_result["success"]:
                # Update the response to include booking confirmation
                result["response"] += f"\n\nYour booking has been confirmed and a confirmation email has been sent to {result['booking_info']['email']}."
        
        return {
            "response": result["response"],
            "conversation_id": result["conversation_id"],
            "is_booking_related": is_booking_related,
            "booking_info": result.get("booking_info"),
            "booking_complete": booking_complete if is_booking_related else None
        }
        
    except Exception as e:
        # Log the error
        print(f"An error occurred during chat processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message."
        )

@router.post("/booking", status_code=status.HTTP_201_CREATED)
async def create_booking(
    booking: BookingDetails,
    conversation_id: Optional[str] = Query(None, description="Optional conversation ID to associate with this booking"),
    db: Session = Depends(get_db)
):
    """
    Create a direct booking without going through the conversational interface.
    """
    try:
        # Initialize booking manager
        booking_manager = BookingManager(db, email_service)
        
        # Process the booking
        booking_info = {
            "name": booking.name,
            "email": booking.email,
            "date": booking.date,
            "time": booking.time
        }
        
        # If conversation_id is provided, update its metadata
        if conversation_id:
            memory_manager.update_metadata(conversation_id, {"booking_info": booking_info})
        
        # Process the booking
        result = booking_manager.process_booking(booking_info)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
            
        return {
            "success": True,
            "booking_id": result["booking_id"],
            "message": "Booking created successfully. A confirmation email has been sent.",
            "email_sent": result["email_sent"]
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error
        print(f"An error occurred during booking creation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your booking."
        )

@router.get("/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """
    Retrieve the history of a conversation by ID.
    """
    conversation = memory_manager.get_conversation(conversation_id)
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found."
        )
        
    return {
        "conversation_id": conversation_id,
        "messages": conversation.messages,
        "metadata": conversation.metadata
    }
