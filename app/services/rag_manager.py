from typing import List, Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime
from app.services.vector_db_manager import VectorDBManager
from app.services.document_processor import DocumentProcessor
from app.services.memory_manager import RedisMemoryManager, Message
from app.services.llm_service import LLMService
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGManager:
    """
    Custom Retrieval-Augmented Generation Manager.
    Handles context retrieval, query understanding, and answer generation.
    """
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_db_manager: VectorDBManager,
        memory_manager: RedisMemoryManager,
        llm_service: Optional[LLMService] = None
    ):
        """
        Initialize the RAG Manager.
        
        Args:
            document_processor: Document processor for text embedding
            vector_db_manager: Vector database manager for retrieval
            memory_manager: Memory manager for conversation history
            llm_service: Optional LLM service for response generation
        """
        self.document_processor = document_processor
        self.vector_db_manager = vector_db_manager
        self.memory_manager = memory_manager
        self.llm_service = llm_service or LLMService()
        
    def _prepare_context(self, query: str, conversation_id: Optional[str] = None, limit: int = 5) -> str:
        """
        Prepare context for answering a query by retrieving relevant document chunks.
        
        Args:
            query: The user's query
            conversation_id: Optional conversation ID for context
            limit: Maximum number of chunks to retrieve
            
        Returns:
            String containing formatted context
        """
        # Generate query embedding
        query_embedding = self.document_processor.embedding_model.encode(query, convert_to_list=True)
        
        # Retrieve similar chunks
        chunks = self.vector_db_manager.search_similar_chunks(query_embedding, limit=limit)
        
        # Format context from chunks
        if not chunks:
            return "No relevant information found."
            
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Document {i}] {chunk['chunk_text']}")
            
        return "\n\n".join(context_parts)
        
    def _get_conversation_context(self, conversation_id: str, max_messages: int = 10) -> str:
        """
        Get conversation history formatted as context.
        
        Args:
            conversation_id: ID of the conversation
            max_messages: Maximum number of messages to include
            
        Returns:
            Formatted conversation history
        """
        if not conversation_id:
            return ""
            
        messages = self.memory_manager.get_messages(conversation_id, limit=max_messages)
        if not messages:
            return ""
            
        formatted_messages = []
        for msg in messages:
            prefix = "User: " if msg.role == "user" else "Assistant: " if msg.role == "assistant" else "System: "
            formatted_messages.append(f"{prefix}{msg.content}")
            
        return "\n".join(formatted_messages)
    
    def _is_meta_conversation_query(self, query: str) -> bool:
        """
        Detect if the query is about the conversation itself.
        
        Args:
            query: The user's query
            
        Returns:
            True if it's a meta-conversation query, False otherwise
        """
        meta_keywords = [
            "conversation", "chat", "talking", "discussed", "said",
            "mentioned", "asked", "told", "question", "answer", 
            "previous", "before", "earlier", "first", "second", "third",
            "last time", "summary", "summarize", "history"
        ]
        
        query_lower = query.lower()
        
        # Check for phrases that indicate asking about the conversation
        phrases = [
            "what did i", "what did you", "what was my", "what was your",
            "you said", "i said", "did i say", "did you say",
            "our conversation", "we talked", "we discussed",
            "tell me what", "repeat what", "summarize our"
        ]
        
        # Check if any phrases are in the query
        phrase_match = any(phrase in query_lower for phrase in phrases)
        
        # Check if any meta keywords are in the query
        keyword_matches = sum(1 for keyword in meta_keywords if keyword in query_lower)
        
        return phrase_match or keyword_matches >= 1
    
    def _detect_booking_request(self, query: str) -> bool:
        """
        Detect if the query is related to booking an interview.
        
        Args:
            query: The user's query
            
        Returns:
            True if booking-related, False otherwise
        """
        booking_keywords = [
            "book", "schedule", "appointment", "interview", "meeting",
            "reservation", "slot", "time", "available", "booking"
        ]
        
        query_lower = query.lower()
        # Check if at least two booking keywords are present
        keyword_matches = sum(1 for keyword in booking_keywords if keyword in query_lower)
        
        return keyword_matches >= 2
        
    def _extract_booking_info(self, query: str) -> Dict[str, Any]:
        """
        Extract booking information from the query.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with extracted booking information
        """
        booking_info = {
            "name": None,
            "email": None,
            "date": None,
            "time": None
        }
        
        # Email extraction
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        email_matches = re.findall(email_pattern, query)
        if email_matches:
            booking_info["email"] = email_matches[0]
            
        # Date extraction (common formats)
        date_patterns = [
            r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',  # MM/DD/YYYY or DD/MM/YYYY
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',     # YYYY/MM/DD
            r'([A-Za-z]+\s+\d{1,2},?\s+\d{4})',   # Month DD, YYYY
            r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})'      # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            date_matches = re.findall(pattern, query)
            if date_matches:
                booking_info["date"] = date_matches[0]
                break
                
        # Time extraction
        time_pattern = r'(\d{1,2}:\d{2})\s*(am|pm|AM|PM)?'
        time_matches = re.findall(time_pattern, query)
        if time_matches:
            time_str = time_matches[0][0]
            am_pm = time_matches[0][1].lower() if time_matches[0][1] else ''
            booking_info["time"] = f"{time_str} {am_pm}".strip()
            
        # Name extraction (basic heuristic)
        # Look for phrases like "my name is [Name]" or "I am [Name]"
        name_patterns = [
            r'my\s+name\s+is\s+([A-Za-z]+(\s+[A-Za-z]+)?)',
            r'I\s+am\s+([A-Za-z]+(\s+[A-Za-z]+)?)'
        ]
        
        for pattern in name_patterns:
            name_matches = re.findall(pattern, query, re.IGNORECASE)
            if name_matches:
                booking_info["name"] = name_matches[0][0]
                break
                
        return booking_info
        
    def handle_booking_request(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """
        Handle a booking request query.
        
        Args:
            query: The user's query
            conversation_id: The conversation ID
            
        Returns:
            Dict with response and any extracted booking information
        """
        booking_info = self._extract_booking_info(query)
        missing_fields = [field for field, value in booking_info.items() if value is None]
        
        if not missing_fields:
            # All booking information provided
            # Update conversation metadata
            self.memory_manager.update_metadata(conversation_id, {"booking_info": booking_info})
            
            return {
                "response": f"I've scheduled your interview for {booking_info['date']} at {booking_info['time']}. "
                           f"A confirmation email will be sent to {booking_info['email']}. "
                           f"Thank you, {booking_info['name']}!",
                "booking_info": booking_info,
                "booking_complete": True
            }
        else:
            # Missing information, prompt for it
            missing_fields_text = ", ".join(missing_fields)
            return {
                "response": f"I'd like to help you book an interview. Could you please provide the following details: {missing_fields_text}?",
                "booking_info": booking_info,
                "booking_complete": False,
                "missing_fields": missing_fields
            }
    
    def process_query(
        self, 
        query: str, 
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query within the conversation context.
        
        Args:
            query: The user's query
            conversation_id: Optional conversation ID
            
        Returns:
            Dictionary containing the response and any relevant metadata
        """
        # Create new conversation if none provided
        if not conversation_id:
            conversation_id = self.memory_manager.create_conversation()
            
        # Add user query to conversation history
        self.memory_manager.add_message(conversation_id, "user", query)
        
        # Check if this is a booking request
        if self._detect_booking_request(query):
            booking_result = self.handle_booking_request(query, conversation_id)
            
            # Save assistant response to conversation history
            self.memory_manager.add_message(conversation_id, "assistant", booking_result["response"])
            
            # Include conversation ID in the result
            booking_result["conversation_id"] = conversation_id
            return booking_result
        
        # Regular RAG query process
        # 1. Retrieve conversation history
        conversation_context = self._get_conversation_context(conversation_id)
        
        # Check if this is a meta-conversation question (about the conversation itself)
        is_meta_conversation = self._is_meta_conversation_query(query)
        
        # 2. Retrieve relevant document chunks
        # If we have conversation history, we can enhance the query
        enhanced_query = query
        if conversation_context and not is_meta_conversation:
            # Use the last few messages to enhance retrieval context
            messages = self.memory_manager.get_messages(conversation_id, limit=3)
            user_messages = [m.content for m in messages if m.role == "user"]
            if user_messages:
                enhanced_query = " ".join(user_messages[-2:] + [query])
        
        # For meta-conversation queries, we'll prioritize conversation history over document context
        # For regular queries, we'll use the document context as usual
        document_context = ""
        if is_meta_conversation:
            # For meta-conversation queries, use conversation history as the primary context
            document_context = f"This is a question about the conversation itself. Here's the conversation history:\n{conversation_context}"
        else:
            # Get document context for regular queries
            document_context = self._prepare_context(enhanced_query)
        
        # 3. Generate response based on the document context and conversation history
        # Format conversation history for the LLM if available
        formatted_history = None
        if conversation_context:
            messages = self.memory_manager.get_messages(conversation_id)
            formatted_history = [
                {"role": msg.role, "content": msg.content}
                for msg in messages if msg.role in ["user", "assistant"]
            ]
        
        # Generate response using the LLM service
        response = self.llm_service.generate_response(
            prompt=query,
            context=document_context,
            conversation_history=formatted_history
        )
        
        # 4. Save the assistant's response to conversation history
        self.memory_manager.add_message(conversation_id, "assistant", response)
        
        return {
            "response": response,
            "conversation_id": conversation_id,
            "document_context": document_context
        }
