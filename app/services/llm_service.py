from typing import List, Dict, Any, Optional
import google.generativeai as genai
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Large Language Models."""
    
    def __init__(self, provider: str = settings.LLM_PROVIDER, api_key: str = settings.LLM_API_KEY):
        """
        Initialize the LLM service.
        
        Args:
            provider: LLM provider (e.g., "gemini", "openai")
            api_key: API key for the LLM provider
        """
        self.provider = provider
        self.api_key = api_key
        
        # Initialize the appropriate client based on provider
        if self.provider == "gemini":
            self._init_gemini()
        else:
            logger.warning(f"Unsupported LLM provider: {self.provider}. Using mock responses.")
    
    def _init_gemini(self):
        """Initialize the Gemini API client."""
        if not self.api_key:
            logger.warning("No Gemini API key provided. LLM functionality will be limited.")
            return
            
        try:
            genai.configure(api_key=self.api_key)
            # Test API key by listing models
            models = genai.list_models()
            available_models = [model.name for model in models]
            logger.info(f"Successfully connected to Gemini API. Available models: {available_models}")
            self.client_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            self.client_initialized = False
    
    def generate_response(
        self, 
        prompt: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user's query
            context: Retrieved document context to inform the response
            conversation_history: List of previous conversation messages
            temperature: Temperature for response generation (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            
        Returns:
            Generated response string
        """
        if self.provider == "gemini":
            return self._generate_with_gemini(prompt, context, conversation_history, temperature, max_tokens)
        else:
            # Fallback for unsupported or uninitialized providers
            return f"Based on the available information:\n\n{context}"
    
    def _generate_with_gemini(
        self,
        prompt: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a response using Gemini API.
        
        Args:
            prompt: The user's query
            context: Retrieved document context
            conversation_history: List of previous conversation messages
            temperature: Generation temperature
            max_tokens: Maximum response length
            
        Returns:
            Generated response from Gemini
        """
        if not self.api_key or not hasattr(self, 'client_initialized') or not self.client_initialized:
            logger.warning("Gemini API not initialized. Using fallback response.")
            return f"Based on the available information:\n\n{context}"
            
        try:
            # Configure the model
            model_name = settings.LLM_MODEL
            temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
            max_out_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
            
            # Build the system prompt
            system_prompt = f"""
            You are an intelligent assistant that answers questions based on the provided context and conversation history.
            
            For questions about documents or general knowledge, use the information in the DOCUMENT CONTEXT.
            For questions about previous messages or the conversation itself, use the conversation history.
            
            If neither the context nor conversation history contains the needed information, say you don't have enough information.
            Be concise, accurate, and helpful.
            
            DOCUMENT CONTEXT:
            {context}
            """
            
            # Get the Gemini model
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": max_out_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                }
            )
            
            # Handle conversation history if provided
            if conversation_history:
                chat_session = model.start_chat(history=[])
                
                # First, add the system prompt
                chat_session.send_message(system_prompt, stream=False)
                
                # Create a formatted history section for reference
                formatted_history = "\n\nCONVERSATION HISTORY:\n"
                for idx, message in enumerate(conversation_history):
                    role = message.get("role", "")
                    content = message.get("content", "")
                    if content and role in ["user", "assistant"]:
                        # Add numbered messages for easy reference in meta-conversation queries
                        message_num = (idx // 2) + 1 if role == "user" else ""
                        role_display = f"User Question {message_num}" if role == "user" else "Assistant"
                        formatted_history += f"{role_display}: {content}\n"
                
                # Add the conversation history to the system prompt
                history_prompt = f"{system_prompt}\n{formatted_history}"
                chat_session.send_message(history_prompt, stream=False)
                
                # Send the current user query
                # For meta-conversation queries, remind the model to use the conversation history
                if "what was my" in prompt.lower() or "previous" in prompt.lower():
                    enhanced_prompt = f"{prompt} (Please reference the conversation history above to answer this question)"
                    response = chat_session.send_message(enhanced_prompt, stream=False)
                else:
                    response = chat_session.send_message(prompt, stream=False)
            else:
                # Simple one-off generation with context
                full_prompt = f"{system_prompt}\n\nQUESTION: {prompt}\n\nANSWER:"
                response = model.generate_content(full_prompt)
            
            # Extract the text from the response
            response_text = response.text
            
            logger.info(f"Generated response with Gemini: {response_text[:50]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            return f"I encountered an error while generating a response. Here's the relevant information I found:\n\n{context}"
    
    def format_conversation_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Format conversation history for Gemini API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted conversation history
        """
        formatted_messages = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                # Gemini doesn't have a system role, so we add it as a user message
                formatted_messages.append({"role": "user", "content": content})
            elif role in ["user", "assistant"]:
                formatted_messages.append({"role": role, "content": content})
        
        return formatted_messages
