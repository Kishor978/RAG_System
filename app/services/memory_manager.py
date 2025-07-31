import json
from typing import List, Dict, Any, Optional
import redis
from datetime import datetime, timedelta
import uuid
from pydantic import BaseModel

class Message(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = None

class Conversation(BaseModel):
    conversation_id: str
    messages: List[Message]
    metadata: Dict[str, Any] = {}
    
class RedisMemoryManager:
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0, 
                 ttl: int = 3600 * 24 * 7):  # Default TTL: 7 days
        """
        Initialize Redis connection for storing conversation history.
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database to use
            ttl: Time-to-live for conversations in seconds (default: 7 days)
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.ttl = ttl
        self._test_connection()
        
    def _test_connection(self):
        """Test Redis connection and log status."""
        try:
            self.redis_client.ping()
            print("Successfully connected to Redis server")
        except redis.ConnectionError as e:
            print(f"Failed to connect to Redis: {e}")
            # Don't raise error here, system might work without Redis
            # but should log warning
    
    def _get_key(self, conversation_id: str) -> str:
        """Generate Redis key for a conversation."""
        return f"conversation:{conversation_id}"
    
    def create_conversation(self, system_message: Optional[str] = None) -> str:
        """
        Create a new conversation with optional system message.
        
        Returns:
            conversation_id: Unique ID for the new conversation
        """
        conversation_id = str(uuid.uuid4())
        conversation = Conversation(
            conversation_id=conversation_id,
            messages=[]
        )
        
        if system_message:
            conversation.messages.append(
                Message(role="system", content=system_message, timestamp=datetime.now().timestamp())
            )
        
        self._save_conversation(conversation)
        return conversation_id
    
    def add_message(self, conversation_id: str, role: str, content: str) -> bool:
        """
        Add a message to an existing conversation.
        
        Args:
            conversation_id: The ID of the conversation
            role: The role of the message sender ("user", "assistant", "system")
            content: The message content
            
        Returns:
            bool: True if successful, False otherwise
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
            
        conversation.messages.append(
            Message(role=role, content=content, timestamp=datetime.now().timestamp())
        )
        
        self._save_conversation(conversation)
        return True
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Retrieve a conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve
            
        Returns:
            Conversation object if found, None otherwise
        """
        try:
            key = self._get_key(conversation_id)
            conversation_data = self.redis_client.get(key)
            
            if not conversation_data:
                print(f"No conversation found with ID: {conversation_id}")
                return None
                
            try:
                conversation_dict = json.loads(conversation_data)
                return Conversation.model_validate(conversation_dict)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error decoding conversation data for ID {conversation_id}: {e}")
                return None
        except redis.RedisError as e:
            print(f"Redis error when retrieving conversation {conversation_id}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error when retrieving conversation {conversation_id}: {e}")
            return None
    
    def get_messages(self, conversation_id: str, limit: int = None) -> List[Message]:
        """
        Get messages from a conversation, optionally limited to the most recent ones.
        
        Args:
            conversation_id: The conversation ID
            limit: Maximum number of most recent messages to return
            
        Returns:
            List of Message objects
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
            
        messages = conversation.messages
        if limit and limit > 0 and limit < len(messages):
            messages = messages[-limit:]
            
        return messages
    
    def update_metadata(self, conversation_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a conversation.
        
        Args:
            conversation_id: The conversation ID
            metadata: Dictionary of metadata to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
            
        # Update existing metadata with new values
        conversation.metadata.update(metadata)
        self._save_conversation(conversation)
        return True
    
    def _save_conversation(self, conversation: Conversation) -> None:
        """
        Save conversation to Redis.
        
        Args:
            conversation: Conversation object to save
        """
        key = self._get_key(conversation.conversation_id)
        conversation_json = conversation.model_dump_json()
        self.redis_client.setex(key, self.ttl, conversation_json)
        
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            conversation_id: The ID of the conversation to delete
            
        Returns:
            bool: True if deleted, False otherwise
        """
        key = self._get_key(conversation_id)
        return bool(self.redis_client.delete(key))
