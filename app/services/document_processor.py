import io
from typing import List, Literal, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from app.models.document import DocumentChunk
from app.utils.text_splitters import fixed_size_chunking, recursive_character_chunking
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO) # Set logging level for better visibility
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, embedding_model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Loaded embedding model: {embedding_model_name}")

    def extract_text(self, file_content: bytes, file_type: str) -> str:
        """
        Extracts text from PDF or TXT file content.
        """
        if file_type == "application/pdf":
            reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif file_type == "text/plain":
            return file_content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def chunk_text(self, text: str, strategy: Literal["fixed_size", "recursive_character"], document_id: str) -> List[DocumentChunk]:
        """
        Applies selected chunking strategy and generates DocumentChunk objects.
        """
        chunks_text: List[str] = []
        if strategy == "fixed_size":
            chunks_text = fixed_size_chunking(
                text,
                chunk_size=settings.DEFAULT_FIXED_CHUNK_SIZE,
                overlap=settings.DEFAULT_FIXED_CHUNK_OVERLAP
            )
        elif strategy == "recursive_character":
            chunks_text = recursive_character_chunking(
                text,
                chunk_size=settings.DEFAULT_RECURSIVE_CHUNK_SIZE,
                overlap=settings.DEFAULT_RECURSIVE_CHUNK_OVERLAP
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        document_chunks: List[DocumentChunk] = []
        for i, chunk_txt in enumerate(chunks_text):
            document_chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    chunk_text=chunk_txt,
                    embedding=[], # Placeholder, will be filled next
                    metadata={"page_num": "N/A"} # Placeholder, improve if actual page info is available
                )
            )
        return document_chunks

    def generate_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        Generates embeddings for a list of DocumentChunk objects.
        """
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, convert_to_list=True)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        return chunks

# Initialize document processor globally or via dependency injection
document_processor = DocumentProcessor()