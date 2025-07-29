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
            return self._extract_text_from_pdf(file_content)
        elif file_type == "text/plain":
            return file_content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extracts text content from a PDF file using pypdf."""
        full_text = []
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PdfReader(pdf_file)

            if reader.is_encrypted:
                logger.error("PDF is encrypted and cannot be processed without a password.")
                raise ValueError("Encrypted PDF: Cannot extract text without a password.")

            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(page_text)
                    else:
                        logger.warning(f"Extracted empty or no text from page {page_num + 1} of PDF.")
                except Exception as page_e:
                    logger.error(f"Error extracting text from page {page_num + 1}: {page_e}", exc_info=True)
                    # Decide if you want to skip this page or fail completely
                    # For now, we'll log and continue to next page
                    pass 

            combined_text = "\n".join(full_text)
            
            # Critical check for empty content AFTER extraction
            if not combined_text.strip():
                logger.warning("PDF extraction resulted in an entirely empty string after processing all pages. This might be a scanned PDF without OCR or malformed.")
                raise ValueError("PDF content is empty or unextractable (e.g., scanned without OCR).")
            
            logger.info("Text extracted from PDF successfully.")
            return combined_text

        except Exception as e:
            logger.error(f"Failed to read or process PDF file for text extraction: {e}", exc_info=True)
            raise ValueError(f"Could not process PDF file: {e}. Check PDF integrity or if it's a scanned document.")

    def chunk_text(self, text: str, strategy: Literal["fixed_size", "recursive_character"], document_id: str) -> List[DocumentChunk]:
        """
        Applies selected chunking strategy and generates DocumentChunk objects.
        """

        # Normalize input text to avoid 1-word-per-line issues
        normalized_text = self._normalize_text(text)

        if strategy == "fixed_size":
            chunks_text = fixed_size_chunking(
                normalized_text,
                chunk_size=settings.DEFAULT_FIXED_CHUNK_SIZE,
                overlap=settings.DEFAULT_FIXED_CHUNK_OVERLAP
            )
        elif strategy == "recursive_character":
            chunks_text = recursive_character_chunking(
                normalized_text,
                chunk_size=settings.DEFAULT_RECURSIVE_CHUNK_SIZE,
                overlap=settings.DEFAULT_RECURSIVE_CHUNK_OVERLAP
            )
        else:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")

        document_chunks = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=i,
                chunk_text=chunk_txt,
                embedding=[],  # Filled later
                metadata={"page_num": "N/A"}
            )
            for i, chunk_txt in enumerate(chunks_text)
        ]

        return document_chunks

    def _normalize_text(self, raw_text: str) -> str:
        """
        Fix text formatting: remove linebreaks after every word, collapse to clean paragraphs.
        """
        lines = raw_text.splitlines()
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        merged = " ".join(non_empty_lines)
        return ' '.join(merged.split())

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