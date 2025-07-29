import io
from typing import List, Literal, Tuple
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from app.models.document import DocumentChunk
from app.utils.text_splitters import fixed_size_chunking, recursive_character_chunking
from app.core.config import settings
import logging
import pdfplumber

logging.basicConfig(level=logging.INFO) # Set logging level for better visibility
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, embedding_model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Loaded embedding model: {embedding_model_name}")
    def normalize_text(self, raw_text: str) -> str:
        lines = raw_text.splitlines()
        joined = " ".join(line.strip() for line in lines if line.strip())
        return ' '.join(joined.split())

    def extract_text(self, file_content: bytes, file_type: str) -> str:
        if file_type == "application/pdf":
            try:
                text = self._extract_text_from_pdf(file_content)
                if not text.strip():
                    raise ValueError("Empty text from PDF — possibly scanned.")
                return self.normalize_text(text)
            except Exception:
                logger.warning("Falling back to OCR for PDF.")
                return self.normalize_text(self.ocr_pdf(file_content))
        elif file_type == "text/plain":
            return self.normalize_text(file_content.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    def _extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        try:
            text_pages = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)
                    else:
                        logger.warning(f"No text found on page {i + 1} (may be scanned image).")
            
            combined_text = "\n\n".join(text_pages)
            if not combined_text.strip():
                raise ValueError("No extractable text. This may be a scanned PDF or image-only.")
            
            logger.info("Text extracted using pdfplumber.")
            return combined_text

        except Exception as e:
            logger.error(f"PDF loading failed: {e}", exc_info=True)
            raise ValueError("PDF parsing failed.")

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