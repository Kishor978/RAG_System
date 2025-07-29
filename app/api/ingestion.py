from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Literal, Optional, List
import uuid
import time

from app.models.document import IngestedDocument, DocumentChunk
from app.services.document_processor import document_processor
from app.services.vector_db_manager import qdrant_manager
from app.services.relational_db_manager import RelationalDBManager
from app.database.connection import get_db, create_db_and_tables # Import create_db_and_tables

router = APIRouter()

@router.post("/ingest", response_model=IngestedDocument, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile = File(...),
    chunking_strategy: Literal["fixed_size", "recursive_character"] = Form("recursive_character"),
    document_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Uploads .pdf or .txt files, extracts text, applies chunking, generates embeddings,
    and stores them in Qdrant and metadata in the SQL DB.
    """
    if not document_id:
        document_id = str(uuid.uuid4())

    # 1. Validate file type
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Only .pdf and .txt are allowed."
        )

    file_content = await file.read()
    
    try:
        # 2. Extract text
        extracted_text = document_processor.extract_text(file_content, file.content_type)
        
        if not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract text from the document. The document might be empty or scanned without OCR."
            )
        print(f"Extracted text length: {len(extracted_text)} characters")  # Debugging: Print length of extracted text
        # 3. Apply chunking strategies
        chunks_without_embeddings: List[DocumentChunk] = document_processor.chunk_text(
            extracted_text, chunking_strategy, document_id
        )
        print(f"Number of chunks generated: {len(chunks_without_embeddings)}")  # Debugging: Print number of chunks
        if not chunks_without_embeddings:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No chunks were generated from the document. Please check chunking parameters or document content."
            )

        # 4. Generate embeddings
        ingestable_chunks = document_processor.generate_embeddings(chunks_without_embeddings)
        print(f"Number of chunks with embeddings: {len(ingestable_chunks)}")  # Debugging: Print number of chunks with embeddings
        # 5. Store embeddings & chunk metadata in Qdrant
        qdrant_manager.upsert_chunks(ingestable_chunks)
        print("Chunks successfully upserted to Qdrant.")  # Debugging: Confirm chunks were upserted
        # 6. Save metadata in SQL DB
        relational_db_manager = RelationalDBManager(db)
        ingested_doc_data = IngestedDocument(
            document_id=document_id,
            filename=file.filename,
            chunking_strategy=chunking_strategy,
            num_chunks=len(ingestable_chunks),
            timestamp=time.time()
        )
        print(f"Ingested document metadata: {ingested_doc_data}")  # Debugging: Print ingested document metadata
        # Save main document metadata
        relational_db_manager.save_document_metadata(ingested_doc_data)
        print("Document metadata successfully saved to SQL DB.")  # Debugging: Confirm metadata was saved
        # Save individual chunk metadata (optional, but useful for debugging/detailed info)
        for chunk in ingestable_chunks:
            relational_db_manager.save_chunk_metadata(chunk)
        print("Chunk metadata successfully saved to SQL DB.")  # Debugging: Confirm chunk metadata was saved
        return ingested_doc_data

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        # Log the error for debugging
        print(f"An unexpected error occurred during ingestion: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred during document ingestion.")