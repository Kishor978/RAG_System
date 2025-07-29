from sqlalchemy.orm import Session
from app.database.connection import DocumentMetadata, ChunkMetadata # Import models
from app.models.document import IngestedDocument, DocumentChunk
from typing import Optional

class RelationalDBManager:
    def __init__(self, db: Session):
        self.db = db

    def save_document_metadata(self, doc_data: IngestedDocument) -> DocumentMetadata:
        db_doc = DocumentMetadata(
            document_id=doc_data.document_id,
            filename=doc_data.filename,
            chunking_strategy=doc_data.chunking_strategy,
            num_chunks=doc_data.num_chunks,
            storage_path=doc_data.storage_path,
            timestamp=doc_data.timestamp
        )
        self.db.add(db_doc)
        self.db.commit()
        self.db.refresh(db_doc)
        return db_doc

    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        return self.db.query(DocumentMetadata).filter(DocumentMetadata.document_id == document_id).first()

    def save_chunk_metadata(self, chunk_data: DocumentChunk) -> ChunkMetadata:
        db_chunk = ChunkMetadata(
            chunk_id=chunk_data.chunk_id,
            document_id=chunk_data.document_id,
            chunk_index=chunk_data.chunk_index,
            chunk_text=chunk_data.chunk_text,
            # We don't store embedding_vector_db_id here for now, as Qdrant manages its own internal IDs.
            # If Qdrant provided external IDs we needed to map, we'd add it.
        )
        self.db.add(db_chunk)
        self.db.commit()
        self.db.refresh(db_chunk)
        return db_chunk

    def get_chunks_metadata_by_document_id(self, document_id: str) -> list[ChunkMetadata]:
        return self.db.query(ChunkMetadata).filter(ChunkMetadata.document_id == document_id).all()