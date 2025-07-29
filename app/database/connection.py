# rag_system/app/database/connection.py
from sqlalchemy import create_engine, Column, String, Integer, Float, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings
import uuid
import json
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent.parent

DATABASE_URL: str = f"sqlite:///{str(BASE_DIR / 'data' / 'metadata.db').replace('\\', '/')}" # Try this instead


# SQLAlchemy Engine
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define ORM model for document metadata
class DocumentMetadata(Base):
    __tablename__ = "document_metadata"
    document_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    chunking_strategy = Column(String, nullable=False)
    num_chunks = Column(Integer, nullable=False)
    storage_path = Column(String, nullable=True)
    timestamp = Column(Float, nullable=False) # Store Unix timestamp as float for simplicity

    def __repr__(self):
        return f"<DocumentMetadata(document_id='{self.document_id}', filename='{self.filename}')>"

class ChunkMetadata(Base):
    __tablename__ = "chunk_metadata"
    chunk_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False) # Store chunk text for quick retrieval/debugging

    def __repr__(self):
        return f"<ChunkMetadata(chunk_id='{self.chunk_id}', document_id='{self.document_id}', index={self.chunk_index})>"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
def create_db_and_tables():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    create_db_and_tables()
    print("Database and tables created/checked successfully.")