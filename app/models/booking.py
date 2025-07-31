from sqlalchemy import Column, String, Integer, Float, Text, DateTime
from sqlalchemy.sql import func
from app.database.connection import Base
import uuid

class Booking(Base):
    __tablename__ = "bookings"
    
    booking_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, nullable=False)
    date = Column(String, nullable=False)  # Store as string for flexibility
    time = Column(String, nullable=False)  # Store as string for flexibility
    status = Column(String, default="confirmed")  # confirmed, canceled, completed
    created_at = Column(DateTime, server_default=func.now())
    
    def __repr__(self):
        return f"<Booking(id={self.booking_id}, name='{self.name}', date='{self.date}')>"

class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True)
    chunking_method = Column(String, nullable=False)
    similarity_algorithm = Column(String, nullable=False)
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    latency = Column(Float, nullable=False)  # in milliseconds
    timestamp = Column(DateTime, server_default=func.now())
    
    def __repr__(self):
        return f"<EvaluationResult(method='{self.chunking_method}', algorithm='{self.similarity_algorithm}', f1={self.f1_score})>"
