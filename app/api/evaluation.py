from fastapi import APIRouter, Depends, HTTPException, status, Body, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
from datetime import datetime
import uuid
import os

from app.database.connection import get_db
from app.services.document_processor import document_processor
from app.services.vector_db_manager import qdrant_manager
from app.services.evaluator import RAGEvaluator
from app.schemas import EvaluationReport, EvaluationMetric
from app.core.config import BASE_DIR

router = APIRouter()

# Initialize evaluator
evaluator = RAGEvaluator(
    document_processor=document_processor,
    vector_db_manager=qdrant_manager
)

class EvaluationRequest(BaseModel):
    evaluation_documents: List[Dict[str, str]] = Field(
        ..., description="List of documents for evaluation with document_id and text"
    )
    test_queries: List[Dict[str, Any]] = Field(
        ..., description="List of test queries with expected relevant document IDs"
    )
    chunking_methods: List[str] = Field(
        default=["fixed_size", "recursive_character"], 
        description="Chunking methods to evaluate"
    )
    similarity_algorithms: List[str] = Field(
        default=["cosine", "dot_product"], 
        description="Similarity algorithms to evaluate"
    )

@router.post("/evaluate", response_model=Dict[str, str])
async def evaluate_rag_system(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Evaluate the RAG system with different configurations.
    This runs the evaluation in the background and saves the results to the database.
    """
    # Validate input
    if not request.evaluation_documents or not request.test_queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Evaluation documents and test queries cannot be empty."
        )
    
    # Generate an evaluation ID
    evaluation_id = str(uuid.uuid4())
    
    # Run evaluation in the background
    background_tasks.add_task(
        run_evaluation_and_save_report,
        evaluation_id,
        request.evaluation_documents,
        request.test_queries,
        request.chunking_methods,
        request.similarity_algorithms,
        db
    )
    
    return {
        "message": "Evaluation started successfully. Check the status endpoint for results.",
        "evaluation_id": evaluation_id
    }

@router.get("/evaluate/{evaluation_id}")
async def get_evaluation_status(evaluation_id: str):
    """
    Get the status of an evaluation or the final report if it's complete.
    """
    # Check if the report file exists
    report_path = BASE_DIR / "data" / f"evaluation_report_{evaluation_id}.md"
    
    if not report_path.exists():
        return {
            "status": "in_progress",
            "message": "Evaluation is still running or hasn't been started."
        }
    
    # Read the report
    with open(report_path, "r") as f:
        report_content = f.read()
    
    return {
        "status": "completed",
        "report": report_content
    }

async def run_evaluation_and_save_report(
    evaluation_id: str,
    evaluation_documents: List[Dict[str, str]],
    test_queries: List[Dict[str, Any]],
    chunking_methods: List[str],
    similarity_algorithms: List[str],
    db: Session
):
    """
    Run the evaluation and save the results.
    This is meant to be run as a background task.
    """
    try:
        # Run the evaluation
        evaluation_report = evaluator.evaluate_chunking_and_search(
            evaluation_documents,
            test_queries,
            chunking_methods,
            similarity_algorithms
        )
        
        # Generate a human-readable report
        report_text = evaluator.generate_report(evaluation_report)
        
        # Save the report as a markdown file
        reports_dir = BASE_DIR / "data"
        os.makedirs(reports_dir, exist_ok=True)
        
        report_path = reports_dir / f"evaluation_report_{evaluation_id}.md"
        with open(report_path, "w") as f:
            f.write(report_text)
        
        # Also save the raw report data as JSON
        raw_report_path = reports_dir / f"evaluation_report_{evaluation_id}.json"
        with open(raw_report_path, "w") as f:
            # Convert the Pydantic model to a dictionary
            report_dict = evaluation_report.model_dump()
            # Convert datetime to string
            report_dict["timestamp"] = report_dict["timestamp"].isoformat()
            json.dump(report_dict, f, indent=2)
        
        # Save metrics to the database (in a real implementation)
        # for metric in evaluation_report.metrics:
        #     db_metric = EvaluationResult(
        #         chunking_method=metric.chunking_method,
        #         similarity_algorithm=metric.similarity_algorithm,
        #         accuracy=metric.accuracy,
        #         precision=metric.precision,
        #         recall=metric.recall,
        #         f1_score=metric.f1_score,
        #         latency=metric.latency
        #     )
        #     db.add(db_metric)
        # db.commit()
        
    except Exception as e:
        # Log the error
        print(f"An error occurred during evaluation: {e}")
        
        # Save the error to a file
        error_path = BASE_DIR / "data" / f"evaluation_error_{evaluation_id}.txt"
        with open(error_path, "w") as f:
            f.write(f"Error during evaluation: {str(e)}")
