import time
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from app.services.document_processor import DocumentProcessor
from app.utils.text_splitters import fixed_size_chunking, recursive_character_chunking
from app.services.vector_db_manager import VectorDBManager
from app.models.document import DocumentChunk
from app.schemas import EvaluationMetric, EvaluationReport
import logging
from qdrant_client import QdrantClient, models
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Service for evaluating RAG system performance with different configurations."""
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        vector_db_manager: VectorDBManager
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            document_processor: Document processor for embedding
            vector_db_manager: Vector database manager
        """
        self.document_processor = document_processor
        self.vector_db_manager = vector_db_manager
        # Use the same Qdrant client but with temporary collections for evaluation
        self.qdrant_client = vector_db_manager.client
        self.vector_size = vector_db_manager.vector_size
        
    def evaluate_chunking_and_search(
        self,
        evaluation_documents: List[Dict[str, str]],
        test_queries: List[Dict[str, Any]],
        chunking_methods: List[str] = ["fixed_size", "recursive_character"],
        similarity_algorithms: List[str] = ["cosine", "dot_product"]
    ) -> EvaluationReport:
        """
        Evaluate different chunking methods and similarity algorithms.
        
        Args:
            evaluation_documents: List of documents with text and IDs for evaluation
            test_queries: List of test queries with expected relevant document IDs
            chunking_methods: List of chunking methods to evaluate
            similarity_algorithms: List of similarity algorithms to evaluate
            
        Returns:
            EvaluationReport with metrics for each combination
        """
        metrics = []
        
        # Process evaluation documents and create chunked versions for each method
        chunked_docs = {}
        for method in chunking_methods:
            chunked_docs[method] = self._process_documents_with_chunking(evaluation_documents, method)
        
        # Evaluate each combination of chunking method and similarity algorithm
        for chunking_method in chunking_methods:
            for similarity_algorithm in similarity_algorithms:
                logger.info(f"Evaluating: {chunking_method} with {similarity_algorithm}")
                
                # Use the pre-processed chunks for this method
                chunks = chunked_docs[chunking_method]
                
                # Run evaluation with this configuration
                metric = self._evaluate_configuration(
                    chunks, test_queries, chunking_method, similarity_algorithm
                )
                
                metrics.append(metric)
                logger.info(f"Results: {metric}")
        
        # Find the best combination based on F1 score
        if metrics:
            best_metric = max(metrics, key=lambda x: x.f1_score)
            best_combination = {
                "chunking_method": best_metric.chunking_method,
                "similarity_algorithm": best_metric.similarity_algorithm,
                "f1_score": str(best_metric.f1_score)  # Convert to string to match schema expectations
            }
        else:
            # Fallback if no metrics were collected
            best_combination = {
                "chunking_method": "none",
                "similarity_algorithm": "none",
                "f1_score": "0.0"
            }
        
        return EvaluationReport(
            metrics=metrics,
            best_combination=best_combination,
            notes=f"Evaluation completed on {len(test_queries)} test queries against {len(evaluation_documents)} documents."
        )
    
    def _process_documents_with_chunking(
        self, 
        documents: List[Dict[str, str]], 
        chunking_method: str
    ) -> List[DocumentChunk]:
        """
        Process documents with a specific chunking method.
        
        Args:
            documents: List of documents with text and IDs
            chunking_method: Chunking method to use
            
        Returns:
            List of processed document chunks
        """
        all_chunks = []
        
        for doc in documents:
            # Create chunks with the specified method
            chunks = self.document_processor.chunk_text(
                doc["text"], chunking_method, doc["document_id"]
            )
            
            # Generate embeddings
            chunks_with_embeddings = self.document_processor.generate_embeddings(chunks)
            all_chunks.extend(chunks_with_embeddings)
        
        return all_chunks
    
    def _create_temp_collection(self, chunks: List[DocumentChunk], distance_metric: str) -> str:
        """
        Create a temporary collection in Qdrant for evaluation purposes.
        
        Args:
            chunks: List of document chunks to add to the collection
            distance_metric: Distance metric to use (cosine or dot_product)
            
        Returns:
            Name of the temporary collection
        """
        # Create a unique name for the temporary collection
        collection_name = f"temp_eval_{uuid.uuid4().hex[:8]}"
        
        # Map string distance metric to Qdrant enum
        if distance_metric.lower() == "cosine":
            distance = models.Distance.COSINE
        elif distance_metric.lower() == "dot_product":
            distance = models.Distance.DOT
        else:
            logger.warning(f"Unknown distance metric: {distance_metric}. Using COSINE as default.")
            distance = models.Distance.COSINE
        
        # Create the temporary collection
        self.qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=self.vector_size, distance=distance),
        )
        
        # Insert chunks into the collection
        points = []
        for chunk in chunks:
            point_id = chunk.chunk_id
            payload = chunk.metadata.copy() if chunk.metadata else {}
            payload.update({
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text
            })
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload
                )
            )
        
        if points:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            logger.info(f"Created temporary collection '{collection_name}' with {len(points)} points")
        
        return collection_name
    
    def _delete_temp_collection(self, collection_name: str):
        """
        Delete a temporary collection.
        
        Args:
            collection_name: Name of the collection to delete
        """
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted temporary collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting temporary collection: {e}")
    
    def _search_similar_chunks(self, collection_name: str, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the specified collection.
        
        Args:
            collection_name: Name of the collection to search
            query_embedding: Query embedding vector
            limit: Maximum number of results to return
            
        Returns:
            List of similar chunks with metadata
        """
        search_result = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        return [
            {
                "chunk_id": hit.payload.get("chunk_id"),
                "document_id": hit.payload.get("document_id"),
                "chunk_text": hit.payload.get("chunk_text"),
                "score": hit.score
            } for hit in search_result
        ]
    
    def _evaluate_configuration(
        self,
        chunks: List[DocumentChunk],
        test_queries: List[Dict[str, Any]],
        chunking_method: str,
        similarity_algorithm: str
    ) -> EvaluationMetric:
        """
        Evaluate a specific configuration on test queries.
        
        Args:
            chunks: List of document chunks
            test_queries: List of test queries with expected document IDs
            chunking_method: Chunking method used
            similarity_algorithm: Similarity algorithm to use
            
        Returns:
            EvaluationMetric object with results
        """
        # Create temporary collection for this evaluation
        temp_collection = self._create_temp_collection(chunks, similarity_algorithm)
        
        try:
            y_true = []  # Expected relevant document IDs
            y_pred = []  # Predicted relevant document IDs
            
            # Track latency
            latencies = []
            
            for query in test_queries:
                query_text = query["query"]
                expected_doc_ids = set(query["relevant_doc_ids"])
                
                # Generate query embedding
                start_time = time.time()
                query_embedding = self.document_processor.embedding_model.encode(query_text, convert_to_list=True)
                
                # Perform search in the temporary collection
                retrieved_chunks = self._search_similar_chunks(
                    temp_collection, query_embedding, limit=5
                )
                end_time = time.time()
                
                # Record latency in milliseconds
                latencies.append((end_time - start_time) * 1000)
                
                # Extract unique document IDs from retrieved chunks
                retrieved_doc_ids = set(chunk["document_id"] for chunk in retrieved_chunks)
                
                # For binary classification metrics, we need to evaluate whether each expected document was retrieved
                for doc_id in expected_doc_ids:
                    y_true.append(1)  # This document should have been retrieved
                    y_pred.append(1 if doc_id in retrieved_doc_ids else 0)
                
                # Also account for false positives
                for doc_id in retrieved_doc_ids:
                    if doc_id not in expected_doc_ids:
                        y_true.append(0)  # This document should not have been retrieved
                        y_pred.append(1)  # But it was retrieved
            
            # Calculate metrics
            if not y_true:
                return EvaluationMetric(
                    chunking_method=chunking_method,
                    similarity_algorithm=similarity_algorithm,
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1_score=0.0,
                    latency=0.0
                )
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            return EvaluationMetric(
                chunking_method=chunking_method,
                similarity_algorithm=similarity_algorithm,
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                latency=float(avg_latency)
            )
            
        finally:
            # Always clean up the temporary collection
            self._delete_temp_collection(temp_collection)
    
    def generate_report(self, evaluation_report: EvaluationReport) -> str:
        """
        Generate a readable report from the evaluation results.
        
        Args:
            evaluation_report: The evaluation report to format
            
        Returns:
            Formatted report as a string
        """
        report_lines = []
        
        # Header
        report_lines.append("# RAG System Evaluation Report")
        report_lines.append(f"Date: {evaluation_report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"Best Configuration: **{evaluation_report.best_combination['chunking_method']}** chunking with **{evaluation_report.best_combination['similarity_algorithm']}** similarity")
        
        # Handle f1_score which is now a string in the best_combination dictionary
        try:
            f1_score = float(evaluation_report.best_combination['f1_score'])
            report_lines.append(f"Best F1 Score: **{f1_score:.4f}**")
        except (ValueError, TypeError):
            # If conversion fails, just use the string value directly
            report_lines.append(f"Best F1 Score: **{evaluation_report.best_combination['f1_score']}**")
        if evaluation_report.notes:
            report_lines.append(f"Notes: {evaluation_report.notes}")
        report_lines.append("")
        
        # Detailed Results
        report_lines.append("## Detailed Results")
        report_lines.append("| Chunking Method | Similarity Algorithm | Accuracy | Precision | Recall | F1 Score | Latency (ms) |")
        report_lines.append("|----------------|----------------------|----------|-----------|--------|----------|--------------|")
        
        # Sort metrics by F1 score (descending)
        sorted_metrics = sorted(evaluation_report.metrics, key=lambda x: x.f1_score, reverse=True)
        
        for metric in sorted_metrics:
            report_lines.append(
                f"| {metric.chunking_method} | {metric.similarity_algorithm} | "
                f"{metric.accuracy:.4f} | {metric.precision:.4f} | {metric.recall:.4f} | "
                f"{metric.f1_score:.4f} | {metric.latency:.2f} |"
            )
        
        report_lines.append("")
        
        # Analysis
        report_lines.append("## Analysis")
        report_lines.append("### Chunking Methods Comparison")
        
        # Compare chunking methods by averaging metrics across similarity algorithms
        chunking_methods = set(metric.chunking_method for metric in evaluation_report.metrics)
        for method in chunking_methods:
            method_metrics = [m for m in evaluation_report.metrics if m.chunking_method == method]
            avg_f1 = sum(m.f1_score for m in method_metrics) / len(method_metrics)
            avg_latency = sum(m.latency for m in method_metrics) / len(method_metrics)
            report_lines.append(f"- **{method}**: Average F1: {avg_f1:.4f}, Average Latency: {avg_latency:.2f} ms")
        
        report_lines.append("")
        report_lines.append("### Similarity Algorithms Comparison")
        
        # Compare similarity algorithms by averaging metrics across chunking methods
        algorithms = set(metric.similarity_algorithm for metric in evaluation_report.metrics)
        for alg in algorithms:
            alg_metrics = [m for m in evaluation_report.metrics if m.similarity_algorithm == alg]
            avg_f1 = sum(m.f1_score for m in alg_metrics) / len(alg_metrics)
            avg_latency = sum(m.latency for m in alg_metrics) / len(alg_metrics)
            report_lines.append(f"- **{alg}**: Average F1: {avg_f1:.4f}, Average Latency: {avg_latency:.2f} ms")
        
        return "\n".join(report_lines)
