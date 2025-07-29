from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.models.document import DocumentChunk
from typing import List, Dict, Any, Optional
import uuid

class VectorDBManager:
    def __init__(self, host: str, port: int, collection_name: str, vector_size: int):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensures the Qdrant collection exists or creates it."""
        try:
            self.client.get_collection(collection_name=self.collection_name)
            print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            print(f"Collection '{self.collection_name}' does not exist. Creating it...")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE),
            )
            print(f"Collection '{self.collection_name}' created.")

    def upsert_chunks(self, chunks: List[DocumentChunk]):
        """
        Upserts document chunks into the Qdrant collection.
        Each chunk becomes a point in Qdrant.
        """
        points = []
        for chunk in chunks:
            # We use a UUID for the Qdrant point ID, could also use chunk.chunk_id
            point_id = chunk.chunk_id
            payload = chunk.metadata.copy()
            payload.update({
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text # Store text in payload for retrieval
            })
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload=payload
                )
            )
        
        if points:
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=points
            )
            print(f"Upserted {len(points)} points to Qdrant. Status: {operation_info.status}")
            return operation_info
        return None

    def search_similar_chunks(self, query_embedding: List[float], limit: int = 5, document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Searches for similar chunks in the Qdrant collection.
        """
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True # Retrieve the chunk text and other metadata
        )
        return [
            {
                "chunk_id": hit.payload.get("chunk_id"),
                "document_id": hit.payload.get("document_id"),
                "chunk_text": hit.payload.get("chunk_text"),
                "score": hit.score
            } for hit in search_result
        ]

    def delete_document_chunks(self, document_id: str):
        """Deletes all chunks associated with a document_id from Qdrant."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
        print(f"Deleted chunks for document_id: {document_id}")

# Initialize Qdrant client globally or via dependency injection
qdrant_manager = VectorDBManager(
    host=settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    collection_name=settings.QDRANT_COLLECTION_NAME,
    vector_size=settings.EMBEDDING_DIMENSION
)