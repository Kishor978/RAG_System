# RAG System Backend

This repository contains the backend for a Retrieval-Augmented Generation (RAG) system, built with FastAPI. It handles document ingestion, text chunking, embedding generation, and storage in a vector database (Qdrant) and relational database (SQLite).

## Features

* **Document Ingestion:** Upload PDF and plain text files.

* **Text Extraction:** Extracts text content from uploaded documents.

* **Text Chunking:** Divides extracted text into smaller, manageable chunks using fixed-size or recursive character splitting strategies.

* **Embedding Generation:** Converts text chunks into numerical vector embeddings using a pre-trained Sentence Transformer model.

* **Vector Database Storage:** Stores text embeddings and chunk metadata in Qdrant for efficient similarity search.

* **Relational Database Storage:** Persists document and chunk metadata in an SQLite database for structured querying and management.

* **Conversational RAG:** Handles multi-turn conversations with context awareness using Redis for memory storage.

* **Interview Booking:** Supports booking interview appointments with email confirmations.

* **Evaluation Framework:** Compares different chunking methods and similarity search algorithms on accuracy, precision, recall, F1-score, and latency.

* **FastAPI Backend:** Provides a robust and asynchronous API for interacting with the system.


## Project Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Kishor978/RAG_System.git
   cd RAG_System
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Redis using Docker:
   ```bash
   # Pull radis docker image
   docker pull redis

   # Using docker run
   docker run --name redis-rag -p 6379:6379 -d redis
   ```

4. Test Redis connection:
   ```bash
   python redis_setup.py
   ```
5. Setup qdrant using docker
   ```bash
   # Pull the docker image for qdrant
   docker pull qdrant/qdrant

   # Run Docker
   docker run -d --name qdrant_local -p 6333:6333 -p 6334:6334 ^ -v %cd%/qdrant_data:/qdrant/storage ^ qdrant/qdrant

   # Verify containers is running
   docker ps
   ```
5. Start the application:
   ```bash
   uvicorn app.main:app --reload
   ```

6. Access the API documentation:
   ```
   http://localhost:8000/docs
   ```


## Architecture

- `app/api`: API endpoints
- `app/services`: Core services (document processing, RAG, memory management)
- `app/models`: Data models
- `app/database`: Database connections
- `app/utils`: Utility functions (text splitters)
- `app/core`: Configuration and settings
