# RAG System Backend

This repository contains the backend for a Retrieval-Augmented Generation (RAG) system, built with FastAPI. It handles document ingestion, text chunking, embedding generation, and storage in a vector database (Qdrant) and relational database (SQLite).

## Features

* **Document Ingestion:** Upload PDF and plain text files.

* **Text Extraction:** Extracts text content from uploaded documents.

* **Text Chunking:** Divides extracted text into smaller, manageable chunks using a recursive character splitting strategy.

* **Embedding Generation:** Converts text chunks into numerical vector embeddings using a pre-trained Sentence Transformer model.

* **Vector Database Storage:** Stores text embeddings and chunk metadata in Qdrant for efficient similarity search.

* **Relational Database Storage:** Persists document and chunk metadata in an SQLite database for structured querying and management.

* **FastAPI Backend:** Provides a robust and asynchronous API for interacting with the system.


## Project Setup
- Clone repo, create environment and install requirements
- Pull the docker image for qdrant

    `docker pull qdrant/qdrant`
- Run Docker

    `docker run -d --name qdrant_local -p 6333:6333 -p 6334:6334 ^ -v %cd%/qdrant_data:/qdrant/storage ^ qdrant/qdrant`
- Run API

    `uvicorn app/main:app --reload`

