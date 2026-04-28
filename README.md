# Semantic Code Search Engine

This project is an end-to-end Semantic Code Search Engine built as a portfolio project. Unlike traditional keyword-based search (which relies on exact string matching), this engine uses machine learning to understand the *meaning* and *intent* behind a natural language query (e.g., "function that reads a CSV and returns a dataframe") and retrieves the most conceptually relevant Python code snippets.

## Architecture

The project consists of several modular components:

1. **Embedding Model (`sentence-transformers`)**: We use the pre-trained `flax-sentence-embeddings/st-codesearch-distilroberta-base` model. This model transforms both our code snippets and the user's natural language queries into 768-dimensional vector embeddings.
2. **Vector Index (`faiss-cpu`)**: FAISS (Facebook AI Similarity Search) is used to efficiently search through the vector embeddings. We utilize an `IndexFlatIP` exact search index. Because our vectors are L2 normalized, the Inner Product operation mathematically computes Cosine Similarity, which focuses on the "direction" (meaning) of the vectors rather than their magnitude.
3. **Backend API (`FastAPI` & `uvicorn`)**: A high-performance Python web framework that wraps our retrieval logic, exposing a `/search` endpoint to accept queries and return JSON results.
4. **Interactive CLI (`rich`)**: An optional, visually appealing command-line interface for querying the API and rendering syntax-highlighted code.

## File Overview

* `generate_corpus.py`: Generates a synthetic dataset (`corpus.json`) of Python code snippets and their natural language descriptions.
* `generate_embeddings.py`: Uses the embedding model to convert the corpus into vectors and saves them to `embeddings.npy`.
* `build_index.py`: Loads the embeddings, normalizes them, builds the FAISS index, and saves it to `code_index.faiss`.
* `search.py`: Contains the `SemanticSearchEngine` class which handles the core retrieval logic.
* `main.py`: The FastAPI server.
* `cli.py`: The interactive command-line interface.
* `evaluate.py`: Calculates Mean Reciprocal Rank (MRR) to objectively measure search quality.

## Design Decisions & Tradeoffs

* **Combined Embeddings**: We embed a concatenation of the snippet's description and the code itself (`"{desc}\n\n{code}"`). This bridges the vocabulary gap between a user's natural language intent and the specific syntactic implementation in the code.
* **Exact vs. Approximate Search**: We opted for Exact Search (`IndexFlatIP`) because our corpus is small, ensuring perfect cosine similarity matches. If scaling to millions of snippets, we would trade slight accuracy for speed by migrating to an Approximate Nearest Neighbour (ANN) index like HNSW or IVF.
* **Why Cosine Similarity over Euclidean (L2)?**: Cosine similarity cares about the angle between vectors (the core semantic meaning) and ignores the magnitude (the length of the text).

## Setup & Running

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Generate the Corpus**: `python generate_corpus.py`
3. **Generate Embeddings**: `python generate_embeddings.py`
4. **Build the FAISS Index**: `python build_index.py`
5. **Start the API Server**: `python main.py`
6. **Query the API**: Open a new terminal and run `python cli.py`
7. **Run Evaluations**: `python evaluate.py`