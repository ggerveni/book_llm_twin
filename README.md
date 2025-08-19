## Book LLM Twin (Local RAG)

A local, privacy-preserving Retrieval-Augmented Generation (RAG) assistant for your own books/documents using free tools: ZenML, Qdrant, Sentence-Transformers, and Ollama, with a Streamlit UI.

### Quickstart

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Start services:
```
docker compose -f docker/docker-compose.yml up -d
ollama serve
```

3. Create a `.env` at the repo root (adjust values as needed):
```
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=book

# Data & chunking
DATA_DIR=data/raw
CHUNK_TOKENS=250
OVERLAP_TOKENS=40

# Embeddings (choose one; for English, good options below)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# EMBEDDING_MODEL=intfloat/e5-base-v2
# EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Ollama LLM
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_NUM_PREDICT=256
OLLAMA_TEMPERATURE=0.2

# Prompt limits
PROMPT_MAX_CONTEXTS=4
PROMPT_MAX_CHARS_PER_CONTEXT=1200

# Optional UI defaults
TOP_K=5
```

4. Put your book PDFs/TXT files under `data/raw` (or set `DATA_DIR` accordingly).

5. Ingest documents (uses only `.env` or explicit CLI flags):
```
python scripts/ingest.py
```

6. Launch the UI:
```
streamlit run app/streamlit_app.py
```

7. In the sidebar, ensure the "Qdrant collection" field matches `QDRANT_COLLECTION` (e.g., `book`). Ask questions in English.

### Notes & Tips
- Changing the embedding model requires re-ingestion into a collection with the matching vector size.
  - If you switch from a 384-d model (e.g., MiniLM) to a 768-d model (e.g., e5-base-v2), either:
    - Re-ingest into a new collection name, or
    - Delete/recreate the existing collection.
- The app now checks for embedding/collection dimension mismatches and will display a clear error if they don’t match.
- To remove old collections via REST:
```
# List collections
(Invoke-RestMethod http://localhost:6333/collections).result.collections | Select-Object name

# Delete (examples)
Invoke-RestMethod -Method DELETE http://localhost:6333/collections/book
```

### Project Structure
- `pipelines/ingestion_pipeline.py`: load → chunk → embed → index (ZenML pipeline)
- `steps/`: modular ZenML steps
- `rag/`: retriever, prompt, and generator
- `app/streamlit_app.py`: Streamlit frontend
- `scripts/ingest.py`: CLI pipeline runner

### Model recommendations (English)
- Embeddings:
  - Quality/speed balance: `intfloat/e5-base-v2`
  - Fast/CPU-friendly: `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`
- Generator (Ollama):
  - Balanced quality: `llama3.1:8b-instruct`
  - Lighter/CPU: `qwen2.5:3b`

