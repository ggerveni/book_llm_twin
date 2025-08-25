# Book LLM Twin (Enhanced Local RAG)

An enhanced local, privacy-preserving Retrieval-Augmented Generation (RAG) assistant for your documents. Built with free tools: ZenML, Qdrant, Sentence-Transformers, and Ollama, featuring an improved Streamlit UI and advanced document processing capabilities.

## ✨ Enhancements Over Original
- Advanced chunking strategies and hybrid search
- Improved document processing and metadata extraction
- Enhanced UI with chat history and source citations
- Fine-tuning preparation and custom prompt templates
- Better error handling and dimension mismatch detection

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Services
```bash
docker compose -f docker/docker-compose.yml up -d
ollama serve
```

### 3. Environment Configuration
Create a `.env` file at the repo root:

```env
# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=book

# Document Processing
DATA_DIR=data/raw
CHUNK_TOKENS=250
OVERLAP_TOKENS=40

# Embeddings (recommended models)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# EMBEDDING_MODEL=intfloat/e5-base-v2
# EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Ollama LLM Configuration
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_NUM_PREDICT=256
OLLAMA_TEMPERATURE=0.2

# RAG Settings
PROMPT_MAX_CONTEXTS=4
PROMPT_MAX_CHARS_PER_CONTEXT=1200
TOP_K=5
```

### 4. Add Your Documents
Place your PDF/TXT files in `data/raw/` (or your configured `DATA_DIR`)

### 5. Ingest Documents
```bash
python scripts/ingest.py
```

### 6. Launch Application
```bash
streamlit run app/streamlit_app.py
```

### 7. Start Querying
- Ensure "Qdrant collection" in sidebar matches your `QDRANT_COLLECTION`
- Ask questions about your documents in natural language

## 📁 Project Structure
```
├── pipelines/ingestion_pipeline.py  # ZenML pipeline for processing
├── steps/                          # Modular ZenML processing steps
├── rag/                           # Retrieval and generation components
├── app/streamlit_app.py           # Enhanced Streamlit frontend
├── scripts/ingest.py              # CLI pipeline runner
└── data/raw/                      # Your documents go here
```

## 🔧 Configuration Tips

### Embedding Models
- **Quality/Speed Balance**: `intfloat/e5-base-v2` (768d)
- **Fast/CPU-Friendly**: `sentence-transformers/all-MiniLM-L6-v2` (384d)
- **Multilingual**: `BAAI/bge-small-en-v1.5` (384d)

### LLM Models (Ollama)
- **Balanced Quality**: `llama3.1:8b-instruct`
- **Lightweight/CPU**: `qwen2.5:3b`

### Collection Management
⚠️ **Important**: Changing embedding models requires re-ingestion due to different vector dimensions.

```bash
# List collections
curl http://localhost:6333/collections

# Delete collection (if needed)
curl -X DELETE http://localhost:6333/collections/book
```

## 🛠 Troubleshooting
- **Dimension Mismatch**: The app will detect and display clear errors if embedding model dimensions don't match your collection
- **Memory Issues**: Reduce `CHUNK_TOKENS` or `TOP_K` for large documents
- **Slow Performance**: Try lighter embedding models or reduce `OLLAMA_NUM_PREDICT`

## 🔄 Planned Enhancements
- [ ] Fine-tuning capabilities for domain-specific documents
- [ ] Advanced hybrid search (vector + keyword)
- [ ] Cross-document querying and summarization
- [ ] Improved PDF parsing with table/image support
- [ ] Multi-language document support

---

**Fork of**: [asulova/book_llm_twin](https://github.com/asulova/book_llm_twin)  
**Enhanced by**: Griseld Gerveni