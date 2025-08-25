"""
Enhanced chunk_documents step with advanced semantic chunking capabilities.
This replaces the original steps/chunk_documents.py with more sophisticated chunking.
"""

from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

# Import our advanced chunking module
from advanced_chunking import (
    chunk_document_advanced, 
    convert_to_zenml_format,
    ChunkingStrategy
)

load_dotenv()

# ZenML step decorator - if ZenML is available
try:
    from zenml import step
    ZENML_AVAILABLE = True
except ImportError:
    # Fallback decorator if ZenML isn't installed
    def step(enable_cache=False):
        def decorator(func):
            return func
        return decorator
    ZENML_AVAILABLE = False


@step(enable_cache=False)
def chunk_documents_advanced(
    documents: List[Dict],
    chunk_tokens: int = 250,
    overlap_sentences: int = 2,
    chunking_strategy: str = "hybrid",
    similarity_threshold: float = 0.5,
    embedding_model_name: Optional[str] = None,
) -> List[Dict]:
    """
    Enhanced document chunking with semantic awareness and multiple strategies.
    
    Args:
        documents: List of document dictionaries from load_documents step
        chunk_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        chunking_strategy: Strategy to use ("sentence_based", "semantic_similarity", "hybrid", "token_aware")
        similarity_threshold: Threshold for semantic similarity chunking
        embedding_model_name: Model name for embeddings (defaults to env var)
        
    Returns:
        List of chunk dictionaries compatible with existing pipeline
    """
    
    # Get embedding model from environment if not provided
    if not embedding_model_name:
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Convert string strategy to enum
    strategy_map = {
        "sentence_based": ChunkingStrategy.SENTENCE_BASED,
        "semantic_similarity": ChunkingStrategy.SEMANTIC_SIMILARITY,
        "hybrid": ChunkingStrategy.HYBRID,
        "token_aware": ChunkingStrategy.TOKEN_AWARE,
    }
    
    strategy = strategy_map.get(chunking_strategy, ChunkingStrategy.HYBRID)
    
    print(f"Using {chunking_strategy} chunking strategy with {embedding_model_name}")
    print(f"Target: {chunk_tokens} tokens per chunk, {overlap_sentences} sentence overlap")
    
    all_chunks = []
    
    for doc in documents:
        text = doc.get("text", "")
        path = doc.get("path", "")
        doc_id = doc.get("doc_id", "unknown")
        
        if not text or not text.strip():
            continue
            
        # Prepare source metadata
        source_metadata = {
            "original_path": path,
            "document_id": doc_id,
            "chunking_method": "advanced_semantic",
            "chunking_strategy": chunking_strategy,
            "similarity_threshold": similarity_threshold,
        }
        
        # Get advanced chunks
        try:
            document_chunks = chunk_document_advanced(
                text=text,
                strategy=strategy,
                max_chunk_tokens=chunk_tokens,
                overlap_sentences=overlap_sentences,
                similarity_threshold=similarity_threshold,
                embedding_model_name=embedding_model_name,
                source_metadata=source_metadata
            )
            
            # Convert to ZenML format for pipeline compatibility
            zenml_chunks = convert_to_zenml_format(document_chunks, doc_id, path)
            all_chunks.extend(zenml_chunks)
            
            print(f"✓ {doc_id}: {len(document_chunks)} chunks created")
            
        except Exception as e:
            print(f"✗ Error chunking {doc_id}: {str(e)}")
            # Fallback to simple chunking if advanced chunking fails
            fallback_chunks = _fallback_simple_chunking(text, doc_id, path, chunk_tokens)
            all_chunks.extend(fallback_chunks)
    
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def _fallback_simple_chunking(text: str, doc_id: str, path: str, chunk_tokens: int) -> List[Dict]:
    """Simple fallback chunking if advanced chunking fails"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_tokens):
        chunk_words = words[i:i + chunk_tokens]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "id": f"{doc_id}::chunk::{i // chunk_tokens}",
            "text": chunk_text,
            "metadata": {
                "source": path,
                "doc_id": doc_id,
                "chunk_id": str(i // chunk_tokens),
                "chunking_method": "simple_fallback",
                "token_count": len(chunk_words),
            }
        })
    
    return chunks


# Standalone function for testing without ZenML
def chunk_documents_standalone(
    documents: List[Dict],
    chunk_tokens: int = 250,
    overlap_sentences: int = 2,
    chunking_strategy: str = "hybrid",
    similarity_threshold: float = 0.5,
    embedding_model_name: Optional[str] = None,
) -> List[Dict]:
    """
    Standalone version of chunk_documents_advanced that doesn't require ZenML.
    Same functionality, can be used for testing or non-ZenML pipelines.
    """
    return chunk_documents_advanced(
        documents=documents,
        chunk_tokens=chunk_tokens,
        overlap_sentences=overlap_sentences,
        chunking_strategy=chunking_strategy,
        similarity_threshold=similarity_threshold,
        embedding_model_name=embedding_model_name,
    )


# Configuration helper
def get_chunking_config_from_env() -> Dict:
    """Get chunking configuration from environment variables"""
    return {
        "chunk_tokens": int(os.getenv("CHUNK_TOKENS", "250")),
        "overlap_sentences": int(os.getenv("OVERLAP_SENTENCES", "2")),
        "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "hybrid"),
        "similarity_threshold": float(os.getenv("SIMILARITY_THRESHOLD", "0.5")),
        "embedding_model_name": os.getenv("EMBEDDING_MODEL"),
    }


if __name__ == "__main__":
    # Test the chunking functionality
    sample_doc = {
        "doc_id": "test_doc",
        "text": "This is a test document. It contains multiple sentences for testing. Each sentence should be processed correctly. The semantic chunker should group related sentences together. This creates more meaningful chunks for retrieval.",
        "path": "/test/path.txt"
    }
    
    config = get_chunking_config_from_env()
    chunks = chunk_documents_standalone([sample_doc], **config)
    
    for chunk in chunks:
        print(f"Chunk {chunk['metadata']['chunk_id']}:")
        print(f"  Text: {chunk['text'][:100]}...")
        print(f"  Tokens: {chunk['metadata'].get('token_count', 'N/A')}")
        print()