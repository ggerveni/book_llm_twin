"""
Enhanced embedding step that prepares documents for both vector and BM25 search.
This step creates embeddings while preserving text for keyword indexing.
"""

from typing import List, Dict, Optional
import uuid
import os
from sentence_transformers import SentenceTransformer

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
def embed_documents(
    chunks: List[Dict], 
    embedding_model_name: str,
    batch_size: Optional[int] = None,
    normalize_embeddings: bool = True,
    add_text_preprocessing: bool = True
) -> List[Dict]:
    """
    Enhanced embedding step that creates embeddings while preserving text for hybrid search.
    
    Args:
        chunks: List of chunk dictionaries from chunking step
        embedding_model_name: Sentence transformer model name
        batch_size: Batch size for embedding (None = auto)
        normalize_embeddings: Whether to normalize embeddings to unit length
        add_text_preprocessing: Whether to add preprocessed text for BM25
        
    Returns:
        List of point dictionaries ready for both vector and keyword indexing
    """
    
    if not chunks:
        print("No chunks provided for embedding")
        return []
    
    print(f"Embedding {len(chunks)} chunks with {embedding_model_name}")
    
    # Load embedding model
    model = SentenceTransformer(embedding_model_name)
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Create embeddings with progress bar
    if batch_size:
        vectors = model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings
        ).tolist()
    else:
        vectors = model.encode(
            texts, 
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings
        ).tolist()
    
    print(f"Generated {len(vectors)} embeddings of dimension {len(vectors[0])}")
    
    # Create points for indexing
    points: List[Dict] = []
    for chunk, vector in zip(chunks, vectors):
        
        # Generate consistent point ID
        raw_id = str(chunk["id"])  # e.g., "doc::chunk::idx"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
        
        # Prepare payload with enhanced metadata
        payload = {
            "text": chunk["text"], 
            "metadata": chunk.get("metadata", {}).copy()
        }
        
        # Add original ID for reference
        payload["metadata"]["original_id"] = raw_id
        
        # Add embedding metadata
        payload["metadata"]["embedding_model"] = embedding_model_name
        payload["metadata"]["vector_dimension"] = len(vector)
        
        # Add preprocessed text for BM25 if enabled
        if add_text_preprocessing:
            preprocessed_text = _preprocess_text_for_bm25(chunk["text"])
            payload["metadata"]["preprocessed_text"] = preprocessed_text
        
        # Add retrieval metadata for hybrid search
        payload["metadata"]["supports_vector_search"] = True
        payload["metadata"]["supports_keyword_search"] = True
        
        point = {
            "id": point_id,
            "vector": vector,
            "payload": payload,
        }
        
        points.append(point)
    
    print(f"Created {len(points)} vector points for indexing")
    return points


def _preprocess_text_for_bm25(text: str) -> str:
    """
    Preprocess text for better BM25 keyword matching.
    
    Args:
        text: Original text
        
    Returns:
        Preprocessed text optimized for keyword search
    """
    
    # Basic preprocessing for BM25
    processed = text.lower()
    
    # Remove excessive whitespace
    processed = " ".join(processed.split())
    
    # Could add more sophisticated preprocessing here:
    # - Stemming with nltk/spacy
    # - Lemmatization 
    # - Custom term expansion
    # - Domain-specific preprocessing
    
    return processed


def embed_documents_with_multiple_models(
    chunks: List[Dict],
    embedding_models: List[str],
    model_weights: Optional[List[float]] = None
) -> List[Dict]:
    """
    Create embeddings using multiple models for ensemble retrieval.
    
    Args:
        chunks: List of chunk dictionaries
        embedding_models: List of model names to use
        model_weights: Weights for combining embeddings (optional)
        
    Returns:
        List of points with ensemble embeddings
    """
    
    if not model_weights:
        model_weights = [1.0 / len(embedding_models)] * len(embedding_models)
    
    if len(model_weights) != len(embedding_models):
        raise ValueError("Number of weights must match number of models")
    
    print(f"Creating ensemble embeddings with {len(embedding_models)} models")
    
    # Get embeddings from each model
    all_embeddings = []
    for model_name in embedding_models:
        print(f"Processing with {model_name}...")
        model = SentenceTransformer(model_name)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        all_embeddings.append(embeddings)
    
    # Combine embeddings using weighted average
    import numpy as np
    ensemble_embeddings = np.zeros_like(all_embeddings[0])
    
    for embeddings, weight in zip(all_embeddings, model_weights):
        ensemble_embeddings += weight * embeddings
    
    # Normalize combined embeddings
    norms = np.linalg.norm(ensemble_embeddings, axis=1, keepdims=True)
    ensemble_embeddings = ensemble_embeddings / norms
    
    # Create points
    points = []
    for chunk, vector in zip(chunks, ensemble_embeddings.tolist()):
        raw_id = str(chunk["id"])
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
        
        payload = {
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}).copy()
        }
        payload["metadata"]["original_id"] = raw_id
        payload["metadata"]["embedding_method"] = "ensemble"
        payload["metadata"]["ensemble_models"] = embedding_models
        payload["metadata"]["model_weights"] = model_weights
        
        points.append({
            "id": point_id,
            "vector": vector,
            "payload": payload
        })
    
    return points


# Standalone function for testing without ZenML
def embed_documents_standalone(
    chunks: List[Dict],
    embedding_model_name: str,
    batch_size: Optional[int] = None,
    normalize_embeddings: bool = True,
    add_text_preprocessing: bool = True
) -> List[Dict]:
    """
    Standalone version of embed_documents that doesn't require ZenML.
    Same functionality, can be used for testing or non-ZenML pipelines.
    """
    return embed_documents(
        chunks=chunks,
        embedding_model_name=embedding_model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        add_text_preprocessing=add_text_preprocessing
    )


def get_embedding_config_from_env() -> Dict:
    """Get embedding configuration from environment variables."""
    return {
        'embedding_model_name': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        'batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', '32')) if os.getenv('EMBEDDING_BATCH_SIZE') else None,
        'normalize_embeddings': os.getenv('NORMALIZE_EMBEDDINGS', 'true').lower() == 'true',
        'add_text_preprocessing': os.getenv('ADD_TEXT_PREPROCESSING', 'true').lower() == 'true'
    }


if __name__ == "__main__":
    # Test the enhanced embedding functionality
    
    # Sample chunks for testing
    sample_chunks = [
        {
            "id": "test_doc::chunk::0",
            "text": "Machine learning is a subset of artificial intelligence that enables computers to learn.",
            "metadata": {
                "source": "test.txt",
                "doc_id": "test_doc",
                "chunk_id": "0",
                "token_count": 15
            }
        },
        {
            "id": "test_doc::chunk::1",
            "text": "Deep learning uses neural networks with multiple layers to understand complex patterns.",
            "metadata": {
                "source": "test.txt", 
                "doc_id": "test_doc",
                "chunk_id": "1",
                "token_count": 13
            }
        }
    ]
    
    # Test embedding
    config = get_embedding_config_from_env()
    points = embed_documents_standalone(
        chunks=sample_chunks,
        **config
    )
    
    print(f"Created {len(points)} embedding points")
    
    # Display results
    for point in points:
        print(f"Point ID: {point['id']}")
        print(f"Vector dimension: {len(point['vector'])}")
        print(f"Text: {point['payload']['text'][:50]}...")
        print(f"Metadata: {point['payload']['metadata']}")
        print()