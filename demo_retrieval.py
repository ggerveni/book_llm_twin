#!/usr/bin/env python3
"""
Demo script to test the enhanced retrieval functionality.
Run this to see how hybrid search, reranking, and query expansion work.
"""

import sys
import os
import time

# Add the current directory to Python path
sys.path.append(os.getcwd())

from rag.hybrid_retriever import HybridRetriever, create_hybrid_retriever_from_env
from rag.bm25_search import BM25Searcher, BM25Result
from rag.reranker import CrossEncoderReranker
from rag.query_processor import QueryProcessor
from rag.retriever import QdrantRetriever


# Sample document data for testing
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1::chunk::0",
        "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve their performance from experience without being explicitly programmed.",
        "metadata": {"source": "ai_intro.pdf", "doc_id": "doc1", "chunk_id": "0"}
    },
    {
        "id": "doc1::chunk::1", 
        "text": "Deep learning is a specialized subset of machine learning that uses artificial neural networks with multiple hidden layers to model and understand complex patterns.",
        "metadata": {"source": "ai_intro.pdf", "doc_id": "doc1", "chunk_id": "1"}
    },
    {
        "id": "doc2::chunk::0",
        "text": "Natural language processing enables computers to understand, interpret, and generate human language in a meaningful way through various computational techniques.",
        "metadata": {"source": "nlp_guide.pdf", "doc_id": "doc2", "chunk_id": "0"}
    },
    {
        "id": "doc2::chunk::1",
        "text": "Large language models like GPT and BERT have revolutionized natural language processing by achieving state-of-the-art results across many NLP tasks.",
        "metadata": {"source": "nlp_guide.pdf", "doc_id": "doc2", "chunk_id": "1"}
    },
    {
        "id": "doc3::chunk::0",
        "text": "Computer vision algorithms analyze and interpret visual information from images and videos to extract meaningful insights and make decisions.",
        "metadata": {"source": "cv_handbook.pdf", "doc_id": "doc3", "chunk_id": "0"}
    },
    {
        "id": "doc3::chunk::1",
        "text": "Convolutional neural networks are particularly effective for image recognition tasks because they can capture spatial relationships in visual data.",
        "metadata": {"source": "cv_handbook.pdf", "doc_id": "doc3", "chunk_id": "1"}
    }
]


def demo_bm25_search():
    """Demonstrate BM25 keyword search functionality"""
    print("=== BM25 KEYWORD SEARCH DEMO ===")
    print("Building BM25 index for demo documents...")
    
    # Create BM25 searcher
    searcher = BM25Searcher(index_path="demo/bm25_index")
    
    # Build index from sample documents
    documents_data = []
    for doc in SAMPLE_DOCUMENTS:
        doc_data = {
            'text': doc['text'],
            'chunk_id': doc['metadata']['chunk_id'],
            'metadata': doc['metadata']
        }
        documents_data.append(doc_data)
    
    searcher.build_index(documents_data)
    
    # Test queries
    test_queries = [
        "machine learning artificial intelligence",
        "neural networks deep learning", 
        "natural language processing",
        "computer vision images"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = searcher.search(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.3f} | {result.text[:60]}...")
    
    print("\n" + "="*60 + "\n")


def demo_query_expansion():
    """Demonstrate query expansion and reformulation"""
    print("=== QUERY EXPANSION DEMO ===")
    print("Testing query expansion with different strategies...")
    
    processor = QueryProcessor(max_expansions=3)
    
    test_queries = [
        "machine learning",
        "What is deep learning?",
        "How does neural network work",
        "computer vision applications"
    ]
    
    for query in test_queries:
        print(f"\nOriginal: '{query}'")
        expanded = processor.expand_query(query)
        
        for i, expansion in enumerate(expanded, 1):
            print(f"  {i}. {expansion}")
        
        if not expanded:
            print("  (No expansions generated)")
    
    print("\n" + "="*60 + "\n")


def demo_reranking():
    """Demonstrate cross-encoder reranking"""
    print("=== RERANKING DEMO ===")
    print("Loading reranking model (this may take a moment)...")
    
    try:
        reranker = CrossEncoderReranker()
        
        query = "machine learning algorithms"
        texts = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Computer vision processes visual information from images",
            "Natural language processing analyzes human language",
            "Algorithms are step-by-step procedures for calculations"
        ]
        
        print(f"Query: '{query}'")
        print("Texts to rerank:")
        for i, text in enumerate(texts, 1):
            print(f"  {i}. {text}")
        
        print("\nReranking scores:")
        scores = reranker.rerank(query, texts)
        
        # Sort by score
        ranked_results = list(zip(texts, scores))
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (text, score) in enumerate(ranked_results, 1):
            print(f"  {i}. Score: {score:.3f} | {text}")
    
    except Exception as e:
        print(f"Reranking demo failed: {e}")
        print("This might be due to missing dependencies or model download issues")
    
    print("\n" + "="*60 + "\n")


def demo_hybrid_retrieval():
    """Demonstrate hybrid retrieval combining vector + BM25 + reranking"""
    print("=== HYBRID RETRIEVAL DEMO ===")
    print("Note: This demo requires Qdrant running and indexed documents")
    
    try:
        # Check if we can create a hybrid retriever
        print("Attempting to create hybrid retriever...")
        
        # This will fail gracefully if Qdrant isn't running
        hybrid_retriever = HybridRetriever(
            collection_name="demo_collection",
            vector_weight=0.7,
            bm25_weight=0.3,
            enable_reranking=True,
            enable_query_expansion=True,
            final_top_k=3
        )
        
        test_queries = [
            "machine learning",
            "What is artificial intelligence?",
            "deep neural networks"
        ]
        
        for query in test_queries:
            print(f"\nHybrid search for: '{query}'")
            
            try:
                results = hybrid_retriever.retrieve(query)
                
                if results:
                    for i, chunk in enumerate(results, 1):
                        print(f"  {i}. Score: {chunk.score:.3f}")
                        print(f"     Text: {chunk.text[:80]}...")
                        print(f"     Source: {chunk.source}")
                else:
                    print("  No results found")
                    
            except Exception as e:
                print(f"  Search failed: {e}")
    
    except Exception as e:
        print(f"Hybrid retrieval demo failed: {e}")
        print("This is expected if Qdrant is not running or no documents are indexed")
        print("To test hybrid retrieval:")
        print("1. Start Qdrant: docker compose -f docker/docker-compose.yml up -d")
        print("2. Index documents: python scripts/ingest.py")
        print("3. Run this demo again")
    
    print("\n" + "="*60 + "\n")


def demo_retrieval_comparison():
    """Compare different retrieval strategies"""
    print("=== RETRIEVAL STRATEGY COMPARISON ===")
    
    # This demo shows conceptual differences
    strategies = {
        "Vector Search": "Uses semantic embeddings to find similar meaning",
        "BM25 Search": "Uses keyword matching with term frequency weighting", 
        "Hybrid Search": "Combines vector + BM25 for better coverage",
        "With Reranking": "Adds cross-encoder for improved relevance",
        "With Query Expansion": "Generates multiple query variants"
    }
    
    query = "machine learning algorithms"
    
    print(f"Query: '{query}'\n")
    
    for strategy, description in strategies.items():
        print(f"{strategy}:")
        print(f"  → {description}")
        
        if strategy == "Vector Search":
            print("  → Finds: 'deep learning', 'neural networks', 'AI models'")
        elif strategy == "BM25 Search":
            print("  → Finds: exact matches for 'machine', 'learning', 'algorithms'")
        elif strategy == "Hybrid Search":
            print("  → Combines both approaches for better recall and precision")
        elif strategy == "With Reranking":
            print("  → Re-scores results based on query-document relevance")
        elif strategy == "With Query Expansion":
            print("  → Also searches: 'ML algorithms', 'What are ML methods?'")
        
        print()
    
    print("Expected improvements:")
    print("• Vector only: Good semantic understanding, may miss exact terms")
    print("• BM25 only: Good keyword matching, may miss semantic similarity") 
    print("• Hybrid: Best of both worlds, ~20-40% improvement in relevance")
    print("• + Reranking: Further 10-20% improvement in result quality")
    print("• + Query expansion: Better coverage for complex queries")
    
    print("\n" + "="*60 + "\n")


def demo_performance_analysis():
    """Analyze performance characteristics of different approaches"""
    print("=== PERFORMANCE ANALYSIS ===")
    
    approaches = [
        ("Vector Search", "Fast", "Medium", "High quality for semantic queries"),
        ("BM25 Search", "Very Fast", "Low", "Good for keyword queries"),
        ("Hybrid Search", "Medium", "Medium", "Best overall relevance"),
        ("+ Reranking", "Slower", "High", "Highest quality results"),
        ("+ Query Expansion", "Slowest", "High", "Best coverage for complex queries")
    ]
    
    print(f"{'Approach':<20} {'Speed':<12} {'Memory':<8} {'Quality'}")
    print("-" * 60)
    
    for approach, speed, memory, quality in approaches:
        print(f"{approach:<20} {speed:<12} {memory:<8} {quality}")
    
    print("\nRecommendations:")
    print("• Development/Testing: Vector search only (fast iteration)")
    print("• Production (speed focus): Hybrid without reranking")
    print("• Production (quality focus): Full hybrid + reranking")
    print("• Complex queries: Add query expansion")
    
    print("\n" + "="*60 + "\n")


def main():
    """Run all retrieval demonstrations"""
    print("ENHANCED RETRIEVAL SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("Testing improved search capabilities for Book LLM Twin\n")
    
    try:
        # Run demos in order of complexity
        demo_bm25_search()
        demo_query_expansion() 
        demo_reranking()
        demo_retrieval_comparison()
        demo_performance_analysis()
        
        # Try hybrid demo last (most likely to fail without setup)
        demo_hybrid_retrieval()
        
        print("🎉 Retrieval demonstrations completed!")
        print("\nNext steps:")
        print("1. Start Qdrant: docker compose -f docker/docker-compose.yml up -d")
        print("2. Ingest documents with BM25: python scripts/ingest.py")  
        print("3. Update your app to use hybrid retrieval")
        print("4. Test with real queries!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Some components may require additional setup or dependencies")


if __name__ == "__main__":
    main()