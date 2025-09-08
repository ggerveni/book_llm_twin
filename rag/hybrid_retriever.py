"""
Enhanced Hybrid Retriever combining vector search and BM25 keyword search.
This module implements hybrid retrieval with configurable weighting and reranking.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv

from .retriever import QdrantRetriever, RetrievedChunk
from .bm25_search import BM25Searcher
from .reranker import CrossEncoderReranker
from .query_processor import QueryProcessor

load_dotenv()


@dataclass
class HybridResult:
    """Result from hybrid search with scoring details"""
    chunk: RetrievedChunk
    vector_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: Optional[float] = None
    final_score: float = None


class HybridRetriever:
    """
    Enhanced retriever that combines vector search, BM25 keyword search,
    query expansion, and reranking for improved retrieval accuracy.
    """
    
    def __init__(
        self,
        collection_name: str,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        bm25_index_path: Optional[str] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        enable_reranking: bool = True,
        enable_query_expansion: bool = True,
        rerank_top_k: int = 20,
        final_top_k: int = 5,
    ):
        """
        Initialize hybrid retriever with configurable components.
        
        Args:
            collection_name: Qdrant collection name
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            embedding_model_name: Sentence transformer model name
            bm25_index_path: Path to BM25 index files
            vector_weight: Weight for vector search scores (0.0-1.0)
            bm25_weight: Weight for BM25 scores (0.0-1.0)
            enable_reranking: Whether to use cross-encoder reranking
            enable_query_expansion: Whether to expand queries
            rerank_top_k: Number of results to rerank
            final_top_k: Final number of results to return
        """
        self.collection_name = collection_name
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_reranking = enable_reranking
        self.enable_query_expansion = enable_query_expansion
        self.rerank_top_k = rerank_top_k
        self.final_top_k = final_top_k
        
        # Initialize vector retriever
        self.vector_retriever = QdrantRetriever(
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            embedding_model_name=embedding_model_name,
            top_k=rerank_top_k  # Get more for reranking
        )
        
        # Initialize BM25 searcher
        bm25_path = bm25_index_path or os.getenv("BM25_INDEX_PATH", f"data/bm25_{collection_name}")
        self.bm25_searcher = BM25Searcher(index_path=bm25_path)
        
        # Initialize reranker if enabled
        self.reranker = None
        if enable_reranking:
            rerank_model = os.getenv("RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
            self.reranker = CrossEncoderReranker(model_name=rerank_model)
        
        # Initialize query processor if enabled
        self.query_processor = None
        if enable_query_expansion:
            self.query_processor = QueryProcessor()
    
    def retrieve(
        self, 
        query: str, 
        score_threshold: Optional[float] = None,
        alpha: Optional[float] = None,
        expand_query: Optional[bool] = None,
        use_reranking: Optional[bool] = None
    ) -> List[RetrievedChunk]:
        """
        Perform hybrid retrieval with optional query expansion and reranking.
        
        Args:
            query: Search query
            score_threshold: Minimum score threshold for results
            alpha: Vector weight override (None to use default)
            expand_query: Override query expansion setting
            use_reranking: Override reranking setting
            
        Returns:
            List of retrieved chunks sorted by relevance
        """
        print(f"🔍 Starting hybrid retrieval for: '{query[:50]}...'")
        
        # Use provided settings or defaults
        use_expansion = expand_query if expand_query is not None else self.enable_query_expansion
        use_rerank = use_reranking if use_reranking is not None else self.enable_reranking
        vector_weight = alpha if alpha is not None else self.vector_weight
        bm25_weight = 1.0 - vector_weight
        
        # Process and expand query if enabled
        queries = [query]
        if use_expansion and self.query_processor:
            print("📝 Expanding query...")
            expanded_queries = self.query_processor.expand_query(query)
            queries.extend(expanded_queries)
            print(f"   Generated {len(expanded_queries)} additional queries")
        
        # Collect results from all queries
        all_hybrid_results = []
        for i, q in enumerate(queries):
            if i > 0:
                print(f"   Processing expanded query {i}: '{q[:40]}...'")
            
            # Get vector search results
            vector_results = self.vector_retriever.retrieve(q, score_threshold=None)
            print(f"   Vector search: {len(vector_results)} results")
            
            # Get BM25 search results
            bm25_results = self.bm25_searcher.search(q, top_k=self.rerank_top_k)
            print(f"   BM25 search: {len(bm25_results)} results")
            
            # Combine and score results
            hybrid_results = self._combine_search_results(
                vector_results, bm25_results, vector_weight, bm25_weight
            )
            all_hybrid_results.extend(hybrid_results)
        
        # Deduplicate results by chunk_id
        unique_results = self._deduplicate_results(all_hybrid_results)
        print(f"🔗 Combined results: {len(unique_results)} unique chunks")
        
        # Apply score threshold if provided
        if score_threshold:
            unique_results = [r for r in unique_results if r.hybrid_score >= score_threshold]
            print(f"🎯 After threshold: {len(unique_results)} chunks")
        
        # Sort by hybrid score and take top candidates for reranking
        unique_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        top_candidates = unique_results[:self.rerank_top_k]
        
        # Apply reranking if enabled
        final_results = top_candidates
        if use_rerank and self.reranker and len(top_candidates) > 1:
            print(f"🏆 Reranking top {len(top_candidates)} candidates...")
            final_results = self._rerank_results(query, top_candidates)
        
        # Return final top-k results
        final_chunks = [result.chunk for result in final_results[:self.final_top_k]]
        print(f"✅ Returning {len(final_chunks)} final results")
        
        return final_chunks
    
    def _combine_search_results(
        self,
        vector_results: List[RetrievedChunk],
        bm25_results: List[Dict],
        vector_weight: float,
        bm25_weight: float
    ) -> List[HybridResult]:
        """Combine vector and BM25 search results with weighted scoring."""
        
        # Create lookup dictionaries
        vector_lookup = {chunk.chunk_id: chunk for chunk in vector_results}
        bm25_lookup = {result['chunk_id']: result for result in bm25_results}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(vector_lookup.keys()) | set(bm25_lookup.keys())
        
        hybrid_results = []
        
        # Normalize scores to 0-1 range
        vector_scores = [chunk.score for chunk in vector_results]
        bm25_scores = [result['score'] for result in bm25_results]
        
        vector_max = max(vector_scores) if vector_scores else 1.0
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        
        for chunk_id in all_chunk_ids:
            # Get normalized scores
            vector_score = 0.0
            bm25_score = 0.0
            
            if chunk_id in vector_lookup:
                vector_score = vector_lookup[chunk_id].score / vector_max
            
            if chunk_id in bm25_lookup:
                bm25_score = bm25_lookup[chunk_id]['score'] / bm25_max
            
            # Calculate hybrid score
            hybrid_score = (vector_weight * vector_score) + (bm25_weight * bm25_score)
            
            # Use chunk from vector results if available, otherwise create from BM25
            chunk = vector_lookup.get(chunk_id)
            if not chunk and chunk_id in bm25_lookup:
                bm25_result = bm25_lookup[chunk_id]
                chunk = RetrievedChunk(
                    text=bm25_result['text'],
                    score=bm25_score,
                    source=bm25_result.get('source', 'Unknown'),
                    chunk_id=chunk_id,
                    doc_id=bm25_result.get('doc_id')
                )
            
            if chunk:
                hybrid_result = HybridResult(
                    chunk=chunk,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    hybrid_score=hybrid_score,
                    final_score=hybrid_score
                )
                hybrid_results.append(hybrid_result)
        
        return hybrid_results
    
    def _deduplicate_results(self, results: List[HybridResult]) -> List[HybridResult]:
        """Remove duplicate chunks, keeping the one with highest score."""
        seen_chunks = {}
        
        for result in results:
            chunk_id = result.chunk.chunk_id
            if chunk_id not in seen_chunks or result.hybrid_score > seen_chunks[chunk_id].hybrid_score:
                seen_chunks[chunk_id] = result
        
        return list(seen_chunks.values())
    
    def _rerank_results(self, query: str, results: List[HybridResult]) -> List[HybridResult]:
        """Apply cross-encoder reranking to results."""
        if not self.reranker:
            return results
        
        # Prepare texts for reranking
        texts = [result.chunk.text for result in results]
        
        # Get reranking scores
        rerank_scores = self.reranker.rerank(query, texts)
        
        # Update results with rerank scores
        for result, rerank_score in zip(results, rerank_scores):
            result.rerank_score = rerank_score
            # Combine hybrid and rerank scores (can be tuned)
            result.final_score = 0.7 * result.hybrid_score + 0.3 * rerank_score
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about the retrieval components."""
        stats = {
            'vector_retriever': {
                'model': self.vector_retriever.embedding_model_name,
                'collection': self.collection_name
            },
            'bm25_searcher': {
                'index_path': self.bm25_searcher.index_path,
                'vocabulary_size': getattr(self.bm25_searcher, 'vocab_size', 'Unknown')
            },
            'settings': {
                'vector_weight': self.vector_weight,
                'bm25_weight': self.bm25_weight,
                'reranking_enabled': self.enable_reranking,
                'query_expansion_enabled': self.enable_query_expansion,
                'rerank_top_k': self.rerank_top_k,
                'final_top_k': self.final_top_k
            }
        }
        
        if self.reranker:
            stats['reranker'] = {
                'model': self.reranker.model_name
            }
        
        return stats


def create_hybrid_retriever_from_env() -> HybridRetriever:
    """Create a HybridRetriever instance using environment variables."""
    return HybridRetriever(
        collection_name=os.getenv("QDRANT_COLLECTION", "book"),
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL"),
        bm25_index_path=os.getenv("BM25_INDEX_PATH"),
        vector_weight=float(os.getenv("VECTOR_WEIGHT", "0.7")),
        bm25_weight=float(os.getenv("BM25_WEIGHT", "0.3")),
        enable_reranking=os.getenv("ENABLE_RERANKING", "true").lower() == "true",
        enable_query_expansion=os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true",
        rerank_top_k=int(os.getenv("RERANK_TOP_K", "20")),
        final_top_k=int(os.getenv("TOP_K", "5"))
    )