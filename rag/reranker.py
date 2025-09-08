"""
Cross-encoder reranking module for improving retrieval relevance.
This module implements various reranking strategies using pre-trained cross-encoder models.
"""

import os
from typing import List, Tuple, Dict, Optional
import numpy as np
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
import torch


@dataclass 
class RerankedResult:
    """Container for reranked search result"""
    text: str
    original_score: float
    rerank_score: float
    combined_score: float
    rank_change: int  # How much the ranking changed (+/- positions)


class CrossEncoderReranker:
    """
    Cross-encoder based reranker that scores query-document pairs directly.
    More accurate than bi-encoder similarity but slower.
    """
    
    def __init__(
        self, 
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
        batch_size: int = 32,
        max_length: int = 512
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: HuggingFace cross-encoder model name
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        print(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name, max_length=max_length)
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    def rerank(
        self, 
        query: str, 
        texts: List[str], 
        original_scores: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[float]:
        """
        Rerank texts based on relevance to query.
        
        Args:
            query: Search query
            texts: List of texts to rerank
            original_scores: Original retrieval scores (optional)
            top_k: Return only top-k results (optional)
            
        Returns:
            List of reranking scores (0-1 range)
        """
        if not texts:
            return []
        
        # Prepare query-text pairs
        pairs = [[query, text] for text in texts]
        
        # Get reranking scores in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self.model.predict(batch_pairs)
            all_scores.extend(batch_scores.tolist())
        
        # Normalize scores to 0-1 range using sigmoid
        normalized_scores = [self._sigmoid(score) for score in all_scores]
        
        return normalized_scores[:top_k] if top_k else normalized_scores
    
    def rerank_with_details(
        self,
        query: str,
        texts: List[str],
        original_scores: Optional[List[float]] = None,
        combination_weight: float = 0.7
    ) -> List[RerankedResult]:
        """
        Rerank with detailed results including rank changes.
        
        Args:
            query: Search query
            texts: List of texts to rerank
            original_scores: Original retrieval scores
            combination_weight: Weight for combining original and rerank scores
            
        Returns:
            List of RerankedResult objects with detailed information
        """
        if not texts:
            return []
        
        # Use uniform scores if none provided
        if original_scores is None:
            original_scores = [1.0] * len(texts)
        
        # Get reranking scores
        rerank_scores = self.rerank(query, texts)
        
        # Create detailed results
        results = []
        for i, (text, orig_score, rerank_score) in enumerate(zip(texts, original_scores, rerank_scores)):
            # Combine original and rerank scores
            combined_score = (combination_weight * rerank_score + 
                            (1 - combination_weight) * orig_score)
            
            result = RerankedResult(
                text=text,
                original_score=orig_score,
                rerank_score=rerank_score,
                combined_score=combined_score,
                rank_change=0  # Will be calculated after sorting
            )
            results.append(result)
        
        # Sort by combined score and calculate rank changes
        original_order = list(range(len(results)))
        results_with_indices = list(zip(results, original_order))
        results_with_indices.sort(key=lambda x: x[0].combined_score, reverse=True)
        
        # Calculate rank changes
        reranked_results = []
        for new_rank, (result, old_rank) in enumerate(results_with_indices):
            result.rank_change = old_rank - new_rank  # Positive = moved up
            reranked_results.append(result)
        
        return reranked_results
    
    def _sigmoid(self, x: float) -> float:
        """Apply sigmoid function to normalize scores."""
        return 1 / (1 + np.exp(-x))
    
    def evaluate_reranking(
        self,
        query: str,
        texts: List[str],
        original_scores: List[float]
    ) -> Dict:
        """
        Evaluate reranking performance vs original ranking.
        
        Returns:
            Dictionary with reranking statistics
        """
        rerank_results = self.rerank_with_details(query, texts, original_scores)
        
        # Calculate statistics
        rank_changes = [abs(r.rank_change) for r in rerank_results]
        positive_changes = sum(1 for r in rerank_results if r.rank_change > 0)
        negative_changes = sum(1 for r in rerank_results if r.rank_change < 0)
        
        stats = {
            'total_items': len(rerank_results),
            'items_moved_up': positive_changes,
            'items_moved_down': negative_changes,
            'items_unchanged': len(rerank_results) - positive_changes - negative_changes,
            'avg_rank_change': np.mean(rank_changes) if rank_changes else 0,
            'max_rank_change': max(rank_changes) if rank_changes else 0,
            'rerank_score_range': {
                'min': min(r.rerank_score for r in rerank_results),
                'max': max(r.rerank_score for r in rerank_results),
                'mean': np.mean([r.rerank_score for r in rerank_results])
            }
        }
        
        return stats


class ListwiseReranker:
    """
    Alternative reranker that considers the entire list of documents
    for more sophisticated ranking decisions.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"):
        self.cross_encoder = CrossEncoderReranker(model_name)
    
    def rerank(
        self,
        query: str,
        texts: List[str],
        original_scores: Optional[List[float]] = None,
        diversity_weight: float = 0.1
    ) -> List[float]:
        """
        Listwise reranking with diversity consideration.
        
        Args:
            query: Search query
            texts: List of texts to rerank
            original_scores: Original retrieval scores
            diversity_weight: Weight for promoting diversity
            
        Returns:
            Reranked scores
        """
        if len(texts) <= 1:
            return self.cross_encoder.rerank(query, texts, original_scores)
        
        # Get base reranking scores
        base_scores = self.cross_encoder.rerank(query, texts, original_scores)
        
        # Calculate diversity penalties
        diversity_penalties = self._calculate_diversity_penalties(texts)
        
        # Combine relevance and diversity
        final_scores = []
        for base_score, diversity_penalty in zip(base_scores, diversity_penalties):
            final_score = base_score - (diversity_weight * diversity_penalty)
            final_scores.append(max(0.0, final_score))  # Ensure non-negative
        
        return final_scores
    
    def _calculate_diversity_penalties(self, texts: List[str]) -> List[float]:
        """
        Calculate diversity penalties based on text similarity.
        Higher penalty for texts similar to others.
        """
        from sentence_transformers import SentenceTransformer
        
        # Use a light model for diversity calculation
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        
        penalties = []
        for i, emb_i in enumerate(embeddings):
            # Calculate similarity to all other texts
            similarities = []
            for j, emb_j in enumerate(embeddings):
                if i != j:
                    sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                    similarities.append(max(0, sim))  # Only positive similarities
            
            # Penalty is average similarity to others
            penalty = np.mean(similarities) if similarities else 0.0
            penalties.append(penalty)
        
        return penalties


class EnsembleReranker:
    """
    Ensemble reranker that combines multiple reranking models.
    """
    
    def __init__(self, model_names: List[str], weights: Optional[List[float]] = None):
        """
        Initialize ensemble reranker.
        
        Args:
            model_names: List of cross-encoder model names
            weights: Weights for combining models (default: equal weights)
        """
        self.rerankers = [CrossEncoderReranker(name) for name in model_names]
        self.weights = weights or [1.0 / len(model_names)] * len(model_names)
        
        if len(self.weights) != len(self.rerankers):
            raise ValueError("Number of weights must match number of models")
    
    def rerank(self, query: str, texts: List[str]) -> List[float]:
        """Ensemble reranking using weighted combination of models."""
        if not texts:
            return []
        
        # Get scores from all models
        all_scores = []
        for reranker in self.rerankers:
            scores = reranker.rerank(query, texts)
            all_scores.append(scores)
        
        # Weighted combination
        ensemble_scores = []
        for i in range(len(texts)):
            weighted_score = sum(
                weight * scores[i] 
                for weight, scores in zip(self.weights, all_scores)
            )
            ensemble_scores.append(weighted_score)
        
        return ensemble_scores


def create_reranker_from_env() -> CrossEncoderReranker:
    """Create reranker instance from environment variables."""
    model_name = os.getenv("RERANKING_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    batch_size = int(os.getenv("RERANK_BATCH_SIZE", "32"))
    max_length = int(os.getenv("RERANK_MAX_LENGTH", "512"))
    
    return CrossEncoderReranker(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length
    )