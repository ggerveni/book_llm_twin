"""
BM25 keyword search implementation for hybrid retrieval.
This module provides fast keyword-based search to complement vector search.
"""

import os
import pickle
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import re
from collections import Counter
import math


@dataclass
class BM25Result:
    """Result from BM25 search"""
    chunk_id: str
    text: str
    score: float
    doc_id: Optional[str] = None
    source: Optional[str] = None


class BM25Searcher:
    """
    BM25 (Best Matching 25) searcher for keyword-based document retrieval.
    Implements the BM25 ranking function used by search engines.
    """
    
    def __init__(
        self,
        index_path: str,
        k1: float = 1.2,
        b: float = 0.75,
        epsilon: float = 0.25
    ):
        """
        Initialize BM25 searcher.
        
        Args:
            index_path: Path to BM25 index files
            k1: Controls term frequency normalization (1.2-2.0)
            b: Controls length normalization (0.0-1.0)
            epsilon: Minimum score for matches
        """
        self.index_path = index_path
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        
        # Index components
        self.documents = []  # List of document texts
        self.doc_metadata = []  # Metadata for each document
        self.term_frequencies = []  # TF for each document
        self.document_frequencies = {}  # DF for each term
        self.idf_cache = {}  # Cached IDF values
        self.avg_doc_length = 0
        self.vocab_size = 0
        
        # Load existing index or initialize empty
        self._load_index()
    
    def build_index(self, documents_data: List[Dict]) -> None:
        """
        Build BM25 index from document data.
        
        Args:
            documents_data: List of dicts with 'text', 'chunk_id', 'metadata' keys
        """
        print(f"Building BM25 index with {len(documents_data)} documents...")
        
        self.documents = []
        self.doc_metadata = []
        self.term_frequencies = []
        self.document_frequencies = {}
        
        total_length = 0
        
        for doc_data in documents_data:
            text = doc_data.get('text', '')
            metadata = doc_data.get('metadata', {})
            
            # Tokenize and count terms
            tokens = self._tokenize(text)
            term_freq = Counter(tokens)
            
            self.documents.append(text)
            self.doc_metadata.append(metadata)
            self.term_frequencies.append(term_freq)
            
            total_length += len(tokens)
            
            # Update document frequencies
            for term in set(tokens):
                self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1
        
        # Calculate average document length
        self.avg_doc_length = total_length / len(documents_data) if documents_data else 0
        self.vocab_size = len(self.document_frequencies)
        
        # Clear IDF cache
        self.idf_cache = {}
        
        print(f"Index built: {len(self.documents)} docs, {self.vocab_size} terms, avg length: {self.avg_doc_length:.1f}")
        
        # Save index
        self._save_index()
    
    def search(self, query: str, top_k: int = 10) -> List[BM25Result]:
        """
        Search documents using BM25 scoring.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of BM25Result objects sorted by score
        """
        if not self.documents:
            return []
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # Calculate BM25 scores for all documents
        scores = []
        for i in range(len(self.documents)):
            score = self._calculate_bm25_score(query_terms, i)
            if score > self.epsilon:
                result = BM25Result(
                    chunk_id=self.doc_metadata[i].get('chunk_id', str(i)),
                    text=self.documents[i],
                    score=score,
                    doc_id=self.doc_metadata[i].get('doc_id'),
                    source=self.doc_metadata[i].get('source')
                )
                scores.append(result)
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:top_k]
    
    def _calculate_bm25_score(self, query_terms: List[str], doc_index: int) -> float:
        """Calculate BM25 score for a document given query terms."""
        score = 0.0
        doc_length = sum(self.term_frequencies[doc_index].values())
        
        for term in query_terms:
            if term in self.term_frequencies[doc_index]:
                # Term frequency in document
                tf = self.term_frequencies[doc_index][term]
                
                # Inverse document frequency
                idf = self._get_idf(term)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def _get_idf(self, term: str) -> float:
        """Get IDF (Inverse Document Frequency) for a term."""
        if term not in self.idf_cache:
            df = self.document_frequencies.get(term, 0)
            if df == 0:
                idf = 0.0
            else:
                # Standard IDF formula with smoothing
                idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5))
            self.idf_cache[term] = max(idf, 0.0)  # Ensure non-negative
        
        return self.idf_cache[term]
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of normalized tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        
        # Remove short tokens and common stopwords
        stopwords = self._get_stopwords()
        tokens = [token for token in tokens if len(token) > 2 and token not in stopwords]
        
        return tokens
    
    def _get_stopwords(self) -> set:
        """Get stopwords for filtering."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this', 'that',
            'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'this', 'that',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should'
        }
    
    def _save_index(self) -> None:
        """Save BM25 index to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        
        index_data = {
            'documents': self.documents,
            'doc_metadata': self.doc_metadata,
            'term_frequencies': [dict(tf) for tf in self.term_frequencies],  # Convert Counter to dict
            'document_frequencies': self.document_frequencies,
            'avg_doc_length': self.avg_doc_length,
            'vocab_size': self.vocab_size,
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            }
        }
        
        # Save as pickle for efficiency
        index_file = os.path.join(self.index_path, 'bm25_index.pkl')
        with open(index_file, 'wb') as f:
            pickle.dump(index_data, f)
        
        # Also save metadata as JSON for inspection
        metadata_file = os.path.join(self.index_path, 'index_metadata.json')
        metadata = {
            'num_documents': len(self.documents),
            'vocab_size': self.vocab_size,
            'avg_doc_length': self.avg_doc_length,
            'parameters': index_data['parameters']
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"BM25 index saved to {self.index_path}")
    
    def _load_index(self) -> None:
        """Load BM25 index from disk."""
        index_file = os.path.join(self.index_path, 'bm25_index.pkl')
        
        if os.path.exists(index_file):
            try:
                with open(index_file, 'rb') as f:
                    index_data = pickle.load(f)
                
                self.documents = index_data['documents']
                self.doc_metadata = index_data['doc_metadata']
                self.term_frequencies = [Counter(tf) for tf in index_data['term_frequencies']]
                self.document_frequencies = index_data['document_frequencies']
                self.avg_doc_length = index_data['avg_doc_length']
                self.vocab_size = index_data['vocab_size']
                
                # Load parameters
                params = index_data.get('parameters', {})
                self.k1 = params.get('k1', self.k1)
                self.b = params.get('b', self.b)
                self.epsilon = params.get('epsilon', self.epsilon)
                
                print(f"BM25 index loaded: {len(self.documents)} docs, {self.vocab_size} terms")
                
            except Exception as e:
                print(f"Error loading BM25 index: {e}")
                self._initialize_empty_index()
        else:
            print(f"No existing BM25 index found at {self.index_path}")
            self._initialize_empty_index()
    
    def _initialize_empty_index(self) -> None:
        """Initialize empty index structures."""
        self.documents = []
        self.doc_metadata = []
        self.term_frequencies = []
        self.document_frequencies = {}
        self.idf_cache = {}
        self.avg_doc_length = 0
        self.vocab_size = 0
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the BM25 index."""
        if not self.documents:
            return {'status': 'empty'}
        
        return {
            'status': 'loaded',
            'num_documents': len(self.documents),
            'vocabulary_size': self.vocab_size,
            'avg_document_length': self.avg_doc_length,
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'epsilon': self.epsilon
            },
            'index_path': self.index_path
        }
    
    def add_documents(self, documents_data: List[Dict]) -> None:
        """
        Add new documents to existing index (incremental indexing).
        
        Args:
            documents_data: List of new documents to add
        """
        if not documents_data:
            return
        
        print(f"Adding {len(documents_data)} documents to BM25 index...")
        
        # Store current state
        old_doc_count = len(self.documents)
        old_total_length = self.avg_doc_length * old_doc_count
        
        # Add new documents
        for doc_data in documents_data:
            text = doc_data.get('text', '')
            metadata = doc_data.get('metadata', {})
            
            tokens = self._tokenize(text)
            term_freq = Counter(tokens)
            
            self.documents.append(text)
            self.doc_metadata.append(metadata)
            self.term_frequencies.append(term_freq)
            
            # Update document frequencies
            for term in set(tokens):
                self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1
        
        # Update statistics
        new_total_length = sum(sum(tf.values()) for tf in self.term_frequencies)
        self.avg_doc_length = new_total_length / len(self.documents)
        self.vocab_size = len(self.document_frequencies)
        
        # Clear IDF cache (needs recalculation)
        self.idf_cache = {}
        
        print(f"Index updated: {len(self.documents)} total docs (+{len(documents_data)})")
        
        # Save updated index
        self._save_index()


def create_bm25_searcher_from_env() -> BM25Searcher:
    """Create BM25Searcher instance from environment variables."""
    collection_name = os.getenv("QDRANT_COLLECTION", "book")
    index_path = os.getenv("BM25_INDEX_PATH", f"data/bm25_{collection_name}")
    
    return BM25Searcher(
        index_path=index_path,
        k1=float(os.getenv("BM25_K1", "1.2")),
        b=float(os.getenv("BM25_B", "0.75")),
        epsilon=float(os.getenv("BM25_EPSILON", "0.25"))
    )