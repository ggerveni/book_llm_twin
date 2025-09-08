"""
Query processing module for query expansion, reformulation, and multi-query handling.
This module improves retrieval by generating alternative query formulations.
"""

import os
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import random
from collections import Counter


@dataclass
class QueryExpansion:
    """Container for expanded query information"""
    original_query: str
    expanded_queries: List[str]
    expansion_method: str
    confidence_scores: List[float]


class QueryProcessor:
    """
    Processes and expands user queries to improve retrieval coverage.
    Supports multiple expansion strategies.
    """
    
    def __init__(
        self,
        max_expansions: int = 3,
        min_query_length: int = 3,
        enable_synonyms: bool = True,
        enable_reformulation: bool = True,
        enable_question_variants: bool = True
    ):
        """
        Initialize query processor.
        
        Args:
            max_expansions: Maximum number of expanded queries to generate
            min_query_length: Minimum length for valid queries
            enable_synonyms: Enable synonym-based expansion
            enable_reformulation: Enable query reformulation
            enable_question_variants: Enable question variant generation
        """
        self.max_expansions = max_expansions
        self.min_query_length = min_query_length
        self.enable_synonyms = enable_synonyms
        self.enable_reformulation = enable_reformulation
        self.enable_question_variants = enable_question_variants
        
        # Load expansion dictionaries
        self.synonym_dict = self._load_synonym_dict()
        self.question_patterns = self._load_question_patterns()
        self.stopwords = self._load_stopwords()
    
    def expand_query(self, query: str) -> List[str]:
        """
        Generate expanded versions of the input query.
        
        Args:
            query: Original search query
            
        Returns:
            List of expanded query strings
        """
        if len(query.strip()) < self.min_query_length:
            return []
        
        expanded_queries = []
        
        # Generate different types of expansions
        if self.enable_synonyms:
            synonym_expansions = self._expand_with_synonyms(query)
            expanded_queries.extend(synonym_expansions)
        
        if self.enable_reformulation:
            reformulated = self._reformulate_query(query)
            expanded_queries.extend(reformulated)
        
        if self.enable_question_variants:
            question_variants = self._generate_question_variants(query)
            expanded_queries.extend(question_variants)
        
        # Remove duplicates and original query
        unique_expansions = []
        seen = {query.lower()}
        
        for expanded in expanded_queries:
            if expanded.lower() not in seen and len(expanded.strip()) >= self.min_query_length:
                unique_expansions.append(expanded)
                seen.add(expanded.lower())
        
        # Limit to max expansions
        return unique_expansions[:self.max_expansions]
    
    def expand_with_details(self, query: str) -> QueryExpansion:
        """
        Generate expanded queries with detailed information.
        
        Args:
            query: Original search query
            
        Returns:
            QueryExpansion object with detailed expansion information
        """
        expanded_queries = self.expand_query(query)
        
        # Calculate confidence scores (simplified)
        confidence_scores = [0.8] * len(expanded_queries)  # Could be more sophisticated
        
        return QueryExpansion(
            original_query=query,
            expanded_queries=expanded_queries,
            expansion_method="multi_strategy",
            confidence_scores=confidence_scores
        )
    
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query by replacing words with synonyms."""
        words = self._tokenize(query)
        expanded = []
        
        for i, word in enumerate(words):
            if word.lower() in self.synonym_dict:
                synonyms = self.synonym_dict[word.lower()]
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query by replacing words with synonyms."""
        words = self._tokenize(query)
        expanded = []
        
        for i, word in enumerate(words):
            if word.lower() in self.synonym_dict:
                synonyms = self.synonym_dict[word.lower()]
                for synonym in synonyms[:2]:  # Limit synonyms per word
                    new_words = words.copy()
                    new_words[i] = synonym
                    expanded_query = " ".join(new_words)
                    expanded.append(expanded_query)
        
        return expanded
    
    def _reformulate_query(self, query: str) -> List[str]:
        """Reformulate query using different phrasings."""
        reformulated = []
        
        # Add question words if not present
        if not self._is_question(query):
            question_words = ["what is", "how does", "why does", "when does", "where is"]
            for qword in question_words[:2]:
                reformulated.append(f"{qword} {query}")
        
        # Convert questions to statements
        if self._is_question(query):
            statement = self._question_to_statement(query)
            if statement:
                reformulated.append(statement)
        
        # Add contextual phrases
        contextual_phrases = [
            f"information about {query}",
            f"details on {query}",
            f"explanation of {query}"
        ]
        reformulated.extend(contextual_phrases[:1])
        
        return reformulated
    
    def _generate_question_variants(self, query: str) -> List[str]:
        """Generate different question variants of the query."""
        variants = []
        
        if self._is_question(query):
            # If already a question, try different question types
            variants.extend(self._alternate_question_forms(query))
        else:
            # If not a question, create question variants
            question_templates = [
                f"What is {query}?",
                f"How does {query} work?",
                f"Can you explain {query}?",
                f"Tell me about {query}"
            ]
            variants.extend(question_templates[:2])
        
        return variants
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization of text."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _is_question(self, query: str) -> bool:
        """Check if query is a question."""
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'is', 'are', 'do', 'does']
        first_word = query.strip().split()[0].lower() if query.strip() else ""
        return first_word in question_words or query.strip().endswith('?')
    
    def _question_to_statement(self, question: str) -> str:
        """Convert question to statement form."""
        question = question.strip().rstrip('?')
        
        # Simple conversion patterns
        if question.lower().startswith('what is'):
            return question[7:].strip()
        elif question.lower().startswith('how does'):
            return question[8:].strip()
        elif question.lower().startswith('why does'):
            return question[8:].strip()
        
        return question
    
    def _alternate_question_forms(self, question: str) -> List[str]:
        """Generate alternate forms of a question."""
        alternatives = []
        
        # Convert between different question types
        if question.lower().startswith('what'):
            alternatives.append(question.replace('what', 'which', 1))
        elif question.lower().startswith('how'):
            alternatives.append(question.replace('how', 'in what way', 1))
        
        return alternatives
    
    def _load_synonym_dict(self) -> Dict[str, List[str]]:
        """Load synonym dictionary for query expansion."""
        # Basic synonym dictionary - could be loaded from file or API
        return {
            'machine': ['computer', 'system', 'device'],
            'learning': ['training', 'education', 'studying'],
            'artificial': ['synthetic', 'simulated', 'automated'],
            'intelligence': ['smart', 'clever', 'cognitive'],
            'algorithm': ['method', 'procedure', 'process'],
            'data': ['information', 'facts', 'records'],
            'model': ['framework', 'structure', 'system'],
            'neural': ['brain-like', 'network', 'cognitive'],
            'deep': ['advanced', 'complex', 'sophisticated'],
            'natural': ['human', 'normal', 'organic'],
            'language': ['text', 'speech', 'communication'],
            'processing': ['analysis', 'computation', 'handling'],
            'network': ['system', 'structure', 'framework'],
            'training': ['learning', 'education', 'development'],
            'prediction': ['forecast', 'estimation', 'projection'],
            'classification': ['categorization', 'grouping', 'sorting'],
            'optimization': ['improvement', 'enhancement', 'refinement'],
            'recognition': ['identification', 'detection', 'perception'],
            'generation': ['creation', 'production', 'synthesis'],
            'understanding': ['comprehension', 'knowledge', 'insight']
        }
    
    def _load_question_patterns(self) -> List[str]:
        """Load question pattern templates."""
        return [
            "What is {topic}?",
            "How does {topic} work?",
            "Why is {topic} important?",
            "When is {topic} used?",
            "Where is {topic} applied?",
            "Can you explain {topic}?",
            "Tell me about {topic}",
            "What are the benefits of {topic}?",
            "How to use {topic}?",
            "What are examples of {topic}?"
        ]
    
    def _load_stopwords(self) -> Set[str]:
        """Load stopwords for query processing."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 
            'that', 'the', 'to', 'was', 'will', 'with', 'the', 'this',
            'but', 'they', 'have', 'had', 'what', 'said', 'each', 'which',
            'their', 'time', 'will', 'about', 'if', 'up', 'out', 'many',
            'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make',
            'like', 'into', 'him', 'has', 'two', 'more', 'very', 'what',
            'know', 'just', 'first', 'get', 'may', 'new', 'way', 'could'
        }


class MultiQueryProcessor:
    """
    Advanced query processor that handles complex multi-part questions
    and breaks them down into simpler sub-queries.
    """
    
    def __init__(self, base_processor: Optional[QueryProcessor] = None):
        self.base_processor = base_processor or QueryProcessor()
        self.conjunction_words = ['and', 'or', 'but', 'also', 'additionally', 'furthermore']
    
    def process_complex_query(self, query: str) -> List[str]:
        """
        Break down complex queries into simpler sub-queries.
        
        Args:
            query: Complex query potentially containing multiple questions
            
        Returns:
            List of simpler sub-queries
        """
        # Split on conjunctions and question marks
        sub_queries = self._split_complex_query(query)
        
        # Clean and validate sub-queries
        cleaned_queries = []
        for sub_query in sub_queries:
            cleaned = sub_query.strip()
            if len(cleaned) >= self.base_processor.min_query_length:
                cleaned_queries.append(cleaned)
        
        # Expand each sub-query if needed
        all_queries = []
        for sub_query in cleaned_queries:
            all_queries.append(sub_query)
            # Add expansions for each sub-query
            expansions = self.base_processor.expand_query(sub_query)
            all_queries.extend(expansions[:1])  # Limit expansions per sub-query
        
        return all_queries[:self.base_processor.max_expansions * 2]  # Overall limit
    
    def _split_complex_query(self, query: str) -> List[str]:
        """Split complex query into sub-queries."""
        # Split on multiple question marks
        parts = re.split(r'\?\s*', query)
        
        # Further split on conjunctions
        all_parts = []
        for part in parts:
            if any(conj in part.lower() for conj in self.conjunction_words):
                # Split on conjunctions
                sub_parts = re.split(r'\s+(?:and|or|but|also|additionally|furthermore)\s+', part, flags=re.IGNORECASE)
                all_parts.extend(sub_parts)
            else:
                all_parts.append(part)
        
        return [part.strip() for part in all_parts if part.strip()]


class SemanticQueryExpander:
    """
    Semantic query expansion using embedding similarity.
    Requires sentence-transformers for computing semantic similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            self.available = False
            print("Warning: sentence-transformers not available for semantic expansion")
    
    def expand_semantically(self, query: str, candidate_texts: List[str], top_k: int = 3) -> List[str]:
        """
        Expand query based on semantic similarity to candidate texts.
        
        Args:
            query: Original query
            candidate_texts: Potential expansion texts from document corpus
            top_k: Number of semantic expansions to return
            
        Returns:
            List of semantically similar expansion queries
        """
        if not self.available or not candidate_texts:
            return []
        
        # Encode query and candidates
        query_embedding = self.model.encode([query])
        candidate_embeddings = self.model.encode(candidate_texts)
        
        # Calculate similarities
        similarities = self.model.similarity(query_embedding, candidate_embeddings)[0]
        
        # Get top-k most similar texts
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        expansions = []
        for idx in top_indices:
            if similarities[idx] > 0.5:  # Similarity threshold
                expansions.append(candidate_texts[idx])
        
        return expansions


def create_query_processor_from_env() -> QueryProcessor:
    """Create QueryProcessor instance from environment variables."""
    return QueryProcessor(
        max_expansions=int(os.getenv("MAX_QUERY_EXPANSIONS", "3")),
        min_query_length=int(os.getenv("MIN_QUERY_LENGTH", "3")),
        enable_synonyms=os.getenv("ENABLE_SYNONYM_EXPANSION", "true").lower() == "true",
        enable_reformulation=os.getenv("ENABLE_QUERY_REFORMULATION", "true").lower() == "true",
        enable_question_variants=os.getenv("ENABLE_QUESTION_VARIANTS", "true").lower() == "true"
    )