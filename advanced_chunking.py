"""
Advanced Semantic Chunking Module

This module provides semantic chunking capabilities that:
1. Preserves sentence boundaries
2. Uses semantic similarity for chunk boundaries
3. Implements smart overlap handling
4. Supports multiple chunking strategies
"""

import re
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    SENTENCE_BASED = "sentence_based"
    SEMANTIC_SIMILARITY = "semantic_similarity" 
    HYBRID = "hybrid"
    TOKEN_AWARE = "token_aware"


@dataclass
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    start_sentence: int
    end_sentence: int
    token_count: int
    semantic_score: Optional[float] = None
    overlap_with_previous: Optional[int] = None
    overlap_with_next: Optional[int] = None


@dataclass
class DocumentChunk:
    """Enhanced chunk with metadata"""
    text: str
    metadata: ChunkMetadata
    source_metadata: Dict = None
    

class SentenceSplitter:
    """Advanced sentence splitting with support for various document types"""
    
    def __init__(self):
        # Enhanced sentence boundary patterns
        self.sentence_endings = re.compile(
            r'(?<=[.!?])\s+(?=[A-Z])|'  # Standard sentence endings
            r'(?<=\w)\n\n(?=\w)|'       # Paragraph breaks
            r'(?<=:)\s*\n(?=\s*[A-Z•\-\d])|'  # List items after colons
            r'(?<=\.)\s*\n(?=\d+\.)|'   # Numbered lists
            r'(?<=\.)\s*\n(?=[A-Z][a-z]+:)'  # Section headers
        )
        
        # Patterns that should NOT be sentence boundaries
        self.false_boundaries = re.compile(
            r'(?:Mr|Ms|Mrs|Dr|Prof|Inc|Ltd|Co|vs|etc|Jr|Sr|Ph\.D|M\.D|B\.A|M\.A)\.\s+',
            re.IGNORECASE
        )
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved boundary detection"""
        # Handle false boundaries first
        text = self._handle_abbreviations(text)
        
        # Split into sentences
        sentences = self.sentence_endings.split(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
                
        return cleaned_sentences
    
    def _handle_abbreviations(self, text: str) -> str:
        """Replace abbreviations with temporary placeholders"""
        # This prevents splitting on abbreviations
        abbreviations = {
            'Mr.': 'MR_TEMP',
            'Ms.': 'MS_TEMP', 
            'Mrs.': 'MRS_TEMP',
            'Dr.': 'DR_TEMP',
            'Prof.': 'PROF_TEMP',
            'Inc.': 'INC_TEMP',
            'Ltd.': 'LTD_TEMP',
            'etc.': 'ETC_TEMP',
        }
        
        result = text
        for abbrev, placeholder in abbreviations.items():
            result = result.replace(abbrev, placeholder)
            
        return result


class SemanticChunker:
    """Advanced semantic chunking with multiple strategies"""
    
    def __init__(
        self, 
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_chunk_tokens: int = 250,
        overlap_sentences: int = 2
    ):
        print(f"🤖 Initializing SemanticChunker with model: {embedding_model_name}")
        print("📥 Loading embedding model (downloading if first time)...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print("✅ Embedding model loaded successfully!")
        
        self.similarity_threshold = similarity_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_sentences = overlap_sentences
        self.sentence_splitter = SentenceSplitter()
        
    def chunk_document(
        self, 
        text: str, 
        strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
        source_metadata: Dict = None
    ) -> List[DocumentChunk]:
        """Main chunking method with strategy selection"""
        
        sentences = self.sentence_splitter.split_into_sentences(text)
        if not sentences:
            return []
            
        if strategy == ChunkingStrategy.SENTENCE_BASED:
            return self._sentence_based_chunking(sentences, source_metadata)
        elif strategy == ChunkingStrategy.SEMANTIC_SIMILARITY:
            return self._semantic_similarity_chunking(sentences, source_metadata)
        elif strategy == ChunkingStrategy.TOKEN_AWARE:
            return self._token_aware_chunking(sentences, source_metadata)
        else:  # HYBRID
            return self._hybrid_chunking(sentences, source_metadata)
    
    def _sentence_based_chunking(self, sentences: List[str], source_metadata: Dict) -> List[DocumentChunk]:
        """Simple sentence-based chunking with smart overlap"""
        chunks = []
        current_chunk_sentences = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if adding this sentence would exceed token limit
            if current_tokens + sentence_tokens > self.max_chunk_tokens and current_chunk_sentences:
                # Create chunk from current sentences
                chunk = self._create_chunk(current_chunk_sentences, i - len(current_chunk_sentences), i - 1, chunks, source_metadata)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk_sentences) - self.overlap_sentences)
                current_chunk_sentences = current_chunk_sentences[overlap_start:]
                current_tokens = sum(self._estimate_tokens(s) for s in current_chunk_sentences)
            
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk_sentences:
            start_idx = len(sentences) - len(current_chunk_sentences)
            chunk = self._create_chunk(current_chunk_sentences, start_idx, len(sentences) - 1, chunks, source_metadata)
            chunks.append(chunk)
            
        return chunks
    
    def _semantic_similarity_chunking(self, sentences: List[str], source_metadata: Dict) -> List[DocumentChunk]:
        """Semantic similarity-based chunking"""
        if len(sentences) < 2:
            return self._sentence_based_chunking(sentences, source_metadata)
            
        # Get sentence embeddings
        embeddings = self.embedding_model.encode(sentences)
        
        # Calculate similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            )
            similarities.append(sim)
        
        # Find chunk boundaries where similarity drops
        chunk_boundaries = [0]  # Start with first sentence
        
        for i, similarity in enumerate(similarities):
            if similarity < self.similarity_threshold:
                chunk_boundaries.append(i + 1)
        
        chunk_boundaries.append(len(sentences))  # End with last sentence
        
        # Create chunks from boundaries
        chunks = []
        for i in range(len(chunk_boundaries) - 1):
            start_idx = chunk_boundaries[i]
            end_idx = chunk_boundaries[i + 1]
            
            # Add overlap with previous chunk
            if i > 0:
                overlap_start = max(chunk_boundaries[i - 1], start_idx - self.overlap_sentences)
                chunk_sentences = sentences[overlap_start:end_idx]
                actual_start = overlap_start
            else:
                chunk_sentences = sentences[start_idx:end_idx]
                actual_start = start_idx
            
            # Check token limit and split if necessary
            total_tokens = sum(self._estimate_tokens(s) for s in chunk_sentences)
            if total_tokens > self.max_chunk_tokens:
                # Fall back to sentence-based chunking for this section
                sub_chunks = self._sentence_based_chunking(chunk_sentences, source_metadata)
                chunks.extend(sub_chunks)
            else:
                chunk = self._create_chunk(chunk_sentences, actual_start, end_idx - 1, chunks, source_metadata)
                chunks.append(chunk)
        
        return chunks
    
    def _token_aware_chunking(self, sentences: List[str], source_metadata: Dict) -> List[DocumentChunk]:
        """Token-aware chunking that respects sentence boundaries"""
        chunks = []
        current_sentences = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_tokens(sentence)
            
            # If single sentence exceeds limit, split it carefully
            if sentence_tokens > self.max_chunk_tokens:
                # Create chunk from current sentences if any
                if current_sentences:
                    chunk = self._create_chunk(current_sentences, i - len(current_sentences), i - 1, chunks, source_metadata)
                    chunks.append(chunk)
                    current_sentences = []
                    current_tokens = 0
                
                # Split the long sentence
                sub_chunks = self._split_long_sentence(sentence, i, source_metadata)
                chunks.extend(sub_chunks)
                continue
            
            # Check if we should start a new chunk
            if current_tokens + sentence_tokens > self.max_chunk_tokens and current_sentences:
                chunk = self._create_chunk(current_sentences, i - len(current_sentences), i - 1, chunks, source_metadata)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_sentences) - self.overlap_sentences)
                current_sentences = current_sentences[overlap_start:]
                current_tokens = sum(self._estimate_tokens(s) for s in current_sentences)
            
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_sentences:
            start_idx = len(sentences) - len(current_sentences)
            chunk = self._create_chunk(current_sentences, start_idx, len(sentences) - 1, chunks, source_metadata)
            chunks.append(chunk)
            
        return chunks
    
    def _hybrid_chunking(self, sentences: List[str], source_metadata: Dict) -> List[DocumentChunk]:
        """Hybrid approach combining semantic similarity and token awareness"""
        # First pass: semantic similarity chunking
        semantic_chunks = self._semantic_similarity_chunking(sentences, source_metadata)
        
        # Second pass: ensure token limits are respected
        final_chunks = []
        for chunk in semantic_chunks:
            if chunk.metadata.token_count > self.max_chunk_tokens:
                # Re-chunk this section with token-aware method
                chunk_sentences = chunk.text.split('. ')  # Rough approximation
                if chunk_sentences[-1] and not chunk_sentences[-1].endswith('.'):
                    chunk_sentences[-1] += '.'
                
                sub_chunks = self._token_aware_chunking(chunk_sentences, source_metadata)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _create_chunk(self, sentences: List[str], start_idx: int, end_idx: int, 
                     existing_chunks: List[DocumentChunk], source_metadata: Dict) -> DocumentChunk:
        """Create a DocumentChunk with metadata"""
        chunk_text = '. '.join(sentences)
        if not chunk_text.endswith('.'):
            chunk_text += '.'
            
        token_count = sum(self._estimate_tokens(s) for s in sentences)
        chunk_id = f"chunk_{len(existing_chunks)}"
        
        # Calculate overlap information
        overlap_prev = None
        overlap_next = None
        
        if existing_chunks:
            # Check overlap with previous chunk
            prev_chunk = existing_chunks[-1]
            if start_idx <= prev_chunk.metadata.end_sentence:
                overlap_prev = prev_chunk.metadata.end_sentence - start_idx + 1
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            start_sentence=start_idx,
            end_sentence=end_idx,
            token_count=token_count,
            overlap_with_previous=overlap_prev,
            overlap_with_next=overlap_next  # Will be set when next chunk is created
        )
        
        return DocumentChunk(
            text=chunk_text,
            metadata=metadata,
            source_metadata=source_metadata or {}
        )
    
    def _split_long_sentence(self, sentence: str, sentence_idx: int, source_metadata: Dict) -> List[DocumentChunk]:
        """Split a very long sentence into smaller chunks"""
        # Split on commas, semicolons, or conjunctions as fallback
        sub_parts = re.split(r'[,;]|\s+(?:and|but|or|however|therefore|moreover)\s+', sentence)
        
        chunks = []
        current_part = ""
        current_tokens = 0
        
        for part in sub_parts:
            part_tokens = self._estimate_tokens(part)
            
            if current_tokens + part_tokens > self.max_chunk_tokens and current_part:
                # Create chunk from current part
                chunk_id = f"long_sentence_{sentence_idx}_{len(chunks)}"
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    start_sentence=sentence_idx,
                    end_sentence=sentence_idx,
                    token_count=current_tokens
                )
                
                chunk = DocumentChunk(
                    text=current_part.strip(),
                    metadata=metadata,
                    source_metadata=source_metadata or {}
                )
                chunks.append(chunk)
                
                current_part = part
                current_tokens = part_tokens
            else:
                current_part += part
                current_tokens += part_tokens
        
        # Add final part
        if current_part.strip():
            chunk_id = f"long_sentence_{sentence_idx}_{len(chunks)}"
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                start_sentence=sentence_idx,
                end_sentence=sentence_idx,
                token_count=current_tokens
            )
            
            chunk = DocumentChunk(
                text=current_part.strip(),
                metadata=metadata,
                source_metadata=source_metadata or {}
            )
            chunks.append(chunk)
            
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (more accurate than word count)"""
        # Approximation: 1 token ≈ 4 characters for English
        return max(1, len(text) // 4)


def chunk_document_advanced(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID,
    max_chunk_tokens: int = 250,
    overlap_sentences: int = 2,
    similarity_threshold: float = 0.5,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    source_metadata: Dict = None
) -> List[DocumentChunk]:
    """
    Main function for advanced document chunking
    
    Args:
        text: The document text to chunk
        strategy: Chunking strategy to use
        max_chunk_tokens: Maximum tokens per chunk
        overlap_sentences: Number of sentences to overlap between chunks
        similarity_threshold: Threshold for semantic similarity chunking
        embedding_model_name: Model for computing embeddings
        source_metadata: Metadata about the source document
        
    Returns:
        List of DocumentChunk objects with enhanced metadata
    """
    chunker = SemanticChunker(
        embedding_model_name=embedding_model_name,
        similarity_threshold=similarity_threshold,
        max_chunk_tokens=max_chunk_tokens,
        overlap_sentences=overlap_sentences
    )
    
    return chunker.chunk_document(text, strategy, source_metadata)


# Helper function for backward compatibility
def convert_to_zenml_format(chunks: List[DocumentChunk], doc_id: str, path: str) -> List[Dict]:
    """Convert DocumentChunk objects to ZenML pipeline format"""
    zenml_chunks = []
    
    for chunk in chunks:
        zenml_chunk = {
            "id": f"{doc_id}::chunk::{chunk.metadata.chunk_id}",
            "text": chunk.text,
            "metadata": {
                "source": path,
                "doc_id": doc_id,
                "chunk_id": chunk.metadata.chunk_id,
                "token_count": chunk.metadata.token_count,
                "start_sentence": chunk.metadata.start_sentence,
                "end_sentence": chunk.metadata.end_sentence,
                "chunking_strategy": "advanced_semantic",
                "overlap_previous": chunk.metadata.overlap_with_previous,
                "overlap_next": chunk.metadata.overlap_with_next,
            }
        }
        
        # Add source metadata if available
        if chunk.source_metadata:
            zenml_chunk["metadata"].update(chunk.source_metadata)
            
        zenml_chunks.append(zenml_chunk)
    
    return zenml_chunks