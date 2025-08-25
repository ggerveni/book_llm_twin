#!/usr/bin/env python3
"""
Demo script to test the advanced chunking functionality.
Run this to see how different chunking strategies work.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.getcwd())

from advanced_chunking import (
    chunk_document_advanced,
    ChunkingStrategy,
    SentenceSplitter
)

# Sample document for testing
SAMPLE_TEXT = """
Artificial Intelligence and Machine Learning have revolutionized many industries. 
These technologies are transforming how we work and live. Machine learning algorithms 
can process vast amounts of data quickly and efficiently.

Natural language processing is a subfield of AI that focuses on the interaction 
between computers and human language. It enables machines to understand, interpret, 
and generate human language in a valuable way. This technology powers chatbots, 
translation services, and sentiment analysis tools.

Deep learning, a subset of machine learning, uses artificial neural networks to 
model and understand complex patterns. These networks are inspired by the human brain 
and can learn from large datasets without explicit programming. Deep learning has 
been particularly successful in image recognition, speech processing, and game playing.

The future of AI looks promising with continued advancements in computing power 
and algorithm development. However, it also raises important questions about ethics, 
privacy, and the impact on employment. As we move forward, it's crucial to develop 
AI responsibly and ensure it benefits all of humanity.
"""

def demonstrate_sentence_splitting():
    """Show how sentence splitting works"""
    print("=== SENTENCE SPLITTING DEMO ===")
    print("📝 Creating sentence splitter...")
    
    splitter = SentenceSplitter()
    print("🔍 Splitting text into sentences...")
    sentences = splitter.split_into_sentences(SAMPLE_TEXT)
    
    print(f"Original text length: {len(SAMPLE_TEXT)} characters")
    print(f"Number of sentences detected: {len(sentences)}\n")
    
    for i, sentence in enumerate(sentences, 1):
        print(f"{i:2d}. {sentence[:60]}...")
    
    print("\n" + "="*60 + "\n")


def demonstrate_chunking_strategies():
    """Compare different chunking strategies"""
    print("=== CHUNKING STRATEGIES COMPARISON ===")
    print("🧠 Loading embedding model (this may take a few minutes on first run)...")
    
    strategies = [
        ("Sentence-Based", ChunkingStrategy.SENTENCE_BASED),
        ("Semantic Similarity", ChunkingStrategy.SEMANTIC_SIMILARITY), 
        ("Token-Aware", ChunkingStrategy.TOKEN_AWARE),
        ("Hybrid", ChunkingStrategy.HYBRID),
    ]
    
    for strategy_name, strategy in strategies:
        print(f"\n--- {strategy_name} Strategy ---")
        print(f"⚙️ Processing with {strategy_name} strategy...")
        
        try:
            chunks = chunk_document_advanced(
                text=SAMPLE_TEXT,
                strategy=strategy,
                max_chunk_tokens=150,  # Smaller for demo
                overlap_sentences=1,
                similarity_threshold=0.6,
                source_metadata={"demo": True}
            )
            
            print(f"Number of chunks: {len(chunks)}")
            
            for i, chunk in enumerate(chunks, 1):
                print(f"\nChunk {i}:")
                print(f"  Tokens: {chunk.metadata.token_count}")
                print(f"  Sentences: {chunk.metadata.start_sentence}-{chunk.metadata.end_sentence}")
                if chunk.metadata.overlap_with_previous:
                    print(f"  Overlap with previous: {chunk.metadata.overlap_with_previous} sentences")
                print(f"  Text: {chunk.text[:120]}...")
                
        except Exception as e:
            print(f"Error with {strategy_name}: {str(e)}")
    
    print("\n" + "="*60 + "\n")


def demonstrate_token_estimation():
    """Show token estimation accuracy"""
    print("=== TOKEN ESTIMATION DEMO ===")
    
    test_texts = [
        "Short sentence.",
        "This is a medium-length sentence with some complexity.",
        "This is a very long sentence that contains multiple clauses, subclauses, and various punctuation marks to test the token estimation accuracy of our chunking system.",
    ]
    
    from advanced_chunking import SemanticChunker
    chunker = SemanticChunker()
    
    for text in test_texts:
        estimated_tokens = chunker._estimate_tokens(text)
        actual_words = len(text.split())
        chars = len(text)
        
        print(f"Text: {text}")
        print(f"  Characters: {chars}")
        print(f"  Words: {actual_words}")  
        print(f"  Estimated tokens: {estimated_tokens}")
        print(f"  Chars/token ratio: {chars/estimated_tokens:.1f}")
        print()


def demonstrate_overlap_handling():
    """Show how overlap between chunks works"""
    print("=== OVERLAP HANDLING DEMO ===")
    
    chunks = chunk_document_advanced(
        text=SAMPLE_TEXT,
        strategy=ChunkingStrategy.HYBRID,
        max_chunk_tokens=100,  # Small chunks to force overlaps
        overlap_sentences=2,
        source_metadata={"demo": "overlap"}
    )
    
    print(f"Created {len(chunks)} chunks with 2-sentence overlap\n")
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:")
        print(f"  Sentences {chunk.metadata.start_sentence}-{chunk.metadata.end_sentence}")
        if chunk.metadata.overlap_with_previous:
            print(f"  ↳ Overlaps {chunk.metadata.overlap_with_previous} sentences with previous")
        print(f"  Text: {chunk.text[:100]}...")
        print()
    
    # Show actual overlap content
    if len(chunks) > 1:
        print("--- Overlap Analysis ---")
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Find overlapping sentences (simplified)
            prev_sentences = prev_chunk.text.split('. ')
            curr_sentences = curr_chunk.text.split('. ')
            
            print(f"Between chunks {i} and {i+1}:")
            print(f"  Previous ends with: ...{prev_sentences[-1][:50]}...")
            print(f"  Current starts with: {curr_sentences[0][:50]}...")
            print()


def demonstrate_long_sentence_handling():
    """Show how very long sentences are handled"""
    print("=== LONG SENTENCE HANDLING DEMO ===")
    
    long_sentence = """This is an extremely long sentence that contains multiple clauses, subclauses, and various punctuation marks, and it also includes several conjunctions like 'and', 'but', 'however', and 'therefore', which should be used as natural breaking points when the sentence exceeds the maximum token limit, and this demonstrates how our advanced chunking system handles such cases gracefully without breaking semantic meaning, while also ensuring that the resulting chunks remain within the specified token limits, and this approach is much better than simply cutting off text at arbitrary positions, because it preserves the natural flow and readability of the content."""
    
    chunks = chunk_document_advanced(
        text=long_sentence,
        strategy=ChunkingStrategy.TOKEN_AWARE,
        max_chunk_tokens=50,  # Very small to force splitting
        source_metadata={"demo": "long_sentence"}
    )
    
    print(f"Long sentence split into {len(chunks)} chunks:")
    print(f"Original length: {len(long_sentence)} characters\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"Sub-chunk {i}:")
        print(f"  Tokens: {chunk.metadata.token_count}")
        print(f"  Text: {chunk.text}")
        print()


def performance_comparison():
    """Compare performance of different strategies"""
    print("=== PERFORMANCE COMPARISON ===")
    
    import time
    
    # Create a longer text for meaningful timing
    long_text = SAMPLE_TEXT * 5  # 5x the original text
    
    strategies = [
        ("Sentence-Based", ChunkingStrategy.SENTENCE_BASED),
        ("Semantic Similarity", ChunkingStrategy.SEMANTIC_SIMILARITY),
        ("Token-Aware", ChunkingStrategy.TOKEN_AWARE),
        ("Hybrid", ChunkingStrategy.HYBRID),
    ]
    
    results = []
    
    for strategy_name, strategy in strategies:
        start_time = time.time()
        
        try:
            chunks = chunk_document_advanced(
                text=long_text,
                strategy=strategy,
                max_chunk_tokens=250,
                overlap_sentences=2
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            results.append({
                'strategy': strategy_name,
                'chunks': len(chunks),
                'time': duration,
                'success': True
            })
            
        except Exception as e:
            results.append({
                'strategy': strategy_name,
                'chunks': 0,
                'time': 0,
                'success': False,
                'error': str(e)
            })
    
    print(f"Text length: {len(long_text)} characters")
    print(f"{'Strategy':<20} {'Chunks':<8} {'Time (s)':<10} {'Status'}")
    print("-" * 50)
    
    for result in results:
        if result['success']:
            print(f"{result['strategy']:<20} {result['chunks']:<8} {result['time']:<10.3f} ✓")
        else:
            print(f"{result['strategy']:<20} {'N/A':<8} {'N/A':<10} ✗ ({result.get('error', 'Unknown error')})")


def main():
    """Run all demonstrations"""
    print("ADVANCED CHUNKING SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("🚀 Starting chunking demonstrations...")
    print("⏰ Note: First run may take 2-5 minutes to download embedding model\n")
    
    try:
        print("1️⃣ Testing sentence splitting...")
        demonstrate_sentence_splitting()
        
        print("2️⃣ Testing chunking strategies...")
        demonstrate_chunking_strategies()
        
        print("3️⃣ Testing token estimation...")
        demonstrate_token_estimation()
        
        print("4️⃣ Testing overlap handling...")
        demonstrate_overlap_handling()
        
        print("5️⃣ Testing long sentence handling...")
        demonstrate_long_sentence_handling()
        
        print("6️⃣ Performance comparison...")
        performance_comparison()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Save these files in your project directory")
        print("2. Update your .env file with CHUNKING_STRATEGY=hybrid")
        print("3. Replace the original chunking step with the enhanced version")
        print("4. Test with your own documents!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        print("Make sure you have sentence-transformers installed:")
        print("pip install sentence-transformers")


if __name__ == "__main__":
    main()