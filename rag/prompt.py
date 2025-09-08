"""
Enhanced prompt building module for RAG with source citations and score awareness.
This module constructs prompts that leverage enhanced retrieval results.
"""

from typing import List, Dict, Optional
import os


def build_prompt(
    question: str, 
    contexts: List[Dict[str, str]], 
    include_scores: bool = False,
    retrieval_method: str = "hybrid"
) -> str:
    """
    Construct a grounded prompt for the LLM using retrieved contexts.
    Enhanced to handle hybrid retrieval results with source citations.

    Args:
        question: User's question
        contexts: List of context dictionaries with text, source, and optionally scores
        include_scores: Whether to include relevance scores in prompt
        retrieval_method: Method used for retrieval (vector, hybrid, etc.)
    """

    context_blocks = _format_context_blocks(contexts, include_scores)
    context_text = "\n\n".join(context_blocks)

    # Enhanced system instructions that encourage source citations
    system_instructions = (
        "You are an assistant that answers strictly based on the provided book/document context. "
        "Use only the CONTEXT to answer. "
        "When possible, mention which source (Source 1, Source 2, etc.) supports your answer. "
        "If the answer is not in the context, say plainly: 'I can't find this in the context.' "
        "Respond in clear, concise English."
    )

    # Add retrieval method context if using enhanced search
    if retrieval_method == "hybrid":
        system_instructions += (
            " The context was retrieved using both semantic and keyword search for comprehensive coverage."
        )

    prompt = (
        f"INSTRUCTIONS:\n{system_instructions}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    return prompt


def build_messages(
    question: str, 
    contexts: List[Dict[str, str]], 
    include_scores: bool = False,
    retrieval_method: str = "hybrid"
) -> List[Dict[str, str]]:
    """
    Return chat-style messages for Ollama chat API with enhanced context formatting.

    Args:
        question: User's question
        contexts: Retrieved context chunks
        include_scores: Whether to include relevance scores
        retrieval_method: Retrieval method used
        
    Returns:
        List of message dictionaries for chat API
    """
    context_blocks = _format_context_blocks(contexts, include_scores)
    context_text = "\n\n".join(context_blocks) or "(No context found)"

    # Enhanced system message
    system = (
        "You are an assistant that answers strictly based on the provided book/document context. "
        "Use only the CONTEXT. Cite sources (Source 1, Source 2, etc.) when possible. "
        "If the answer is not present, reply: 'I can't find this in the context.' "
        "Respond in clear, concise English."
    )

    # Add retrieval method context
    if retrieval_method == "hybrid":
        system += " The context uses both semantic and keyword search for better coverage."

    user = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_multi_query_prompt(
    question: str,
    contexts_per_query: List[List[Dict[str, str]]],
    expanded_queries: List[str],
    include_scores: bool = False
) -> str:
    """
    Build prompt for multi-query retrieval results.
    
    Args:
        question: Original question
        contexts_per_query: List of context lists, one per expanded query
        expanded_queries: List of expanded queries used
        include_scores: Whether to include relevance scores
        
    Returns:
        Formatted prompt string
    """
    
    system_instructions = (
        "You are an assistant that answers based on context from multiple search queries. "
        "The context below was retrieved using the original question and related queries "
        "to provide comprehensive coverage. Use all relevant context to answer. "
        "Cite sources when possible and indicate if information comes from different searches. "
        "If the answer is not in any context, say: 'I can't find this in the context.'"
    )
    
    # Format contexts by query
    all_context_blocks = []
    query_section_num = 1
    
    for expanded_query, contexts in zip(expanded_queries, contexts_per_query):
        if contexts:
            all_context_blocks.append(f"=== Results for query: '{expanded_query}' ===")
            context_blocks = _format_context_blocks(contexts, include_scores, query_section_num)
            all_context_blocks.extend(context_blocks)
            query_section_num += len(contexts)
            all_context_blocks.append("")  # Empty line between query sections
    
    context_text = "\n\n".join(all_context_blocks)
    
    prompt = (
        f"INSTRUCTIONS:\n{system_instructions}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        "ANSWER:"
    )
    
    return prompt


def _format_context_blocks(
    contexts: List[Dict[str, str]], 
    include_scores: bool = False,
    start_index: int = 1
) -> List[str]:
    """
    Enhanced context block formatting with score and metadata support.
    
    Args:
        contexts: List of context dictionaries
        include_scores: Whether to include relevance scores
        start_index: Starting index for source numbering
        
    Returns:
        List of formatted context blocks
    """
    # Get configuration from environment
    try:
        max_contexts = int(os.getenv("PROMPT_MAX_CONTEXTS", "4"))
    except (ValueError, TypeError):
        max_contexts = 4
        
    try:
        max_chars_per_context = int(os.getenv("PROMPT_MAX_CHARS_PER_CONTEXT", "1200"))
    except (ValueError, TypeError):
        max_chars_per_context = 1200

    trimmed_blocks: List[str] = []
    
    for idx, ctx in enumerate(contexts[:max_contexts]):
        source_num = start_index + idx
        source = ctx.get("source", "Unknown")
        text = (ctx.get("text", "") or "").strip()
        
        # Truncate text if too long
        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + " …"
        
        # Build source header
        source_header = f"[Source {source_num}: {source}]"
        
        # Add score if available and requested
        if include_scores and "score" in ctx:
            try:
                score = float(ctx["score"])
                source_header += f" (Relevance: {score:.2f})"
            except (ValueError, TypeError):
                pass
        
        # Add retrieval method if available
        if "retrieval_method" in ctx:
            source_header += f" [{ctx['retrieval_method']}]"
        
        formatted_block = f"{source_header}\n{text}"
        trimmed_blocks.append(formatted_block)
    
    return trimmed_blocks


def build_comparison_prompt(
    question: str,
    vector_contexts: List[Dict[str, str]],
    bm25_contexts: List[Dict[str, str]],
    hybrid_contexts: List[Dict[str, str]]
) -> str:
    """
    Build prompt comparing results from different retrieval methods.
    Useful for debugging and analysis.
    
    Args:
        question: User's question
        vector_contexts: Results from vector search
        bm25_contexts: Results from BM25 search
        hybrid_contexts: Results from hybrid search
        
    Returns:
        Comparison prompt string
    """
    
    prompt_parts = [
        "RETRIEVAL METHOD COMPARISON",
        "=" * 50,
        f"QUESTION: {question}",
        ""
    ]
    
    if vector_contexts:
        prompt_parts.extend([
            "VECTOR SEARCH RESULTS:",
            *_format_context_blocks(vector_contexts, include_scores=True),
            ""
        ])
    
    if bm25_contexts:
        prompt_parts.extend([
            "BM25 KEYWORD SEARCH RESULTS:",
            *_format_context_blocks(bm25_contexts, include_scores=True, start_index=len(vector_contexts) + 1),
            ""
        ])
    
    if hybrid_contexts:
        prompt_parts.extend([
            "HYBRID SEARCH RESULTS:",
            *_format_context_blocks(hybrid_contexts, include_scores=True, start_index=len(vector_contexts) + len(bm25_contexts) + 1),
            ""
        ])
    
    return "\n".join(prompt_parts)


def extract_sources_from_response(response: str) -> List[str]:
    """
    Extract source references from LLM response.
    
    Args:
        response: LLM response text
        
    Returns:
        List of source references found in the response
    """
    import re
    
    # Pattern to match "Source X" references
    source_pattern = r'Source\s+(\d+)'
    matches = re.findall(source_pattern, response, re.IGNORECASE)
    
    return [f"Source {match}" for match in matches]


def validate_response_grounding(
    response: str, 
    contexts: List[Dict[str, str]], 
    min_source_references: int = 1
) -> Dict[str, any]:
    """
    Validate that the response is properly grounded in the provided context.
    
    Args:
        response: LLM response
        contexts: Context chunks provided
        min_source_references: Minimum number of source references expected
        
    Returns:
        Dictionary with validation results
    """
    
    source_refs = extract_sources_from_response(response)
    
    # Check for hallucination indicators
    hallucination_phrases = [
        "i can't find this in the context",
        "not mentioned in the context",
        "the context doesn't contain",
        "based on my knowledge",  # Should rely only on context
        "in general",  # Should be specific to context
    ]
    
    has_refusal = any(phrase in response.lower() for phrase in hallucination_phrases[:3])
    has_hallucination_risk = any(phrase in response.lower() for phrase in hallucination_phrases[3:])
    
    return {
        "source_references": source_refs,
        "num_source_references": len(source_refs),
        "meets_min_references": len(source_refs) >= min_source_references,
        "has_proper_refusal": has_refusal,
        "hallucination_risk": has_hallucination_risk,
        "is_well_grounded": len(source_refs) >= min_source_references and not has_hallucination_risk,
        "available_sources": len(contexts)
    }