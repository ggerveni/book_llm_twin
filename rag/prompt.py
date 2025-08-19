from typing import List, Dict
import os


def build_prompt(question: str, contexts: List[Dict[str, str]]) -> str:
    """
    Construct a grounded prompt for the LLM using retrieved contexts.

    The assistant is instructed to answer in English and only rely on the
    provided context. If the answer cannot be found, it should say so explicitly.
    """

    context_blocks = _format_context_blocks(contexts)

    context_text = "\n\n".join(context_blocks)

    system_instructions = (
        "You are an assistant that answers strictly based on the provided book/document context. "
        "Use only the CONTEXT to answer. "
        "If the answer is not in the context, say plainly: 'I can't find this in the context.' "
        "Respond in clear, concise English."
    )

    prompt = (
        f"INSTRUCTIONS:\n{system_instructions}\n\n"
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    return prompt


def build_messages(question: str, contexts: List[Dict[str, str]]):
    """Return chat-style messages for Ollama chat API.

    messages = [
      {"role": "system", "content": ...},
      {"role": "user", "content": ...},
    ]
    """
    context_blocks = _format_context_blocks(contexts)
    context_text = "\n\n".join(context_blocks) or "(No context found)"

    system = (
        "You are an assistant that answers strictly based on the provided book/document context. "
        "Use only the CONTEXT. If the answer is not present, reply: "
        "'I can't find this in the context.' Respond in clear, concise English."
    )

    user = (
        f"CONTEXT:\n{context_text}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _format_context_blocks(contexts: List[Dict[str, str]]) -> List[str]:
    """Trim and limit contexts to avoid overly long prompts that can break models."""
    try:
        max_contexts = int(os.getenv("PROMPT_MAX_CONTEXTS"))
    except Exception:
        max_contexts = None
    try:
        max_chars_per_context = int(os.getenv("PROMPT_MAX_CHARS_PER_CONTEXT"))
    except Exception:
        max_chars_per_context = None

    if not max_contexts:
        raise ValueError("PROMPT_MAX_CONTEXTS is not set. Define it in your .env file.")
    if not max_chars_per_context:
        raise ValueError("PROMPT_MAX_CHARS_PER_CONTEXT is not set. Define it in your .env file.")

    trimmed_blocks: List[str] = []
    for idx, ctx in enumerate(contexts[:max_contexts], start=1):
        source = ctx.get("source", "")
        text = (ctx.get("text", "") or "").strip()
        if len(text) > max_chars_per_context:
            text = text[:max_chars_per_context] + " …"
        trimmed_blocks.append(f"[Source {idx}: {source}]\n{text}")
    return trimmed_blocks


