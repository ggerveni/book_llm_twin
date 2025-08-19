from __future__ import annotations

import os
from typing import List, Dict, Iterator, Optional

from dotenv import load_dotenv
import logging
import ollama

from .prompt import build_prompt, build_messages


load_dotenv()
logger = logging.getLogger(__name__)


def _format_error(prefix: str, err: Exception) -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    return (
        f"{prefix}: {type(err).__name__}: {str(err)}\n"
        f"(Check the Ollama service at {host} and the model name/tag in the sidebar.)"
    )

def generate_answer_stream(
    question: str,
    contexts: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Iterator[str]:
    """Stream an answer token-by-token from Ollama for better perceived latency."""
    model_name = model or os.getenv("OLLAMA_MODEL")
    if not model_name:
        yield "OLLAMA_MODEL is not set. Define it in your .env file."
        return
    prompt = build_prompt(question, contexts)

    num_predict = max_tokens or int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
    temp_value = temperature if temperature is not None else float(
        os.getenv("OLLAMA_TEMPERATURE", "0.2")
    )

    try:
        stream = ollama.generate(
            model=model_name,
            prompt=prompt,
            stream=True,
            options={
                "num_predict": num_predict,
                "temperature": temp_value,
            },
        )
        any_chunk = False
        for part in stream:
            chunk = part.get("response", "")
            if chunk:
                any_chunk = True
                yield chunk
        if any_chunk:
            return
        logger.warning("Ollama generate(stream): completed with no chunks")
    except Exception as err:
        yield _format_error("Error during generate(stream)" , err)
        return

    # Fallback to chat streaming
    messages = build_messages(question, contexts)
    try:
        stream = ollama.chat(
            model=model_name,
            messages=messages,
            stream=True,
            options={
                "num_predict": num_predict,
                "temperature": temp_value,
            },
        )
        any_chunk = False
        for part in stream:
            content = (part.get("message", {}) or {}).get("content", "")
            if content:
                any_chunk = True
                yield content
        if not any_chunk:
            logger.warning("Ollama chat(stream): completed with no chunks")
        return
    except Exception as err:
        yield _format_error("Error during chat(stream)", err)


