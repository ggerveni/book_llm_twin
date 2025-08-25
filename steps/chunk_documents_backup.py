from typing import List, Dict

from zenml import step


def _chunk_text(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_tokens, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return chunks


@step(enable_cache=False)
def chunk_documents(
    documents: List[Dict],
    chunk_tokens: int = 250,
    overlap_tokens: int = 40,
) -> List[Dict]:
    """Split documents into overlapping chunks suitable for embedding."""

    out: List[Dict] = []
    for doc in documents:
        text = doc.get("text", "")
        path = doc.get("path", "")
        doc_id = doc.get("doc_id")
        chunks = _chunk_text(text, chunk_tokens, overlap_tokens)
        for idx, chunk in enumerate(chunks):
            out.append(
                {
                    "id": f"{doc_id}::chunk::{idx}",
                    "text": chunk,
                    "metadata": {"source": path, "doc_id": doc_id, "chunk_id": str(idx)},
                }
            )
    return out


