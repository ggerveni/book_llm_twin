from typing import List, Dict

from sentence_transformers import SentenceTransformer
from zenml import step
import uuid


@step(enable_cache=False)
def embed_documents(
    chunks: List[Dict], embedding_model_name: str
) -> List[Dict]:
    """Embed text chunks using Sentence-Transformers."""
    model = SentenceTransformer(embedding_model_name)
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True).tolist()

    points: List[Dict] = []
    for chunk, vector in zip(chunks, vectors):
        raw_id = str(chunk["id"])  # e.g., "doc::chunk::idx"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
        payload = {"text": chunk["text"], "metadata": chunk.get("metadata", {}).copy()}
        payload["metadata"]["original_id"] = raw_id
        points.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": payload,
            }
        )
    return points


