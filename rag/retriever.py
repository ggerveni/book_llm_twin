from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Filter, PointStruct
from sentence_transformers import SentenceTransformer


load_dotenv()


@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str
    chunk_id: str
    doc_id: Optional[str] = None


class QdrantRetriever:
    def __init__(
        self,
        collection_name: str,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_model_name: Optional[str] = None,
        top_k: int = 5,
    ) -> None:
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.embedding_model_name = (
            embedding_model_name
            or os.getenv("EMBEDDING_MODEL")
        )
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL is not set. Define it in your .env file.")
        if not self.embedding_model_name:
            raise ValueError("EMBEDDING_MODEL is not set. Define it in your .env file.")
        self.top_k = top_k

        self.client = QdrantClient(url=self.qdrant_url, api_key=self.qdrant_api_key)
        self.embedder = SentenceTransformer(self.embedding_model_name)

    def retrieve(self, query: str, score_threshold: Optional[float] = None) -> List[RetrievedChunk]:
        query_vec = self.embedder.encode(query).tolist()

        # Robust guard: verify embedding dim vs collection vector size for various client return types
        expected_dim: Optional[int] = None
        try:
            info = self.client.get_collection(self.collection_name)
            # Try attribute-style
            try:
                expected_dim = int(getattr(getattr(getattr(info, "config", None), "params", None).vectors, "size"))  # type: ignore[attr-defined]
            except Exception:
                # Try nested under .result
                try:
                    expected_dim = int(getattr(getattr(getattr(getattr(info, "result", None), "config", None), "params", None).vectors, "size"))  # type: ignore[attr-defined]
                except Exception:
                    # Try dict-style
                    if isinstance(info, dict):
                        cfg = (((info.get("result") or {}).get("config") or {}).get("params") or {}).get("vectors")
                        if isinstance(cfg, dict) and "size" in cfg:
                            expected_dim = int(cfg["size"])
                        elif isinstance(cfg, dict) and cfg:
                            first = next(iter(cfg.values()))
                            if isinstance(first, dict) and "size" in first:
                                expected_dim = int(first["size"])
        except Exception:
            pass
        if expected_dim and expected_dim != len(query_vec):
            raise ValueError(
                f"Embedding dimension mismatch: collection expects {expected_dim}, got {len(query_vec)} from '{self.embedding_model_name}'. "
                f"Re-ingest with the same embedding model or switch to the model used for ingestion."
            )

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=self.top_k,
                with_payload=True,
                score_threshold=score_threshold,
            )
        except UnexpectedResponse as e:
            msg = str(e)
            hint = (
                "Qdrant returned an error during search. This is often caused by a vector dimension mismatch. "
                f"Query vector dim = {len(query_vec)} from '{self.embedding_model_name}'. "
            )
            if expected_dim:
                hint += f"Collection expects dim = {expected_dim}. "
            hint += (
                "Re-ingest your data with the same embedding model you're using for retrieval, "
                "or switch the retrieval embedding to match the model used at ingestion, or recreate the collection."
            )
            raise ValueError(hint + (f"\nRaw error: {msg}" if msg else ""))

        retrieved: List[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            text = payload.get("text", "")
            metadata = payload.get("metadata", {}) or {}
            source = metadata.get("source", metadata.get("path", "Unknown"))
            chunk_id = metadata.get("chunk_id", str(uuid.uuid4()))
            doc_id = metadata.get("doc_id")
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    score=hit.score or 0.0,
                    source=source,
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                )
            )
        return retrieved


