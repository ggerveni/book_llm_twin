import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PointStruct
from zenml import step


load_dotenv()


@step(enable_cache=False)
def index_qdrant(
    points: List[Dict],
    collection_name: str,
    qdrant_url: str = "",
    qdrant_api_key: str = "",
) -> int:
    """Create (if needed) and upsert points into a Qdrant collection.

    Returns number of indexed points.
    """

    if not points:
        return 0

    qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise ValueError("QDRANT_URL is not set. Define it in your .env file.")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    vector_dim = len(points[0]["vector"])  # infer from first vector

    # Ensure collection exists with correct vector size
    try:
        client.get_collection(collection_name)
    except UnexpectedResponse:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

    qdrant_points = [
        PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points
    ]
    client.upsert(collection_name=collection_name, points=qdrant_points)

    return len(points)


