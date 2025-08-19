from zenml import pipeline

from steps.load_documents import load_documents
from steps.chunk_documents import chunk_documents
from steps.embed_documents import embed_documents
from steps.index_qdrant import index_qdrant


@pipeline(enable_cache=False)
def ingestion_pipeline(
    data_dir: str,
    collection_name: str,
    embedding_model_name: str,
    qdrant_url: str = "",
    qdrant_api_key: str = "",
    chunk_tokens: int = 250,
    overlap_tokens: int = 40,
):
    docs = load_documents(data_dir=data_dir)
    chunks = chunk_documents(
        documents=docs, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
    )
    points = embed_documents(chunks=chunks, embedding_model_name=embedding_model_name)
    index_qdrant(
        points=points,
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )


