from __future__ import annotations

import argparse
import os
import sys
from dotenv import load_dotenv

# Ensure project root is on PYTHONPATH when running as a script
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pipelines.ingestion_pipeline import ingestion_pipeline


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--data_dir", type=str, default=os.getenv("DATA_DIR"), help="Data folder")
    parser.add_argument(
        "--collection", type=str, default=os.getenv("QDRANT_COLLECTION")
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default=os.getenv("EMBEDDING_MODEL"),
    )
    parser.add_argument("--qdrant_url", type=str, default=os.getenv("QDRANT_URL"))
    parser.add_argument("--qdrant_api_key", type=str, default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--chunk_tokens", type=int, default=None)
    parser.add_argument("--overlap_tokens", type=int, default=None)

    args = parser.parse_args()

    if not args.data_dir:
        raise SystemExit("DATA_DIR is not set. Define it in .env or pass --data_dir.")
    if not args.collection:
        raise SystemExit("QDRANT_COLLECTION is not set. Define it in .env or pass --collection.")
    if not args.embedding:
        raise SystemExit("EMBEDDING_MODEL is not set. Define it in .env or pass --embedding.")

    # Resolve chunking params strictly from CLI or env
    if args.chunk_tokens is None:
        ct_env = os.getenv("CHUNK_TOKENS")
        if not ct_env:
            raise SystemExit("CHUNK_TOKENS is not set. Define it in .env or pass --chunk_tokens.")
        try:
            args.chunk_tokens = int(ct_env)
        except Exception:
            raise SystemExit("CHUNK_TOKENS must be an integer.")

    if args.overlap_tokens is None:
        ot_env = os.getenv("OVERLAP_TOKENS")
        if not ot_env:
            raise SystemExit("OVERLAP_TOKENS is not set. Define it in .env or pass --overlap_tokens.")
        try:
            args.overlap_tokens = int(ot_env)
        except Exception:
            raise SystemExit("OVERLAP_TOKENS must be an integer.")

    ingestion_pipeline(
        data_dir=args.data_dir,
        collection_name=args.collection,
        embedding_model_name=args.embedding,
        qdrant_url=(args.qdrant_url or ""),
        qdrant_api_key=(args.qdrant_api_key or ""),
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
    )


if __name__ == "__main__":
    main()


