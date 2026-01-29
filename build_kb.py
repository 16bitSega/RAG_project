from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_settings
from src.rag.index import build_or_refresh_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default=None, help="Path to chunks.jsonl")
    args = parser.parse_args()

    settings = get_settings()
    chunks = Path(args.chunks) if args.chunks else settings.normalized_chunks_path

    stats = build_or_refresh_index(
        chunks_jsonl=chunks,
        chroma_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
    )
    print(stats)


if __name__ == "__main__":
    main()
