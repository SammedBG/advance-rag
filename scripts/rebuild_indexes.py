import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.core.logging import setup_logging
from app.services.container import ServiceContainer

setup_logging()
logger = logging.getLogger("rebuild_indexes_cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild BM25 sparse index and Qdrant dense vector index deterministically from persistent storage."
    )
    parser.add_argument(
        "--re-embed",
        action="store_true",
        help="Also re-embed and upsert all dense vectors into Qdrant",
    )

    args = parser.parse_args()
    container = ServiceContainer()

    chunk_repo = container.get_chunk_repository()
    bm25_index = container.get_bm25_index()
    embedding_service = container.get_embedding_service()
    qdrant_service = container.get_qdrant_service()

    # If repository is empty (e.g. fresh environment), index default raw docs
    child_chunks = [c for c in chunk_repo.get_all() if c.chunk_type == "child"]

    if not child_chunks:
        logger.info("Persistent ChunkRepository is empty. Ingesting raw documents from 'data/raw' first...")
        from scripts.index_documents import main as index_main
        sys_argv = ["index_documents.py", "--data-dir", "data/raw"]
        old_argv = sys.argv
        sys.argv = sys_argv
        index_main()
        sys.argv = old_argv
        child_chunks = [c for c in chunk_repo.get_all() if c.chunk_type == "child"]

    logger.info("Rebuilding BM25 sparse index from %d child chunks...", len(child_chunks))
    bm25_index.build(child_chunks)
    logger.info("BM25 sparse index rebuild complete (%d total chunks).", bm25_index.chunk_count)

    if args.re_embed and child_chunks:
        logger.info("Re-embedding %d child chunks into Qdrant...", len(child_chunks))
        texts = [c.content for c in child_chunks]
        vectors = embedding_service.embed_batch(texts)
        qdrant_service.upsert_chunks(chunks=child_chunks, vectors=vectors)
        logger.info("Qdrant dense index upsert complete.")


if __name__ == "__main__":
    main()
