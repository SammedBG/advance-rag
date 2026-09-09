import argparse
import logging
from pathlib import Path
import sys

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.core.logging import setup_logging
from app.ingestion.loader import DocumentLoader
from app.services.container import ServiceContainer

setup_logging()
logger = logging.getLogger("index_documents_cli")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index documents into Advanced RAG platform (Dense Qdrant + Sparse BM25 + SQL Repository)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw documents to index (default: data/raw)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional single file path to index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing even if content hash is unchanged",
    )

    args = parser.parse_args()
    container = ServiceContainer()
    indexer = container.get_indexer()
    loader = DocumentLoader()

    files_to_index: list[Path] = []
    if args.file:
        files_to_index.append(Path(args.file))
    else:
        data_path = Path(args.data_dir)
        if not data_path.exists():
            logger.error("Data directory '%s' does not exist.", args.data_dir)
            sys.exit(1)
        for ext in loader.SUPPORTED_EXTENSIONS:
            files_to_index.extend(data_path.glob(f"*{ext}"))

    if not files_to_index:
        logger.warning("No supported document files found to index.")
        return

    logger.info("Found %d document files to index.", len(files_to_index))
    total_indexed_chunks = 0

    for file_path in sorted(files_to_index):
        try:
            doc, chunks = indexer.index_file(str(file_path), force=args.force)
            total_indexed_chunks += len(chunks)
            logger.info("Indexed '%s' (%d chunks total)", doc.title, len(chunks))
        except Exception as exc:
            logger.error("Failed to index '%s': %s", file_path, exc)

    logger.info("Indexing complete. Processed %d documents, %d total chunks in index.", len(files_to_index), total_indexed_chunks)


if __name__ == "__main__":
    main()
