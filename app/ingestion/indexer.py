import hashlib
import logging
from typing import Any

from app.ingestion.pipeline import IngestionPipeline
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.retrieval.bm25 import BM25Index
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IngestionIndexer:
    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        bm25_index: BM25Index,
        chunk_repository: ChunkRepository,
        document_repository: DocumentRepository | None = None,
    ) -> None:
        self.ingestion_pipeline = ingestion_pipeline
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.bm25_index = bm25_index
        self.chunk_repository = chunk_repository
        self.document_repository = document_repository

    def index_file(
        self,
        file_path: str,
        force: bool = False,
    ) -> tuple[Document, list[DocumentChunk]]:
        logger.info("Indexing file: %s", file_path)
        document, chunks = self.ingestion_pipeline.process(file_path)
        return self._index_document_and_chunks(document, chunks, force=force)

    def index_document(
        self,
        document: Document,
        force: bool = False,
    ) -> tuple[Document, list[DocumentChunk]]:
        logger.info("Indexing document object: %s ('%s')", document.document_id, document.title)
        document, chunks = self.ingestion_pipeline.process_document(document)
        return self._index_document_and_chunks(document, chunks, force=force)

    def _index_document_and_chunks(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        force: bool = False,
    ) -> tuple[Document, list[DocumentChunk]]:
        current_hash = compute_content_hash(document.content)
        document.metadata["content_hash"] = current_hash

        # Idempotency check: if document has not changed, skip re-embedding unless forced
        if not force and self.document_repository is not None:
            existing_doc = self.document_repository.get_by_id(document.document_id)
            if existing_doc and existing_doc.metadata.get("content_hash") == current_hash:
                logger.info(
                    "Document '%s' content hash matches (%s); skipping re-embedding.",
                    document.document_id,
                    current_hash[:8],
                )
                existing_chunks = self.chunk_repository.get_by_document(document.document_id)
                if existing_chunks:
                    return document, existing_chunks

        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]

        if not child_chunks:
            raise ValueError("No child chunks were generated.")

        logger.info(
            "Document '%s': %d parent chunks, %d child chunks.",
            document.document_id,
            len(parent_chunks),
            len(child_chunks),
        )

        # Clean up stale chunks from previous version if any
        self._cleanup_stale_chunks(document.document_id, chunks)

        # Store Document metadata in repository
        if self.document_repository is not None:
            self.document_repository.save(document)

        # Store both parent and child chunks in persistent repository
        self.chunk_repository.save_many(chunks)

        # Dense embedding & Qdrant upsert
        texts = [c.content for c in child_chunks]
        vectors = self.embedding_service.embed_batch(texts)

        self.qdrant_service.upsert_chunks(
            chunks=child_chunks,
            vectors=vectors,
        )

        logger.info("Upserted %d vectors to Qdrant.", len(vectors))

        # Accumulate sparse index
        self.bm25_index.add(child_chunks)

        logger.info("BM25 index now has %d total chunks.", self.bm25_index.chunk_count)
        return document, chunks

    def _cleanup_stale_chunks(self, document_id: str, new_chunks: list[DocumentChunk]) -> None:
        """
        Delete stale chunk records if document content was modified.
        """
        existing_chunks = self.chunk_repository.get_by_document(document_id)
        if not existing_chunks:
            return

        new_chunk_ids = {c.chunk_id for c in new_chunks}
        stale_chunks = [c for c in existing_chunks if c.chunk_id not in new_chunk_ids]

        if stale_chunks:
            logger.info("Removing %d stale chunks for document '%s'", len(stale_chunks), document_id)
            for stale in stale_chunks:
                self.chunk_repository.delete(stale.chunk_id)