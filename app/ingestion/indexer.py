from app.ingestion.pipeline import IngestionPipeline
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.retrieval.bm25 import BM25Index
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


class IngestionIndexer:
    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        bm25_index: BM25Index,
        chunk_repository: ChunkRepository,
    ) -> None:
        self.ingestion_pipeline = ingestion_pipeline
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.bm25_index = bm25_index
        self.chunk_repository = chunk_repository

    def index_file(
        self,
        file_path: str,
    ) -> tuple[Document, list[DocumentChunk]]:
        document, chunks = self.ingestion_pipeline.process(
            file_path
        )

        child_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child"
        ]

        if not child_chunks:
            raise ValueError(
                "No child chunks were generated."
            )

        # Store both parent and child chunks.
        self.chunk_repository.save_many(chunks)

        texts = [
            chunk.content
            for chunk in child_chunks
        ]

        vectors = self.embedding_service.embed_batch(
            texts
        )

        self.qdrant_service.upsert_chunks(
            chunks=child_chunks,
            vectors=vectors,
        )

        self.bm25_index.build(child_chunks)

        return document, chunks