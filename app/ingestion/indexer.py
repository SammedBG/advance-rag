from app.ingestion.pipeline import IngestionPipeline
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


class IngestionIndexer:

    def __init__(
        self,
        ingestion_pipeline: IngestionPipeline,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
    ):
        self.ingestion_pipeline = (
            ingestion_pipeline
        )

        self.embedding_service = (
            embedding_service
        )

        self.qdrant_service = qdrant_service

    def index_file(
        self,
        file_path: str,
    ) -> tuple[
        Document,
        list[DocumentChunk],
    ]:

        document, chunks = (
            self.ingestion_pipeline.process(
                file_path
            )
        )

        # Only child chunks are embedded.
        child_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child"
        ]

        if not child_chunks:
            raise ValueError(
                "No child chunks were generated."
            )

        texts = [
            chunk.content
            for chunk in child_chunks
        ]

        vectors = (
            self.embedding_service.embed_batch(
                texts
            )
        )

        self.qdrant_service.upsert_chunks(
            chunks=child_chunks,
            vectors=vectors,
        )

        return document, child_chunks