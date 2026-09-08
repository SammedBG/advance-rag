from app.models.chunk import DocumentChunk
from app.services.qdrant import QdrantService


class VectorStore:

    def __init__(
        self,
        qdrant_service: QdrantService,
    ):
        self.qdrant = qdrant_service

    def add(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:

        self.qdrant.upsert_chunks(
            chunks=chunks,
            vectors=vectors,
        )

    def search(
        self,
        vector: list[float],
        limit: int = 10,
    ):

        return self.qdrant.search(
            vector=vector,
            limit=limit,
        )