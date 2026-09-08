from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


class VectorSearch:

    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
    ):
        self.embedding_service = (
            embedding_service
        )

        self.qdrant_service = (
            qdrant_service
        )

    def search(
        self,
        query: str,
        limit: int = 5,
    ):

        query_vector = (
            self.embedding_service.embed(
                query
            )
        )

        return self.qdrant_service.search(
            vector=query_vector,
            limit=limit,
        )