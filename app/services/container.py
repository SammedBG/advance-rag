from functools import lru_cache

from app.core.config import settings
from app.ingestion.pipeline import IngestionPipeline
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


@lru_cache
def get_embedding_service() -> EmbeddingService:

    return EmbeddingService(
        model_name=settings.embedding_model
    )


@lru_cache
def get_qdrant_service() -> QdrantService:

    embedding_service = (
        get_embedding_service()
    )

    return QdrantService(
        url=settings.qdrant_url,
        collection_name=(
            settings.qdrant_collection
        ),
        vector_size=(
            embedding_service.dimension
        ),
    )


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:

    return IngestionPipeline()