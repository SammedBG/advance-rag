from functools import lru_cache

from app.context.parent_expander import ParentExpander
from app.core.config import settings
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import Reranker
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        model_name=settings.embedding_model,
    )


@lru_cache
def get_qdrant_service() -> QdrantService:
    embedding_service = get_embedding_service()

    return QdrantService(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
        vector_size=embedding_service.dimension,
    )


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


@lru_cache
def get_bm25_index() -> BM25Index:
    return BM25Index()


@lru_cache
def get_reranker() -> Reranker:
    return Reranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )


@lru_cache
def get_parent_expander() -> ParentExpander:
    return ParentExpander([])