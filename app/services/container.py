from functools import lru_cache

from app.context.compressor import ContextCompressor
from app.context.parent_expander import ParentExpander
from app.context.selector import ContextSelector
from app.core.config import settings
from app.generation.llm import GroqLLM
from app.generation.prompt_builder import PromptBuilder
from app.generation.service import GenerationService
from app.ingestion.pipeline import IngestionPipeline
from app.repositories.chunk_repository import ChunkRepository
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import Reranker
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(
        model_name=settings.embedding_model
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
def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository()


@lru_cache
def get_parent_expander() -> ParentExpander:
    return ParentExpander(
        chunk_repository=get_chunk_repository(),
    )


@lru_cache
def get_context_selector() -> ContextSelector:
    return ContextSelector(
        max_tokens=3000,
        min_score=0.0,
    )


@lru_cache
def get_context_compressor() -> ContextCompressor:
    return ContextCompressor(
        max_tokens=2000,
    )


@lru_cache
def get_llm() -> GroqLLM:
    return GroqLLM(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )


@lru_cache
def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@lru_cache
def get_generation_service() -> GenerationService:
    return GenerationService(
        llm=get_llm(),
        prompt_builder=get_prompt_builder(),
    )