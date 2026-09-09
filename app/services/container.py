from functools import lru_cache

from app.agent.nodes import AgentDependencies
from app.agent.service import RAGAgent
from app.cache.service import RedisCacheService
from app.context.compressor import ContextCompressor
from app.context.parent_expander import ParentExpander
from app.context.selector import ContextSelector
from app.core.config import settings
from app.generation.grounding import GroundingValidator
from app.generation.llm import GroqLLM
from app.generation.prompt_builder import PromptBuilder
from app.generation.service import GenerationService
from app.ingestion.indexer import IngestionIndexer
from app.ingestion.pipeline import IngestionPipeline
from app.mcp.service import MCPService
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridSearch
from app.retrieval.reranker import Reranker
from app.retrieval.rrf import ReciprocalRankFusion
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=settings.embedding_model)


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
    return Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache
def get_chunk_repository() -> ChunkRepository:
    return ChunkRepository()


@lru_cache
def get_document_repository() -> DocumentRepository:
    return DocumentRepository()


@lru_cache
def get_conversation_repository() -> ConversationRepository:
    return ConversationRepository()


@lru_cache
def get_cache_service() -> RedisCacheService:
    return RedisCacheService(
        redis_url=settings.redis_url,
        enabled=settings.cache_enabled,
    )


@lru_cache
def get_parent_expander() -> ParentExpander:
    return ParentExpander(chunk_repository=get_chunk_repository())


@lru_cache
def get_context_selector() -> ContextSelector:
    return ContextSelector(max_tokens=3000, min_score=0.0)


@lru_cache
def get_context_compressor() -> ContextCompressor:
    return ContextCompressor(max_tokens=2000)


@lru_cache
def get_llm() -> GroqLLM:
    return GroqLLM(api_key=settings.llm_api_key, model=settings.llm_model)


@lru_cache
def get_prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@lru_cache
def get_generation_service() -> GenerationService:
    return GenerationService(llm=get_llm(), prompt_builder=get_prompt_builder())


@lru_cache
def get_grounding_validator() -> GroundingValidator:
    return GroundingValidator(threshold=0.35)


@lru_cache
def get_hybrid_search() -> HybridSearch:
    return HybridSearch(
        embedding_service=get_embedding_service(),
        qdrant_service=get_qdrant_service(),
        bm25_index=get_bm25_index(),
        rrf=ReciprocalRankFusion(),
        reranker=get_reranker(),
    )


@lru_cache
def get_mcp_service() -> MCPService:
    return MCPService(server_url=settings.mcp_server_url)


@lru_cache
def get_agent_dependencies() -> AgentDependencies:
    return AgentDependencies(
        search_service=get_hybrid_search(),
        parent_expander=get_parent_expander(),
        context_selector=get_context_selector(),
        context_compressor=get_context_compressor(),
        generation_service=get_generation_service(),
        grounding_validator=get_grounding_validator(),
        mcp_service=get_mcp_service(),
    )


@lru_cache
def get_rag_agent() -> RAGAgent:
    return RAGAgent(dependencies=get_agent_dependencies())


@lru_cache
def get_indexer() -> IngestionIndexer:
    return IngestionIndexer(
        ingestion_pipeline=get_ingestion_pipeline(),
        embedding_service=get_embedding_service(),
        qdrant_service=get_qdrant_service(),
        bm25_index=get_bm25_index(),
        chunk_repository=get_chunk_repository(),
        document_repository=get_document_repository(),
    )


class ServiceContainer:
    """
    Unified Dependency Injection Service Container.
    """

    def get_embedding_service(self) -> EmbeddingService:
        return get_embedding_service()

    def get_qdrant_service(self) -> QdrantService:
        return get_qdrant_service()

    def get_ingestion_pipeline(self) -> IngestionPipeline:
        return get_ingestion_pipeline()

    def get_bm25_index(self) -> BM25Index:
        return get_bm25_index()

    def get_reranker(self) -> Reranker:
        return get_reranker()

    def get_chunk_repository(self) -> ChunkRepository:
        return get_chunk_repository()

    def get_document_repository(self) -> DocumentRepository:
        return get_document_repository()

    def get_conversation_repository(self) -> ConversationRepository:
        return get_conversation_repository()

    def get_cache_service(self) -> RedisCacheService:
        return get_cache_service()

    def get_parent_expander(self) -> ParentExpander:
        return get_parent_expander()

    def get_context_selector(self) -> ContextSelector:
        return get_context_selector()

    def get_context_compressor(self) -> ContextCompressor:
        return get_context_compressor()

    def get_llm(self) -> GroqLLM:
        return get_llm()

    def get_prompt_builder(self) -> PromptBuilder:
        return get_prompt_builder()

    def get_generation_service(self) -> GenerationService:
        return get_generation_service()

    def get_grounding_validator(self) -> GroundingValidator:
        return get_grounding_validator()

    def get_search_service(self) -> HybridSearch:
        return get_hybrid_search()

    def get_hybrid_search(self) -> HybridSearch:
        return get_hybrid_search()

    def get_mcp_service(self) -> MCPService:
        return get_mcp_service()

    def get_agent_dependencies(self) -> AgentDependencies:
        return get_agent_dependencies()

    def get_rag_agent(self) -> RAGAgent:
        return get_rag_agent()

    def get_indexer(self) -> IngestionIndexer:
        return get_indexer()