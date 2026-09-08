from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import RerankResult, Reranker
from app.retrieval.rrf import ReciprocalRankFusion
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


class HybridSearch:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        bm25_index: BM25Index,
        rrf: ReciprocalRankFusion,
        reranker: Reranker,
    ) -> None:
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.bm25_index = bm25_index
        self.rrf = rrf
        self.reranker = reranker

    def search(
        self,
        query: str,
        limit: int = 5,
        retrieval_limit: int = 20,
    ) -> list[RerankResult]:
        if not query.strip():
            return []

        if limit <= 0:
            return []

        if retrieval_limit < limit:
            retrieval_limit = limit

        query_vector = self.embedding_service.embed(query)

        dense_results = self.qdrant_service.search(
            vector=query_vector,
            limit=retrieval_limit,
        )

        sparse_results = self.bm25_index.search(
            query=query,
            limit=retrieval_limit,
        )

        fused_results = self.rrf.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            limit=retrieval_limit,
        )

        return self.reranker.rerank(
            query=query,
            results=fused_results,
            limit=limit,
        )