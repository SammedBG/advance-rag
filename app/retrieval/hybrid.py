import logging

from app.models.search import SearchFilter
from app.query.transformation import QueryTransformation
from app.query.understanding import QueryUnderstanding
from app.retrieval.bm25 import BM25Index
from app.retrieval.filters import MetadataFilterBuilder
from app.retrieval.reranker import RerankResult, Reranker
from app.retrieval.rrf import ReciprocalRankFusion, FusionResult
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService

logger = logging.getLogger(__name__)


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

        self.query_understanding = QueryUnderstanding()
        self.query_transformation = QueryTransformation()

    def search(
        self,
        query: str,
        limit: int = 5,
        retrieval_limit: int = 20,
        filters: SearchFilter | dict | None = None,
        enable_hyde: bool = False,
    ) -> list[RerankResult]:
        if not query.strip():
            return []

        if limit <= 0:
            return []

        if retrieval_limit < limit:
            retrieval_limit = limit

        qdrant_filter = MetadataFilterBuilder.build_qdrant_filter(filters)
        bm25_predicate = MetadataFilterBuilder.build_chunk_predicate(filters)

        # ---------------------------------------------------------
        # 1. Query Understanding
        # ---------------------------------------------------------
        structured_query = self.query_understanding.understand(
            query
        )

        logger.info(
            "Query understood: intent=%s, keywords=%s",
            structured_query.intent,
            structured_query.keywords,
        )

        # ---------------------------------------------------------
        # 2. Query Transformation
        # ---------------------------------------------------------
        transformed_query = self.query_transformation.transform(
            structured_query,
            enable_hyde=enable_hyde,
        )

        retrieval_queries = transformed_query.retrieval_queries

        if not retrieval_queries:
            retrieval_queries = [query]

        logger.info(
            "Retrieval queries: %d",
            len(retrieval_queries),
        )

        # ---------------------------------------------------------
        # 3. Retrieve using every transformed query
        # ---------------------------------------------------------
        dense_results_by_query: list[list] = []
        sparse_results_by_query: list[list] = []

        for retrieval_query in retrieval_queries:
            query_vector = self.embedding_service.embed(
                retrieval_query
            )

            dense_results = self.qdrant_service.search(
                vector=query_vector,
                limit=retrieval_limit,
                query_filter=qdrant_filter,
            )

            sparse_results = self.bm25_index.search(
                query=retrieval_query,
                limit=retrieval_limit,
                filter_fn=bm25_predicate,
            )

            dense_results_by_query.append(
                dense_results
            )

            sparse_results_by_query.append(
                sparse_results
            )

        # ---------------------------------------------------------
        # 4. Global RRF across all queries and both retrievers
        # ---------------------------------------------------------
        fused_results = self._global_rrf(
            dense_results_by_query=dense_results_by_query,
            sparse_results_by_query=sparse_results_by_query,
            limit=retrieval_limit,
        )

        logger.info(
            "RRF fusion: %d candidates.",
            len(fused_results),
        )

        # ---------------------------------------------------------
        # 5. Rerank using the ORIGINAL user query
        # ---------------------------------------------------------
        reranked = self.reranker.rerank(
            query=query,
            results=fused_results,
            limit=limit,
        )

        logger.info(
            "Reranked: %d results.",
            len(reranked),
        )

        return reranked

    def _global_rrf(
        self,
        dense_results_by_query: list[list],
        sparse_results_by_query: list[list],
        limit: int,
    ) -> list[FusionResult]:
        """
        Perform one global Reciprocal Rank Fusion calculation.

        Every transformed query contributes independently to the
        final RRF score.

        Dense and sparse rankings are treated as separate ranked
        lists, while identical chunks accumulate score across all
        lists.
        """

        if limit <= 0:
            return []

        fused: dict[str, FusionResult] = {}

        # ---------------------------------------------------------
        # Dense retrieval lists
        # ---------------------------------------------------------
        for query_results in dense_results_by_query:
            for rank, result in enumerate(
                query_results,
                start=1,
            ):
                chunk = self.rrf._to_chunk(result)
                chunk_id = chunk.chunk_id

                if chunk_id not in fused:
                    fused[chunk_id] = FusionResult(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        score=0.0,
                    )

                fused[chunk_id].score += (
                    1.0 / (self.rrf.k + rank)
                )

                existing_rank = fused[chunk_id].dense_rank

                if (
                    existing_rank is None
                    or rank < existing_rank
                ):
                    fused[chunk_id].dense_rank = rank

        # ---------------------------------------------------------
        # Sparse retrieval lists
        # ---------------------------------------------------------
        for query_results in sparse_results_by_query:
            for rank, result in enumerate(
                query_results,
                start=1,
            ):
                chunk = self.rrf._to_chunk(result)
                chunk_id = chunk.chunk_id

                if chunk_id not in fused:
                    fused[chunk_id] = FusionResult(
                        chunk_id=chunk_id,
                        chunk=chunk,
                        score=0.0,
                    )

                fused[chunk_id].score += (
                    1.0 / (self.rrf.k + rank)
                )

                existing_rank = fused[chunk_id].sparse_rank

                if (
                    existing_rank is None
                    or rank < existing_rank
                ):
                    fused[chunk_id].sparse_rank = rank

        return sorted(
            fused.values(),
            key=lambda result: result.score,
            reverse=True,
        )[:limit]