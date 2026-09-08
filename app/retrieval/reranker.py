from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from app.retrieval.rrf import FusionResult


@dataclass
class RerankResult:
    chunk_id: str
    chunk: object
    score: float
    dense_rank: int | None
    sparse_rank: int | None


class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[FusionResult],
        limit: int = 5,
    ) -> list[RerankResult]:
        if not query.strip():
            return []

        if not results:
            return []

        if limit <= 0:
            return []

        pairs = [
            (query, result.chunk.content)
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = [
            RerankResult(
                chunk_id=result.chunk_id,
                chunk=result.chunk,
                score=float(score),
                dense_rank=result.dense_rank,
                sparse_rank=result.sparse_rank,
            )
            for result, score in zip(results, scores)
        ]

        reranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return reranked[:limit]