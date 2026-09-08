from dataclasses import dataclass

from app.models.chunk import DocumentChunk


@dataclass
class FusionResult:
    chunk_id: str
    chunk: DocumentChunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF k must be greater than zero.")

        self.k = k

    def fuse(
        self,
        dense_results: list,
        sparse_results: list,
        limit: int = 10,
    ) -> list[FusionResult]:
        if limit <= 0:
            return []

        fused: dict[str, FusionResult] = {}

        # Dense retrieval results
        for rank, result in enumerate(dense_results, start=1):
            chunk = self._to_chunk(result)
            chunk_id = chunk.chunk_id

            if chunk_id not in fused:
                fused[chunk_id] = FusionResult(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=0.0,
                )

            fused[chunk_id].score += 1.0 / (self.k + rank)
            fused[chunk_id].dense_rank = rank

        # Sparse / BM25 results
        for rank, result in enumerate(sparse_results, start=1):
            chunk = self._to_chunk(result)
            chunk_id = chunk.chunk_id

            if chunk_id not in fused:
                fused[chunk_id] = FusionResult(
                    chunk_id=chunk_id,
                    chunk=chunk,
                    score=0.0,
                )

            fused[chunk_id].score += 1.0 / (self.k + rank)
            fused[chunk_id].sparse_rank = rank

        return sorted(
            fused.values(),
            key=lambda result: result.score,
            reverse=True,
        )[:limit]

    @staticmethod
    def _to_chunk(result) -> DocumentChunk:
        """
        Normalize both retrieval result types into DocumentChunk.

        Supported:
        - BM25Result -> result.chunk
        - Qdrant ScoredPoint -> result.payload
        """

        # BM25Result
        if hasattr(result, "chunk"):
            chunk = result.chunk

            if not isinstance(chunk, DocumentChunk):
                raise TypeError(
                    "BM25 result contains an invalid chunk type."
                )

            return chunk

        # Qdrant ScoredPoint
        if hasattr(result, "payload"):
            payload = result.payload or {}

            required_fields = [
                "chunk_id",
                "document_id",
                "content",
                "title",
                "chunk_type",
                "token_count",
            ]

            missing_fields = [
                field
                for field in required_fields
                if field not in payload
            ]

            if missing_fields:
                raise ValueError(
                    "Qdrant payload is missing required fields: "
                    + ", ".join(missing_fields)
                )

            return DocumentChunk(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                content=str(payload["content"]),
                title=str(payload["title"]),
                heading_path=list(
                    payload.get("heading_path") or []
                ),
                chunk_index=int(
                    payload.get("chunk_index", 0)
                ),
                chunk_type=str(payload["chunk_type"]),
                parent_id=payload.get("parent_id"),
                token_count=int(payload["token_count"]),
                metadata=dict(
                    payload.get("metadata") or {}
                ),
            )

        raise ValueError(
            f"Unsupported retrieval result type: "
            f"{type(result).__name__}"
        )