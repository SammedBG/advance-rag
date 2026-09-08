from dataclasses import dataclass

from app.models.chunk import DocumentChunk
from app.repositories.chunk_repository import ChunkRepository
from app.retrieval.reranker import RerankResult


@dataclass
class ExpandedContext:
    chunk_id: str
    parent_id: str | None
    document_id: str
    title: str
    heading_path: list[str]
    content: str
    retrieval_score: float
    dense_rank: int | None
    sparse_rank: int | None


class ParentExpander:
    def __init__(
        self,
        chunk_repository: ChunkRepository,
    ) -> None:
        self.chunk_repository = chunk_repository

    def expand(
        self,
        results: list[RerankResult],
    ) -> list[ExpandedContext]:
        expanded: list[ExpandedContext] = []
        seen_parent_ids: set[str] = set()

        for result in results:
            child = result.chunk

            if not child.parent_id:
                expanded.append(
                    self._build_context(
                        chunk=child,
                        result=result,
                    )
                )
                continue

            if child.parent_id in seen_parent_ids:
                continue

            parent = self.chunk_repository.get(
                child.parent_id
            )

            if parent is None:
                expanded.append(
                    self._build_context(
                        chunk=child,
                        result=result,
                    )
                )
                continue

            seen_parent_ids.add(child.parent_id)

            expanded.append(
                self._build_context(
                    chunk=parent,
                    result=result,
                )
            )

        return expanded

    @staticmethod
    def _build_context(
        chunk: DocumentChunk,
        result: RerankResult,
    ) -> ExpandedContext:
        return ExpandedContext(
            chunk_id=chunk.chunk_id,
            parent_id=chunk.parent_id,
            document_id=chunk.document_id,
            title=chunk.title,
            heading_path=chunk.heading_path,
            content=chunk.content,
            retrieval_score=result.score,
            dense_rank=result.dense_rank,
            sparse_rank=result.sparse_rank,
        )