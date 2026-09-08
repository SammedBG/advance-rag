from dataclasses import dataclass

from app.models.chunk import DocumentChunk
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
        chunks: list[DocumentChunk],
    ) -> None:
        self._parents: dict[str, DocumentChunk] = {
            chunk.chunk_id: chunk
            for chunk in chunks
            if chunk.chunk_type == "parent"
        }

    def expand(
        self,
        results: list[RerankResult],
    ) -> list[ExpandedContext]:
        expanded: list[ExpandedContext] = []
        seen_parents: set[str] = set()

        for result in results:
            child = result.chunk
            parent_id = child.parent_id

            # If there is no parent, use the retrieved child itself.
            if not parent_id:
                expanded.append(
                    self._create_context(
                        chunk=child,
                        score=result.score,
                        dense_rank=result.dense_rank,
                        sparse_rank=result.sparse_rank,
                    )
                )
                continue

            # Avoid returning the same parent multiple times.
            if parent_id in seen_parents:
                continue

            parent = self._parents.get(parent_id)

            # If the parent is unavailable, fall back to the child.
            if parent is None:
                expanded.append(
                    self._create_context(
                        chunk=child,
                        score=result.score,
                        dense_rank=result.dense_rank,
                        sparse_rank=result.sparse_rank,
                    )
                )
                continue

            seen_parents.add(parent_id)

            expanded.append(
                self._create_context(
                    chunk=parent,
                    score=result.score,
                    dense_rank=result.dense_rank,
                    sparse_rank=result.sparse_rank,
                )
            )

        return expanded

    @staticmethod
    def _create_context(
        chunk: DocumentChunk,
        score: float,
        dense_rank: int | None,
        sparse_rank: int | None,
    ) -> ExpandedContext:
        return ExpandedContext(
            chunk_id=chunk.chunk_id,
            parent_id=chunk.parent_id,
            document_id=chunk.document_id,
            title=chunk.title,
            heading_path=chunk.heading_path,
            content=chunk.content,
            retrieval_score=score,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
        )