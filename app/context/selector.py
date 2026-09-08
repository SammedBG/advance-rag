from dataclasses import dataclass

import tiktoken

from app.context.parent_expander import ExpandedContext


@dataclass
class SelectedContext:
    chunk_id: str
    parent_id: str | None
    document_id: str
    title: str
    heading_path: list[str]
    content: str
    retrieval_score: float
    dense_rank: int | None
    sparse_rank: int | None
    token_count: int


class ContextSelector:
    def __init__(
        self,
        max_tokens: int = 3000,
        min_score: float = 0.0,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        self.max_tokens = max_tokens
        self.min_score = min_score
        self.encoder = tiktoken.get_encoding(
            encoding_name
        )

    def select(
        self,
        contexts: list[ExpandedContext],
    ) -> list[SelectedContext]:
        if not contexts:
            return []

        selected: list[SelectedContext] = []
        seen_content: set[str] = set()
        total_tokens = 0

        # Parent expansion preserves reranker ordering,
        # so we process the highest-ranked context first.
        for context in contexts:
            if context.retrieval_score < self.min_score:
                continue

            normalized_content = self._normalize_content(
                context.content
            )

            if not normalized_content:
                continue

            content_hash = normalized_content

            if content_hash in seen_content:
                continue

            token_count = len(
                self.encoder.encode(
                    context.content
                )
            )

            if total_tokens + token_count > self.max_tokens:
                continue

            selected.append(
                SelectedContext(
                    chunk_id=context.chunk_id,
                    parent_id=context.parent_id,
                    document_id=context.document_id,
                    title=context.title,
                    heading_path=context.heading_path,
                    content=context.content,
                    retrieval_score=context.retrieval_score,
                    dense_rank=context.dense_rank,
                    sparse_rank=context.sparse_rank,
                    token_count=token_count,
                )
            )

            seen_content.add(content_hash)
            total_tokens += token_count

            if total_tokens >= self.max_tokens:
                break

        return selected

    @staticmethod
    def _normalize_content(content: str) -> str:
        return " ".join(content.lower().split())

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))