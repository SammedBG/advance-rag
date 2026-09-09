import logging
from dataclasses import dataclass
from typing import Any, Callable

from rank_bm25 import BM25Okapi

from app.models.chunk import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    chunk: DocumentChunk
    score: float
    rank: int


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Replace the entire index with the given chunks."""
        self._chunks = []
        self._tokenized_corpus = []
        self._bm25 = None
        self.add(chunks)

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Append chunks to the existing index and rebuild BM25."""
        child_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child"
        ]

        if not child_chunks:
            return

        # Deduplicate by chunk_id against existing chunks
        existing_ids = {
            chunk.chunk_id for chunk in self._chunks
        }

        new_chunks = [
            chunk
            for chunk in child_chunks
            if chunk.chunk_id not in existing_ids
        ]

        if not new_chunks:
            logger.debug(
                "No new chunks to add to BM25 index."
            )
            return

        self._chunks.extend(new_chunks)
        self._tokenized_corpus.extend(
            self._tokenize(chunk.content)
            for chunk in new_chunks
        )

        if not self._tokenized_corpus:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(
            "BM25 index updated: %d total chunks.",
            len(self._chunks),
        )

    @property
    def chunk_count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def search(
        self,
        query: str,
        limit: int = 10,
        filter_fn: Any = None,
    ) -> list[BM25Result]:
        if not query.strip():
            return []

        if self._bm25 is None:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[BM25Result] = []

        for index in ranked_indexes:
            chunk = self._chunks[index]
            if filter_fn is not None and not filter_fn(chunk):
                continue

            results.append(
                BM25Result(
                    chunk=chunk,
                    score=float(scores[index]),
                    rank=len(results) + 1,
                )
            )

            if len(results) >= limit:
                break

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()