from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.models.chunk import DocumentChunk


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
        child_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_type == "child"
        ]

        self._chunks = child_chunks
        self._tokenized_corpus = [
            self._tokenize(chunk.content)
            for chunk in child_chunks
        ]

        if not self._tokenized_corpus:
            self._bm25 = None
            return

        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(
        self,
        query: str,
        limit: int = 10,
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

        for rank, index in enumerate(ranked_indexes[:limit], start=1):
            results.append(
                BM25Result(
                    chunk=self._chunks[index],
                    score=float(scores[index]),
                    rank=rank,
                )
            )

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()