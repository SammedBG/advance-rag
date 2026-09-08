from dataclasses import dataclass
import re

import tiktoken

from app.context.selector import SelectedContext


@dataclass
class CompressedContext:
    chunk_id: str
    parent_id: str | None
    document_id: str
    title: str
    heading_path: list[str]
    content: str
    retrieval_score: float
    dense_rank: int | None
    sparse_rank: int | None
    original_token_count: int
    compressed_token_count: int


class ContextCompressor:
    def __init__(
        self,
        max_tokens: int = 2000,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding(
            encoding_name
        )

    def compress(
        self,
        query: str,
        contexts: list[SelectedContext],
    ) -> list[CompressedContext]:
        if not query.strip():
            return []

        if not contexts:
            return []

        query_terms = self._extract_terms(query)

        compressed: list[CompressedContext] = []
        total_tokens = 0

        for context in contexts:
            sentences = self._split_into_sentences(
                context.content
            )

            if not sentences:
                continue

            ranked_sentences = [
                (
                    sentence,
                    self._sentence_score(
                        sentence,
                        query_terms,
                    ),
                    index,
                )
                for index, sentence in enumerate(sentences)
            ]

            ranked_sentences.sort(
                key=lambda item: (
                    item[1],
                    -item[2],
                ),
                reverse=True,
            )

            selected_sentences: list[str] = []

            for sentence, score, _ in ranked_sentences:
                # Keep relevant sentences.
                if score <= 0:
                    continue

                candidate = " ".join(
                    selected_sentences + [sentence]
                )

                candidate_tokens = len(
                    self.encoder.encode(candidate)
                )

                if (
                    total_tokens + candidate_tokens
                    <= self.max_tokens
                ):
                    selected_sentences.append(sentence)

            if not selected_sentences:
                # Fall back to the beginning of the context
                # rather than returning nothing.
                selected_sentences = [
                    sentences[0]
                ]

            # Restore the original document order.
            selected_set = set(selected_sentences)

            ordered_sentences = [
                sentence
                for sentence in sentences
                if sentence in selected_set
            ]

            content = " ".join(ordered_sentences)

            compressed_token_count = len(
                self.encoder.encode(content)
            )

            if (
                total_tokens + compressed_token_count
                > self.max_tokens
            ):
                remaining_tokens = (
                    self.max_tokens - total_tokens
                )

                if remaining_tokens <= 0:
                    break

                content = self._truncate_to_tokens(
                    content,
                    remaining_tokens,
                )

                compressed_token_count = len(
                    self.encoder.encode(content)
                )

            if not content.strip():
                continue

            compressed.append(
                CompressedContext(
                    chunk_id=context.chunk_id,
                    parent_id=context.parent_id,
                    document_id=context.document_id,
                    title=context.title,
                    heading_path=context.heading_path,
                    content=content,
                    retrieval_score=context.retrieval_score,
                    dense_rank=context.dense_rank,
                    sparse_rank=context.sparse_rank,
                    original_token_count=context.token_count,
                    compressed_token_count=compressed_token_count,
                )
            )

            total_tokens += compressed_token_count

            if total_tokens >= self.max_tokens:
                break

        return compressed

    @staticmethod
    def _extract_terms(query: str) -> set[str]:
        terms = re.findall(
            r"[a-zA-Z0-9_./:-]+",
            query.lower(),
        )

        return {
            term
            for term in terms
            if len(term) > 1
        }

    @staticmethod
    def _split_into_sentences(
        text: str,
    ) -> list[str]:
        text = text.strip()

        if not text:
            return []

        # Preserve code blocks and normal text reasonably
        # while splitting prose into sentence-like units.
        parts = re.split(
            r"(?<=[.!?])\s+|\n{2,}",
            text,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    @staticmethod
    def _sentence_score(
        sentence: str,
        query_terms: set[str],
    ) -> float:
        normalized = sentence.lower()

        if not query_terms:
            return 0.0

        matches = sum(
            1
            for term in query_terms
            if term in normalized
        )

        score = float(matches)

        # Preserve technical/error-code sentences.
        if re.search(
            r"\b(error|warning|exception|code|command|"
            r"configuration|config|yaml|api|http)\b",
            normalized,
        ):
            score += 0.5

        return score

    def _truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
    ) -> str:
        token_ids = self.encoder.encode(text)

        if len(token_ids) <= max_tokens:
            return text

        return self.encoder.decode(
            token_ids[:max_tokens]
        )