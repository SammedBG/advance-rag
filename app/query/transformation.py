from dataclasses import dataclass

from app.query.models import StructuredQuery


@dataclass
class TransformedQuery:
    original_query: str
    rewritten_query: str
    retrieval_queries: list[str]


class QueryTransformation:
    """
    Deterministic query transformation layer.

    Converts a structured user query into one or more
    retrieval-friendly queries.
    """

    def transform(
        self,
        structured_query: StructuredQuery,
    ) -> TransformedQuery:
        original_query = structured_query.original_query.strip()

        if not original_query:
            raise ValueError(
                "Original query cannot be empty."
            )

        rewritten_query = self._rewrite(
            structured_query
        )

        retrieval_queries = self._build_retrieval_queries(
            structured_query,
            rewritten_query,
        )

        return TransformedQuery(
            original_query=original_query,
            rewritten_query=rewritten_query,
            retrieval_queries=retrieval_queries,
        )

    def _rewrite(
        self,
        structured_query: StructuredQuery,
    ) -> str:
        technical_terms = structured_query.technical_terms
        keywords = structured_query.keywords
        intent = structured_query.intent

        terms = self._unique(
            technical_terms + keywords
        )

        if not terms:
            return structured_query.original_query

        if intent == "comparison":
            return (
                f"comparison of "
                f"{' '.join(terms)}"
            )

        if intent == "configuration":
            return (
                f"configuration and setup of "
                f"{' '.join(terms)}"
            )

        if intent == "troubleshooting":
            return (
                f"troubleshooting "
                f"{' '.join(terms)}"
            )

        if intent == "definition":
            return (
                f"definition and explanation of "
                f"{' '.join(terms)}"
            )

        if intent == "procedure":
            return (
                f"procedure and steps for "
                f"{' '.join(terms)}"
            )

        return " ".join(terms)

    def _build_retrieval_queries(
        self,
        structured_query: StructuredQuery,
        rewritten_query: str,
    ) -> list[str]:
        queries: list[str] = []

        self._append_unique(
            queries,
            structured_query.original_query,
        )

        self._append_unique(
            queries,
            rewritten_query,
        )

        if structured_query.technical_terms:
            technical_query = " ".join(
                structured_query.technical_terms
            )

            self._append_unique(
                queries,
                technical_query,
            )

        return queries

    @staticmethod
    def _unique(
        values: list[str],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            normalized = value.strip()

            if not normalized:
                continue

            if normalized not in result:
                result.append(normalized)

        return result

    @staticmethod
    def _append_unique(
        values: list[str],
        value: str,
    ) -> None:
        normalized = value.strip()

        if not normalized:
            return

        if normalized not in values:
            values.append(normalized)