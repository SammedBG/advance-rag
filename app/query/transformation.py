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
        enable_hyde: bool = False,
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

        if enable_hyde:
            hyde_doc = self.generate_hyde_passage(original_query, structured_query.intent)
            self._append_unique(retrieval_queries, hyde_doc)

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

    def generate_hyde_passage(
        self,
        query: str,
        intent: str | None = None,
    ) -> str:
        """
        Generate a hypothetical document passage (HyDE) to improve dense vector retrieval.
        Creates an archetype documentation paragraph matching technical expectations.
        """
        clean_q = query.strip().rstrip("?")
        if intent == "troubleshooting" or "error" in clean_q.lower() or "issue" in clean_q.lower():
            return (
                f"When diagnosing and resolving {clean_q}, common root causes include misconfiguration, "
                f"resource constraints, network partition, or failed dependencies. Check logs, inspect status codes, "
                f"verify environment settings, and restart failed instances."
            )
        elif intent == "comparison" or " vs " in clean_q.lower():
            return (
                f"Comparison regarding {clean_q}: The primary differences lie in performance characteristics, "
                f"concurrency models, architectural paradigms, memory overhead, and specific operational trade-offs."
            )
        elif intent == "configuration" or "setup" in clean_q.lower() or "how to" in clean_q.lower():
            return (
                f"To configure and set up {clean_q}: Define the required parameters in the configuration file, "
                f"specify environment variables, apply the deployment manifests, and verify running services."
            )
        else:
            return (
                f"Overview and documentation for {clean_q}: This component provides core functionality, "
                f"standard API interfaces, automated lifecycle management, and scalable infrastructure support."
            )

    def decompose_complex_query(
        self,
        query: str,
    ) -> list[str]:
        """
        Decompose multi-part questions connected by conjunctions into distinct sub-queries.
        """
        sub_queries: list[str] = [query.strip()]
        lower_q = query.lower()

        split_delimiters = [" and also ", " as well as ", " and how to ", " along with ", " additionally "]
        for delim in split_delimiters:
            if delim in lower_q:
                parts = lower_q.split(delim)
                for part in parts:
                    clean_part = part.strip().rstrip("?")
                    if len(clean_part) > 5 and clean_part not in sub_queries:
                        sub_queries.append(clean_part)

        return sub_queries

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