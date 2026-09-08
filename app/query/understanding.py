import re

from app.query.models import StructuredQuery


class QueryUnderstanding:
    """
    Deterministic query understanding layer.

    Extracts:
    - user intent
    - keywords
    - technical terms
    - entities
    - metadata-style filters
    """

    INTENT_PATTERNS = {
        "comparison": [
            r"\bdifference between\b",
            r"\bcompare\b",
            r"\bvs\.?\b",
            r"\bversus\b",
        ],
        "configuration": [
            r"\bhow to configure\b",
            r"\bhow do i configure\b",
            r"\bconfiguration\b",
            r"\bconfigure\b",
            r"\bsetup\b",
            r"\bset up\b",
        ],
        "troubleshooting": [
            r"\berror\b",
            r"\bexception\b",
            r"\bfailed\b",
            r"\bfails?\b",
            r"\bnot working\b",
            r"\bissue\b",
            r"\bproblem\b",
            r"\bdebug\b",
            r"\bfix\b",
        ],
        "definition": [
            r"^what is\b",
            r"^what are\b",
            r"\bmeaning of\b",
            r"\bdefine\b",
        ],
        "procedure": [
            r"\bhow do i\b",
            r"\bhow can i\b",
            r"\bsteps to\b",
            r"\bsteps for\b",
            r"\bhow to\b",
        ],
    }

    TECHNICAL_TERMS = {
        "api",
        "authentication",
        "authorization",
        "backend",
        "bm25",
        "cache",
        "chunk",
        "chunks",
        "database",
        "docker",
        "documentation",
        "embedding",
        "fastapi",
        "github",
        "kubernetes",
        "langchain",
        "langgraph",
        "llm",
        "mcp",
        "metadata",
        "model",
        "models",
        "postgresql",
        "prompt",
        "prompts",
        "python",
        "qdrant",
        "rag",
        "redis",
        "reranking",
        "retrieval",
        "rrf",
        "sql",
        "vector",
        "persistence",
        "pipeline",
    }

    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }

    QUESTION_WORDS = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
    }

    def understand(
        self,
        query: str,
    ) -> StructuredQuery:
        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        normalized_query = self._normalize(
            query
        )

        intent = self._detect_intent(
            normalized_query
        )

        tokens = self._tokenize(
            normalized_query
        )

        technical_terms = (
            self._extract_technical_terms(
                tokens
            )
        )

        keywords = self._extract_keywords(
            tokens,
            technical_terms,
        )

        entities = self._extract_entities(
            query
        )

        filters = self._extract_filters(
            query
        )

        return StructuredQuery(
            original_query=query.strip(),
            intent=intent,
            keywords=keywords,
            technical_terms=technical_terms,
            entities=entities,
            filters=filters,
            language="en",
        )

    @staticmethod
    def _normalize(
        query: str,
    ) -> str:
        query = query.lower().strip()

        query = re.sub(
            r"\s+",
            " ",
            query,
        )

        return query

    @staticmethod
    def _tokenize(
        query: str,
    ) -> list[str]:
        return re.findall(
            r"[a-zA-Z0-9][a-zA-Z0-9_.:/-]*",
            query,
        )

    def _detect_intent(
        self,
        query: str,
    ) -> str:
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(
                    pattern,
                    query,
                    flags=re.IGNORECASE,
                ):
                    return intent

        return "informational"

    def _extract_technical_terms(
        self,
        tokens: list[str],
    ) -> list[str]:
        terms: list[str] = []

        for token in tokens:
            normalized = token.lower()

            if normalized in self.TECHNICAL_TERMS:
                if normalized not in terms:
                    terms.append(normalized)

        return terms

    def _extract_keywords(
        self,
        tokens: list[str],
        technical_terms: list[str],
    ) -> list[str]:
        keywords: list[str] = []

        for token in tokens:
            normalized = token.lower()

            if normalized in self.STOP_WORDS:
                continue

            if len(normalized) < 2:
                continue

            if normalized not in keywords:
                keywords.append(normalized)

        for term in technical_terms:
            if term not in keywords:
                keywords.append(term)

        return keywords

    def _extract_entities(
        self,
        query: str,
    ) -> list[str]:
        entities: list[str] = []

        patterns = [
            r"\b[A-Z][A-Za-z0-9.-]{2,}\b",
            r"\b[a-zA-Z0-9]+-[a-zA-Z0-9.-]+\b",
        ]

        for pattern in patterns:
            matches = re.findall(
                pattern,
                query,
            )

            for match in matches:
                normalized = match.lower()

                if normalized in self.QUESTION_WORDS:
                    continue

                if match not in entities:
                    entities.append(match)

        return entities

    @staticmethod
    def _extract_filters(
        query: str,
    ) -> dict[str, str]:
        filters: dict[str, str] = {}

        patterns = {
            "document": (
                r"\bdocument\s*[:=]\s*([^\s,]+)"
            ),
            "title": (
                r"\btitle\s*[:=]\s*([^,]+)"
            ),
            "section": (
                r"\bsection\s*[:=]\s*([^,]+)"
            ),
            "source": (
                r"\bsource\s*[:=]\s*([^\s,]+)"
            ),
        }

        for key, pattern in patterns.items():
            match = re.search(
                pattern,
                query,
                flags=re.IGNORECASE,
            )

            if match:
                filters[key] = (
                    match.group(1).strip()
                )

        return filters