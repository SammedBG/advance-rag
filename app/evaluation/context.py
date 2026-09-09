from dataclasses import dataclass
from typing import Any


@dataclass
class ContextMetrics:
    query_id: str
    selected_count: int = 0
    compressed_count: int = 0
    selected_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    keyword_retention: float = 0.0
    budget_compliant: bool = True


class ContextEvaluator:
    """
    Evaluates context selection, token efficiency, and keyword preservation.
    """

    def evaluate(
        self,
        query_id: str,
        selected_contexts: list[Any],
        compressed_contexts: list[Any],
        relevant_keywords: list[str],
        max_budget: int = 2000,
    ) -> ContextMetrics:
        selected_tokens = sum(
            getattr(c, "token_count", 0) for c in selected_contexts
        )
        compressed_tokens = sum(
            getattr(c, "compressed_token_count", getattr(c, "token_count", 0))
            for c in compressed_contexts
        )

        compression_ratio = 0.0
        if selected_tokens > 0:
            compression_ratio = round(compressed_tokens / selected_tokens, 3)

        # Measure keyword retention in compressed text
        combined_compressed = " ".join(
            getattr(c, "content", "") for c in compressed_contexts
        ).lower()

        if relevant_keywords:
            matched = sum(
                1 for kw in relevant_keywords if kw.lower() in combined_compressed
            )
            retention = round(matched / len(relevant_keywords), 3)
        else:
            retention = 1.0

        budget_compliant = compressed_tokens <= max_budget

        return ContextMetrics(
            query_id=query_id,
            selected_count=len(selected_contexts),
            compressed_count=len(compressed_contexts),
            selected_tokens=selected_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            keyword_retention=retention,
            budget_compliant=budget_compliant,
        )
