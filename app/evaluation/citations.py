from dataclasses import dataclass
from app.generation.citation import CitationValidator


@dataclass
class CitationMetrics:
    query_id: str
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    valid_citation_rate: float = 1.0
    has_hallucinated_citations: bool = False
    citation_density_per_100_words: float = 0.0


class CitationEvaluator:
    """
    Evaluates citation accuracy, validity, and hallucinated references.
    """

    def __init__(self, validator: CitationValidator | None = None) -> None:
        self.validator = validator or CitationValidator()

    def evaluate(
        self,
        query_id: str,
        answer: str,
        context_count: int,
    ) -> CitationMetrics:
        result = self.validator.validate(
            answer=answer,
            context_count=context_count,
        )

        total = len(result.citations)
        invalid = len(result.invalid_citations)
        valid = total - invalid

        rate = round(valid / total, 4) if total > 0 else 1.0

        words = answer.split()
        word_count = max(len(words), 1)
        density = round((total / word_count) * 100, 2)

        return CitationMetrics(
            query_id=query_id,
            total_citations=total,
            valid_citations=valid,
            invalid_citations=invalid,
            valid_citation_rate=rate,
            has_hallucinated_citations=invalid > 0,
            citation_density_per_100_words=density,
        )
