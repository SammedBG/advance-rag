import re
from dataclasses import dataclass


@dataclass
class CitationValidation:
    valid: bool
    citations: list[int]
    invalid_citations: list[int]
    missing_citations: bool


class CitationValidator:
    CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    def validate(
        self,
        answer: str,
        context_count: int,
    ) -> CitationValidation:
        if not answer.strip():
            return CitationValidation(
                valid=False,
                citations=[],
                invalid_citations=[],
                missing_citations=True,
            )

        matches = self.CITATION_PATTERN.findall(answer)

        citations = sorted(
            {int(match) for match in matches}
        )

        invalid_citations = [
            citation
            for citation in citations
            if citation < 1 or citation > context_count
        ]

        missing_citations = len(citations) == 0

        valid = (
            not missing_citations
            and not invalid_citations
        )

        return CitationValidation(
            valid=valid,
            citations=citations,
            invalid_citations=invalid_citations,
            missing_citations=missing_citations,
        )