import re
from dataclasses import dataclass


@dataclass
class GroundingResult:
    score: float
    grounded: bool
    matched_terms: list[str]
    unmatched_terms: list[str]


class GroundingValidator:
    STOP_WORDS = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "by",
        "from",
        "into",
        "about",
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "can",
        "could",
        "should",
        "would",
        "will",
        "does",
        "do",
        "did",
        "not",
        "only",
        "also",
    }

    def __init__(
        self,
        threshold: float = 0.35,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "Grounding threshold must be between 0 and 1."
            )

        self.threshold = threshold

    def validate(
        self,
        answer: str,
        contexts: list[str],
    ) -> GroundingResult:
        if not answer.strip():
            return GroundingResult(
                score=0.0,
                grounded=False,
                matched_terms=[],
                unmatched_terms=[],
            )

        if not contexts:
            return GroundingResult(
                score=0.0,
                grounded=False,
                matched_terms=[],
                unmatched_terms=[],
            )

        answer_terms = self._extract_terms(answer)

        if not answer_terms:
            return GroundingResult(
                score=0.0,
                grounded=False,
                matched_terms=[],
                unmatched_terms=[],
            )

        context_text = " ".join(contexts).lower()

        matched_terms = []
        unmatched_terms = []

        for term in answer_terms:
            if term in context_text:
                matched_terms.append(term)
            else:
                unmatched_terms.append(term)

        score = len(matched_terms) / len(answer_terms)

        return GroundingResult(
            score=round(score, 3),
            grounded=score >= self.threshold,
            matched_terms=matched_terms,
            unmatched_terms=unmatched_terms,
        )

    def _extract_terms(self, text: str) -> list[str]:
        terms = re.findall(
            r"[a-zA-Z0-9_./:-]+",
            text.lower(),
        )

        return sorted(
            {
                term
                for term in terms
                if len(term) > 2
                and term not in self.STOP_WORDS
                and not term.isdigit()
            }
        )