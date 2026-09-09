from dataclasses import dataclass
import re
from typing import Any

from app.generation.grounding import GroundingValidator


@dataclass
class GenerationMetrics:
    query_id: str
    faithfulness_score: float = 0.0
    grounded: bool = False
    answer_relevance_score: float = 0.0
    token_f1: float = 0.0
    token_precision: float = 0.0
    token_recall: float = 0.0


class GenerationEvaluator:
    """
    Evaluates generation faithfulness and semantic relevance to ground-truth answers.
    """

    def __init__(self, grounding_validator: GroundingValidator | None = None) -> None:
        self.grounding_validator = grounding_validator or GroundingValidator()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return [w for w in cleaned.split() if len(w) > 2]

    def compute_token_f1(self, generated: str, reference: str) -> tuple[float, float, float]:
        gen_tokens = self._tokenize(generated)
        ref_tokens = self._tokenize(reference)

        if not gen_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0

        gen_set = set(gen_tokens)
        ref_set = set(ref_tokens)

        overlap = gen_set.intersection(ref_set)
        precision = len(overlap) / len(gen_set) if gen_set else 0.0
        recall = len(overlap) / len(ref_set) if ref_set else 0.0

        if precision + recall == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return round(precision, 4), round(recall, 4), round(f1, 4)

    def evaluate(
        self,
        query_id: str,
        generated_answer: str,
        expected_answer: str,
        contexts: list[str],
    ) -> GenerationMetrics:
        # Faithfulness check via GroundingValidator
        grounding_result = self.grounding_validator.validate(
            answer=generated_answer,
            contexts=contexts,
        )

        precision, recall, f1 = self.compute_token_f1(
            generated=generated_answer,
            reference=expected_answer,
        )

        return GenerationMetrics(
            query_id=query_id,
            faithfulness_score=round(grounding_result.score, 4),
            grounded=grounding_result.grounded,
            answer_relevance_score=f1,
            token_f1=f1,
            token_precision=precision,
            token_recall=recall,
        )
