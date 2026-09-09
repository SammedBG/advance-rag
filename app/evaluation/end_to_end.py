from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

from app.evaluation.citations import CitationEvaluator, CitationMetrics
from app.evaluation.context import ContextEvaluator, ContextMetrics
from app.evaluation.generation import GenerationEvaluator, GenerationMetrics
from app.evaluation.retrieval import RetrievalEvaluator, RetrievalMetrics
from app.models.search import SearchRequest
from app.services.container import (
    get_context_compressor,
    get_context_selector,
    get_generation_service,
    get_grounding_validator,
    get_hybrid_search,
    get_parent_expander,
)

logger = logging.getLogger(__name__)


@dataclass
class TestCaseResult:
    query_id: str
    query: str
    category: str
    retrieval: RetrievalMetrics
    context: ContextMetrics
    generation: GenerationMetrics
    citation: CitationMetrics
    generated_answer: str
    expected_answer: str


@dataclass
class EvaluationScorecard:
    total_queries: int
    mean_mrr: float
    mean_hit_rate_at_1: float
    mean_hit_rate_at_5: float
    mean_precision_at_5: float
    mean_recall_at_5: float
    mean_ndcg_at_5: float
    mean_compression_ratio: float
    mean_keyword_retention: float
    mean_faithfulness_score: float
    grounding_pass_rate: float
    mean_answer_f1: float
    mean_citation_validity_rate: float
    test_results: list[TestCaseResult]


class PipelineEvaluator:
    def __init__(self) -> None:
        self.retrieval_eval = RetrievalEvaluator()
        self.context_eval = ContextEvaluator()
        self.generation_eval = GenerationEvaluator(
            grounding_validator=get_grounding_validator()
        )
        self.citation_eval = CitationEvaluator()

    def evaluate_dataset(
        self,
        dataset_path: str | Path,
        limit_contexts: int = 5,
    ) -> EvaluationScorecard:
        path = Path(dataset_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        test_cases = data.get("test_cases", [])
        logger.info("Evaluating %d test cases from %s", len(test_cases), path)

        search_service = get_hybrid_search()
        parent_expander = get_parent_expander()
        context_selector = get_context_selector()
        context_compressor = get_context_compressor()
        generation_service = get_generation_service()

        results: list[TestCaseResult] = []

        for tc in test_cases:
            query_id = tc["query_id"]
            query = tc["query"]
            category = tc.get("category", "general")
            relevant_titles = tc.get("relevant_doc_titles", [])
            relevant_keywords = tc.get("relevant_keywords", [])
            expected_answer = tc.get("expected_answer", "")

            # 1. Retrieval
            reranked = search_service.search(
                query=query,
                limit=limit_contexts,
                retrieval_limit=max(limit_contexts * 4, 20),
            )

            retrieval_metrics = self.retrieval_eval.evaluate_query(
                query_id=query_id,
                query=query,
                retrieved_items=reranked,
                relevant_titles=relevant_titles,
                relevant_keywords=relevant_keywords,
                total_ground_truth_relevant=max(len(relevant_titles), 1),
            )

            # 2. Context expansion & selection & compression
            expanded = parent_expander.expand(reranked)
            selected = context_selector.select(expanded)
            compressed = context_compressor.compress(
                query=query,
                contexts=selected,
            )

            context_metrics = self.context_eval.evaluate(
                query_id=query_id,
                selected_contexts=selected,
                compressed_contexts=compressed,
                relevant_keywords=relevant_keywords,
                max_budget=context_compressor.max_tokens,
            )

            # 3. Generation
            if compressed:
                gen_result = generation_service.generate(
                    query=query,
                    contexts=compressed,
                )
                generated_answer = gen_result.answer
            else:
                generated_answer = "I could not find this information in the provided documentation."

            # 4. Generation evaluation
            contexts_text = [getattr(c, "content", "") for c in compressed]
            generation_metrics = self.generation_eval.evaluate(
                query_id=query_id,
                generated_answer=generated_answer,
                expected_answer=expected_answer,
                contexts=contexts_text,
            )

            # 5. Citation evaluation
            citation_metrics = self.citation_eval.evaluate(
                query_id=query_id,
                answer=generated_answer,
                context_count=len(compressed),
            )

            results.append(
                TestCaseResult(
                    query_id=query_id,
                    query=query,
                    category=category,
                    retrieval=retrieval_metrics,
                    context=context_metrics,
                    generation=generation_metrics,
                    citation=citation_metrics,
                    generated_answer=generated_answer,
                    expected_answer=expected_answer,
                )
            )

        n = max(len(results), 1)

        scorecard = EvaluationScorecard(
            total_queries=len(results),
            mean_mrr=round(sum(r.retrieval.mrr for r in results) / n, 4),
            mean_hit_rate_at_1=round(
                sum(r.retrieval.hit_rate_at_1 for r in results) / n, 4
            ),
            mean_hit_rate_at_5=round(
                sum(r.retrieval.hit_rate_at_5 for r in results) / n, 4
            ),
            mean_precision_at_5=round(
                sum(r.retrieval.precision_at_5 for r in results) / n, 4
            ),
            mean_recall_at_5=round(
                sum(r.retrieval.recall_at_5 for r in results) / n, 4
            ),
            mean_ndcg_at_5=round(
                sum(r.retrieval.ndcg_at_5 for r in results) / n, 4
            ),
            mean_compression_ratio=round(
                sum(r.context.compression_ratio for r in results) / n, 4
            ),
            mean_keyword_retention=round(
                sum(r.context.keyword_retention for r in results) / n, 4
            ),
            mean_faithfulness_score=round(
                sum(r.generation.faithfulness_score for r in results) / n, 4
            ),
            grounding_pass_rate=round(
                sum(1 for r in results if r.generation.grounded) / n, 4
            ),
            mean_answer_f1=round(
                sum(r.generation.token_f1 for r in results) / n, 4
            ),
            mean_citation_validity_rate=round(
                sum(r.citation.valid_citation_rate for r in results) / n, 4
            ),
            test_results=results,
        )

        return scorecard

    def generate_markdown_report(self, scorecard: EvaluationScorecard) -> str:
        lines = [
            "# RAG Pipeline Evaluation Scorecard",
            "",
            f"**Total Evaluated Queries:** {scorecard.total_queries}",
            "",
            "## 1. Summary Metrics",
            "",
            "| Dimension | Metric | Score |",
            "|---|---|---|",
            f"| **Retrieval** | MRR | `{scorecard.mean_mrr:.4f}` |",
            f"| **Retrieval** | HitRate@1 | `{scorecard.mean_hit_rate_at_1:.4f}` |",
            f"| **Retrieval** | HitRate@5 | `{scorecard.mean_hit_rate_at_5:.4f}` |",
            f"| **Retrieval** | Precision@5 | `{scorecard.mean_precision_at_5:.4f}` |",
            f"| **Retrieval** | Recall@5 | `{scorecard.mean_recall_at_5:.4f}` |",
            f"| **Retrieval** | nDCG@5 | `{scorecard.mean_ndcg_at_5:.4f}` |",
            f"| **Context** | Token Compression Ratio | `{scorecard.mean_compression_ratio:.4f}` |",
            f"| **Context** | Keyword Retention | `{scorecard.mean_keyword_retention:.4f}` |",
            f"| **Generation** | Faithfulness Score | `{scorecard.mean_faithfulness_score:.4f}` |",
            f"| **Generation** | Grounding Pass Rate | `{scorecard.grounding_pass_rate:.4f}` |",
            f"| **Generation** | Answer Token F1 | `{scorecard.mean_answer_f1:.4f}` |",
            f"| **Citations** | Citation Validity Rate | `{scorecard.mean_citation_validity_rate:.4f}` |",
            "",
            "## 2. Detailed Per-Query Results",
            "",
            "| Query ID | Category | MRR | Recall@5 | Comp Ratio | Faithfulness | Answer F1 | Valid Citations |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for r in scorecard.test_results:
            lines.append(
                f"| `{r.query_id}` | {r.category} | {r.retrieval.mrr:.2f} | "
                f"{r.retrieval.recall_at_5:.2f} | {r.context.compression_ratio:.2f} | "
                f"{r.generation.faithfulness_score:.2f} | {r.generation.token_f1:.2f} | "
                f"{r.citation.valid_citation_rate:.2f} |"
            )

        return "\n".join(lines)
