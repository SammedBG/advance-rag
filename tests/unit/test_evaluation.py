from app.evaluation.citations import CitationEvaluator
from app.evaluation.context import ContextEvaluator
from app.evaluation.generation import GenerationEvaluator
from app.evaluation.retrieval import RetrievalEvaluator
from app.models.chunk import DocumentChunk


def test_retrieval_evaluator_metrics():
    evaluator = RetrievalEvaluator()

    chunk1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        content="CrashLoopBackOff occurs when a pod restarts repeatedly.",
        title="Kubernetes Troubleshooting",
        chunk_index=0,
        chunk_type="child",
        token_count=10,
    )

    chunk2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc2",
        content="Docker multi-stage builds optimize image size.",
        title="Docker Guide",
        chunk_index=0,
        chunk_type="child",
        token_count=10,
    )

    metrics = evaluator.evaluate_query(
        query_id="q1",
        query="What is CrashLoopBackOff?",
        retrieved_items=[chunk1, chunk2],
        relevant_titles=["Kubernetes Troubleshooting"],
        relevant_keywords=["crashloopbackoff", "pod", "restarts"],
        total_ground_truth_relevant=1,
    )

    assert metrics.mrr == 1.0
    assert metrics.hit_rate_at_1 == 1.0
    assert metrics.hit_rate_at_5 == 1.0
    assert metrics.precision_at_1 == 1.0
    assert metrics.precision_at_3 == 0.5
    assert metrics.recall_at_1 == 1.0
    assert metrics.ndcg_at_5 == 1.0


def test_context_evaluator():
    evaluator = ContextEvaluator()

    chunk1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        content="CrashLoopBackOff occurs when a pod restarts repeatedly due to crashes.",
        title="Kubernetes Troubleshooting",
        chunk_index=0,
        chunk_type="child",
        token_count=20,
    )

    metrics = evaluator.evaluate(
        query_id="q1",
        selected_contexts=[chunk1],
        compressed_contexts=[chunk1],
        relevant_keywords=["crashloopbackoff", "restarts"],
    )

    assert metrics.selected_count == 1
    assert metrics.compressed_count == 1
    assert metrics.keyword_retention == 1.0
    assert metrics.budget_compliant is True


def test_generation_evaluator_token_f1():
    evaluator = GenerationEvaluator()

    generated = "CrashLoopBackOff occurs when a container repeatedly crashes and restarts."
    expected = "CrashLoopBackOff occurs when a pod repeatedly starts and crashes."

    p, r, f1 = evaluator.compute_token_f1(generated, expected)
    assert f1 > 0.6
    assert p > 0.0
    assert r > 0.0


def test_citation_evaluator():
    evaluator = CitationEvaluator()

    answer = "According to the docs [1], pods restart upon failure."
    metrics = evaluator.evaluate(query_id="q1", answer=answer, context_count=2)

    assert metrics.total_citations == 1
    assert metrics.valid_citations == 1
    assert metrics.valid_citation_rate == 1.0
    assert metrics.has_hallucinated_citations is False
