import pytest

from app.query.models import StructuredQuery
from app.query.transformation import QueryTransformation


def test_hyde_passage_generation():
    transformer = QueryTransformation()

    # Troubleshooting intent
    passage_troubleshoot = transformer.generate_hyde_passage(
        query="Why is pod in CrashLoopBackOff?",
        intent="troubleshooting",
    )
    assert "CrashLoopBackOff" in passage_troubleshoot
    assert "root causes" in passage_troubleshoot.lower() or "diagnosing" in passage_troubleshoot.lower()

    # Comparison intent
    passage_comp = transformer.generate_hyde_passage(
        query="FastAPI vs Flask",
        intent="comparison",
    )
    assert "differences" in passage_comp.lower() or "comparison" in passage_comp.lower()

    # Configuration intent
    passage_config = transformer.generate_hyde_passage(
        query="How to configure Redis eviction policy?",
        intent="configuration",
    )
    assert "configure" in passage_config.lower() or "parameters" in passage_config.lower()


def test_query_decomposition():
    transformer = QueryTransformation()

    decomposed = transformer.decompose_complex_query(
        "Explain Redis memory optimization and also how to configure cache-aside pattern"
    )
    assert len(decomposed) >= 2
    assert any("redis memory optimization" in q.lower() for q in decomposed)
    assert any("configure cache-aside pattern" in q.lower() for q in decomposed)


def test_transform_with_hyde_enabled():
    transformer = QueryTransformation()
    structured = StructuredQuery(
        original_query="What is Redis eviction policy?",
        intent="definition",
        keywords=["redis", "eviction", "policy"],
        technical_terms=["redis", "eviction"],
        entities=["redis"],
        filters={},
    )

    # With HyDE enabled
    result = transformer.transform(structured, enable_hyde=True)
    assert len(result.retrieval_queries) > len(transformer.transform(structured, enable_hyde=False).retrieval_queries)
    assert any("Overview and documentation" in q for q in result.retrieval_queries)
