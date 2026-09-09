from app.models.chunk import DocumentChunk
from app.models.search import SearchFilter
from app.retrieval.filters import MetadataFilterBuilder


def test_metadata_filters_predicate():
    chunk1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        content="Kubernetes pod crash troubleshooting",
        title="K8s Guide",
        chunk_index=0,
        chunk_type="child",
        token_count=10,
        metadata={"technology": "kubernetes", "source_type": "markdown"},
    )

    chunk2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc2",
        content="FastAPI async background tasks",
        title="FastAPI Guide",
        chunk_index=0,
        chunk_type="child",
        token_count=10,
        metadata={"technology": "fastapi", "source_type": "python"},
    )

    # Filter by technology
    filter_tech = SearchFilter(technology="kubernetes")
    pred_tech = MetadataFilterBuilder.build_chunk_predicate(filter_tech)
    assert pred_tech(chunk1) is True
    assert pred_tech(chunk2) is False

    # Filter by document_id
    filter_doc = SearchFilter(document_ids=["doc2"])
    pred_doc = MetadataFilterBuilder.build_chunk_predicate(filter_doc)
    assert pred_doc(chunk1) is False
    assert pred_doc(chunk2) is True


def test_qdrant_filter_builder():
    filter_obj = SearchFilter(
        document_ids=["doc1", "doc2"],
        technology="python",
        custom={"category": "tutorial"},
    )
    qdrant_filter = MetadataFilterBuilder.build_qdrant_filter(filter_obj)
    assert qdrant_filter is not None
    assert len(qdrant_filter.must) == 3
