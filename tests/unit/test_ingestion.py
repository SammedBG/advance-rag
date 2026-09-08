from app.ingestion.pipeline import IngestionPipeline


def test_document_ingestion_and_metadata():

    pipeline = IngestionPipeline()

    document, chunks = pipeline.process(
        "data/raw/kubernetes.md"
    )

    assert document.document_id == "kubernetes"

    assert document.title == "kubernetes"

    assert document.source_type == "md"

    assert len(chunks) > 0

    child_chunks = [
        chunk
        for chunk in chunks
        if chunk.chunk_type == "child"
    ]

    assert len(child_chunks) > 0

    crashloop_chunk = next(
        chunk
        for chunk in child_chunks
        if "CrashLoopBackOff" in chunk.content
    )

    assert (
        "Pod Restarting"
        in crashloop_chunk.heading_path
    )

    assert (
        "CrashLoopBackOff"
        in crashloop_chunk.heading_path
    )

    assert (
        crashloop_chunk.metadata[
            "document_id"
        ]
        == "kubernetes"
    )

    assert (
        crashloop_chunk.metadata[
            "source_type"
        ]
        == "md"
    )

    assert (
        crashloop_chunk.metadata[
            "chunk_type"
        ]
        == "child"
    )

    assert (
        crashloop_chunk.metadata[
            "parent_id"
        ]
        == crashloop_chunk.parent_id
    )