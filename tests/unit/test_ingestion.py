from app.ingestion.pipeline import IngestionPipeline


def test_document_ingestion_and_chunking():

    pipeline = IngestionPipeline()

    document, chunks = pipeline.process(
        "data/raw/kubernetes.md"
    )

    assert document.title == "kubernetes"

    assert len(chunks) > 0

    crashloop_chunk = next(
        chunk
        for chunk in chunks
        if "CrashLoopBackOff" in chunk.content
    )

    assert "Pod Restarting" in crashloop_chunk.heading_path
    assert "CrashLoopBackOff" in crashloop_chunk.heading_path