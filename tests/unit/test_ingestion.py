from app.ingestion.pipeline import IngestionPipeline


def test_document_ingestion():
    pipeline = IngestionPipeline()

    document = pipeline.process(
        "data/raw/kubernetes.md"
    )

    assert document.title == "kubernetes"
    assert document.file_type == "md"
    assert "CrashLoopBackOff" in document.content