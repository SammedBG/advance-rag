import json
from pathlib import Path
import pytest
import tempfile

from app.core.config import settings
from app.ingestion.loader import DocumentLoader
from app.ingestion.indexer import compute_content_hash, IngestionIndexer
from app.ingestion.pipeline import IngestionPipeline
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.retrieval.bm25 import BM25Index
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


def test_json_document_loading():
    loader = DocumentLoader()
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+", delete=False, encoding="utf-8") as f:
        json_data = {
            "title": "PostgreSQL Architecture Guide",
            "overview": "PostgreSQL is an advanced open-source relational database.",
            "replication": "Streaming replication enables high availability.",
        }
        json.dump(json_data, f)
        temp_path = f.name

    doc = loader.load(temp_path)
    assert doc.document_id in temp_path
    assert doc.source_type == "json"
    assert "Streaming replication" in doc.content
    assert "# PostgreSQL Architecture Guide" in doc.content
    Path(temp_path).unlink()


def test_html_document_loading():
    loader = DocumentLoader()
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w+", delete=False, encoding="utf-8") as f:
        html_data = """
        <!DOCTYPE html>
        <html>
        <head><title>Docker Container Best Practices</title></head>
        <body>
            <script>console.log("ignore me");</script>
            <h1>Docker Optimizations</h1>
            <p>Use multi-stage builds to minimize image sizes.</p>
            <h2>Security</h2>
            <p>Run containers as non-root users.</p>
        </body>
        </html>
        """
        f.write(html_data)
        temp_path = f.name

    doc = loader.load(temp_path)
    assert doc.document_id in temp_path
    assert doc.source_type == "html"
    assert "console.log" not in doc.content
    assert "multi-stage builds" in doc.content
    Path(temp_path).unlink()


def test_content_hash_and_stale_chunk_deletion():
    pipeline = IngestionPipeline()
    embedding_svc = EmbeddingService(model_name=settings.embedding_model)
    qdrant_svc = QdrantService(
        url=":memory:",
        collection_name="test_ingestion_hash",
        vector_size=embedding_svc.dimension,
    )
    bm25 = BM25Index()
    chunk_repo = ChunkRepository()
    doc_repo = DocumentRepository()

    indexer = IngestionIndexer(
        ingestion_pipeline=pipeline,
        embedding_service=embedding_svc,
        qdrant_service=qdrant_svc,
        bm25_index=bm25,
        chunk_repository=chunk_repo,
        document_repository=doc_repo,
    )

    loader = DocumentLoader()

    # Initial document version (3 sections)
    doc_v1 = loader.load_text(
        title="Microservices Design",
        content="# Microservices Design\n\n## Section 1\nService mesh basics.\n\n## Section 2\nAPI Gateway routing.\n\n## Section 3\nEvent streaming.",
    )

    doc, chunks_v1 = indexer.index_document(doc_v1)
    assert len(chunks_v1) > 0
    assert doc_repo.count() == 1
    assert chunk_repo.count() == len(chunks_v1)

    # Re-indexing with exact same content -> should detect hash match and skip re-embedding
    doc_v1_dup = loader.load_text(
        title="Microservices Design",
        content="# Microservices Design\n\n## Section 1\nService mesh basics.\n\n## Section 2\nAPI Gateway routing.\n\n## Section 3\nEvent streaming.",
    )
    doc_same, chunks_dup = indexer.index_document(doc_v1_dup)
    assert len(chunks_dup) == len(chunks_v1)

    # Document version 2 (reduced to 1 section) -> stale chunks must be deleted
    doc_v2 = loader.load_text(
        title="Microservices Design",
        content="# Microservices Design\n\n## Section 1\nService mesh basics only.",
    )
    doc_updated, chunks_v2 = indexer.index_document(doc_v2, force=True)
    assert len(chunks_v2) < len(chunks_v1)
    
    # Verify stale chunks were purged from chunk_repo
    stored_chunks = chunk_repo.get_by_document("microservices_design")
    assert len(stored_chunks) == len(chunks_v2)
