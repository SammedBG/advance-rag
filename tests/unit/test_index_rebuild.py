from app.core.config import settings
from app.ingestion.loader import DocumentLoader
from app.ingestion.indexer import IngestionIndexer
from app.ingestion.pipeline import IngestionPipeline
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.retrieval.bm25 import BM25Index
from app.services.embedding import EmbeddingService
from app.services.qdrant import QdrantService


def test_deterministic_bm25_index_rebuild():
    pipeline = IngestionPipeline()
    embedding_svc = EmbeddingService(model_name=settings.embedding_model)
    qdrant_svc = QdrantService(
        url=":memory:",
        collection_name="test_rebuild",
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

    # Index 2 documents
    doc1 = loader.load_text("Doc 1", "# Doc 1\n\nContent for first document.")
    doc2 = loader.load_text("Doc 2", "# Doc 2\n\nContent for second document.")

    indexer.index_document(doc1)
    indexer.index_document(doc2)

    initial_bm25_count = bm25.chunk_count
    assert initial_bm25_count > 0

    # Simulate fresh restart with an empty in-memory BM25 index
    fresh_bm25 = BM25Index()
    assert fresh_bm25.chunk_count == 0

    # Deterministically rebuild from ChunkRepository
    child_chunks = [c for c in chunk_repo.get_all() if c.chunk_type == "child"]
    fresh_bm25.build(child_chunks)

    assert fresh_bm25.chunk_count == initial_bm25_count

    # Test retrieval works identically on rebuilt index
    results = fresh_bm25.search("Content for first", limit=5)
    assert len(results) > 0
    assert results[0].chunk.document_id == "doc_1"
