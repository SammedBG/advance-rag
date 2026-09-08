from fastapi import APIRouter, HTTPException

from app.ingestion.indexer import IngestionIndexer
from app.services.container import (
    get_bm25_index,
    get_embedding_service,
    get_ingestion_pipeline,
    get_qdrant_service,
)


router = APIRouter(
    prefix="/index",
    tags=["indexing"],
)


@router.post("")
def index_document(file_path: str):
    try:
        indexer = IngestionIndexer(
            ingestion_pipeline=get_ingestion_pipeline(),
            embedding_service=get_embedding_service(),
            qdrant_service=get_qdrant_service(),
            bm25_index=get_bm25_index(),
        )

        document, chunks = indexer.index_file(file_path)

        return {
            "status": "indexed",
            "document_id": document.document_id,
            "chunks_indexed": len(chunks),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc