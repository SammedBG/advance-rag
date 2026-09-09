from fastapi import APIRouter, Depends, HTTPException

from app.core.rate_limiter import rate_limit
from app.core.security import UserPrincipal, get_current_user
from app.ingestion.indexer import IngestionIndexer
from app.services.container import (
    get_bm25_index,
    get_chunk_repository,
    get_document_repository,
    get_embedding_service,
    get_ingestion_pipeline,
    get_qdrant_service,
)

router = APIRouter(
    prefix="/index",
    tags=["indexing"],
    dependencies=[Depends(rate_limit(30))],
)


@router.post("")
def index_document(
    file_path: str,
    user: UserPrincipal = Depends(get_current_user),
):
    try:
        indexer = IngestionIndexer(
            ingestion_pipeline=get_ingestion_pipeline(),
            embedding_service=get_embedding_service(),
            qdrant_service=get_qdrant_service(),
            bm25_index=get_bm25_index(),
            chunk_repository=get_chunk_repository(),
            document_repository=get_document_repository(),
        )

        document, chunks = indexer.index_file(file_path)

        return {
            "status": "indexed",
            "document_id": document.document_id,
            "chunks_indexed": len(chunks),
            "indexed_by": user.user_id,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc