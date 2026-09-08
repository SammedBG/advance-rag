from fastapi import APIRouter, HTTPException

from app.retrieval.search import VectorSearch
from app.services.container import (
    get_embedding_service,
    get_qdrant_service,
)


router = APIRouter(
    prefix="/search",
    tags=["search"],
)


@router.get("")
def search(
    query: str,
    limit: int = 5,
):

    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100.",
        )

    try:

        search_service = VectorSearch(
            embedding_service=(
                get_embedding_service()
            ),
            qdrant_service=(
                get_qdrant_service()
            ),
        )

        results = search_service.search(
            query=query,
            limit=limit,
        )

        return {
            "query": query,
            "results": [
                {
                    "score": result.score,
                    "chunk_id": result.payload.get(
                        "chunk_id"
                    ),
                    "document_id": result.payload.get(
                        "document_id"
                    ),
                    "parent_id": result.payload.get(
                        "parent_id"
                    ),
                    "title": result.payload.get(
                        "title"
                    ),
                    "heading_path": result.payload.get(
                        "heading_path"
                    ),
                    "content": result.payload.get(
                        "content"
                    ),
                    "metadata": result.payload.get(
                        "metadata"
                    ),
                }
                for result in results
            ],
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc