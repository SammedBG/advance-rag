from fastapi import APIRouter, HTTPException

from app.context.parent_expander import ParentExpander
from app.retrieval.hybrid import HybridSearch
from app.retrieval.rrf import ReciprocalRankFusion
from app.services.container import (
    get_bm25_index,
    get_embedding_service,
    get_parent_expander,
    get_qdrant_service,
    get_reranker,
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
        search_service = HybridSearch(
            embedding_service=get_embedding_service(),
            qdrant_service=get_qdrant_service(),
            bm25_index=get_bm25_index(),
            rrf=ReciprocalRankFusion(),
            reranker=get_reranker(),
        )

        reranked_results = search_service.search(
            query=query,
            limit=limit,
            retrieval_limit=max(limit * 4, 20),
        )

        parent_expander: ParentExpander = get_parent_expander()

        expanded_results = parent_expander.expand(
            reranked_results
        )

        return {
            "query": query,
            "results": [
                {
                    "score": result.retrieval_score,
                    "chunk_id": result.chunk_id,
                    "parent_id": result.parent_id,
                    "document_id": result.document_id,
                    "title": result.title,
                    "heading_path": result.heading_path,
                    "content": result.content,
                    "dense_rank": result.dense_rank,
                    "sparse_rank": result.sparse_rank,
                }
                for result in expanded_results
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc