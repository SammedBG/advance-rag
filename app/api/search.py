from fastapi import APIRouter, HTTPException

from app.context.compressor import ContextCompressor
from app.context.parent_expander import ParentExpander
from app.context.selector import ContextSelector
from app.generation.citation import CitationValidator
from app.generation.grounding import GroundingValidator
from app.retrieval.hybrid import HybridSearch
from app.retrieval.rrf import ReciprocalRankFusion
from app.services.container import (
    get_bm25_index,
    get_context_compressor,
    get_context_selector,
    get_embedding_service,
    get_generation_service,
    get_grounding_validator,
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

        parent_expander: ParentExpander = (
            get_parent_expander()
        )

        expanded_results = parent_expander.expand(
            reranked_results
        )

        context_selector: ContextSelector = (
            get_context_selector()
        )

        selected_contexts = context_selector.select(
            expanded_results
        )

        context_compressor: ContextCompressor = (
            get_context_compressor()
        )

        compressed_contexts = context_compressor.compress(
            query=query,
            contexts=selected_contexts,
        )

        if not compressed_contexts:
            return {
                "query": query,
                "answer": (
                    "I could not find this information "
                    "in the provided documentation."
                ),
                "citations": [],
                "citation_validation": {
                    "valid": False,
                    "invalid_citations": [],
                    "missing_citations": True,
                },
                "grounding": {
                    "score": 0.0,
                    "grounded": False,
                    "matched_terms": [],
                    "unmatched_terms": [],
                },
                "results": [],
                "generation": None,
                "context_stats": {
                    "contexts_selected": len(
                        selected_contexts
                    ),
                    "contexts_compressed": 0,
                    "selected_tokens": sum(
                        context.token_count
                        for context in selected_contexts
                    ),
                    "compressed_tokens": 0,
                    "compression_ratio": 0.0,
                    "max_tokens": (
                        context_compressor.max_tokens
                    ),
                },
            }

        generation_service = get_generation_service()

        generated_answer = generation_service.generate(
            query=query,
            contexts=compressed_contexts,
        )

        citation_validator = CitationValidator()

        citation_validation = citation_validator.validate(
            answer=generated_answer.answer,
            context_count=len(compressed_contexts),
        )

        grounding_validator: GroundingValidator = (
            get_grounding_validator()
        )

        grounding_validation = grounding_validator.validate(
            answer=generated_answer.answer,
            contexts=[
                context.content
                for context in compressed_contexts
            ],
        )

        return {
            "query": query,
            "answer": generated_answer.answer,
            "citations": citation_validation.citations,
            "citation_validation": {
                "valid": citation_validation.valid,
                "invalid_citations": (
                    citation_validation.invalid_citations
                ),
                "missing_citations": (
                    citation_validation.missing_citations
                ),
            },
            "grounding": {
                "score": grounding_validation.score,
                "grounded": grounding_validation.grounded,
                "matched_terms": (
                    grounding_validation.matched_terms
                ),
                "unmatched_terms": (
                    grounding_validation.unmatched_terms
                ),
            },
            "results": [
                {
                    "source": index,
                    "score": context.retrieval_score,
                    "chunk_id": context.chunk_id,
                    "parent_id": context.parent_id,
                    "document_id": context.document_id,
                    "title": context.title,
                    "heading_path": context.heading_path,
                    "content": context.content,
                    "dense_rank": context.dense_rank,
                    "sparse_rank": context.sparse_rank,
                    "original_token_count": (
                        context.original_token_count
                    ),
                    "compressed_token_count": (
                        context.compressed_token_count
                    ),
                }
                for index, context in enumerate(
                    compressed_contexts,
                    start=1,
                )
            ],
            "generation": {
                "model": generated_answer.model,
                "input_tokens": generated_answer.input_tokens,
                "output_tokens": generated_answer.output_tokens,
                "total_tokens": generated_answer.total_tokens,
            },
            "context_stats": {
                "contexts_selected": len(
                    selected_contexts
                ),
                "contexts_compressed": len(
                    compressed_contexts
                ),
                "selected_tokens": sum(
                    context.token_count
                    for context in selected_contexts
                ),
                "compressed_tokens": sum(
                    context.compressed_token_count
                    for context in compressed_contexts
                ),
                "compression_ratio": round(
                    sum(
                        context.compressed_token_count
                        for context in compressed_contexts
                    )
                    / max(
                        sum(
                            context.token_count
                            for context in selected_contexts
                        ),
                        1,
                    ),
                    3,
                ),
                "max_tokens": (
                    context_compressor.max_tokens
                ),
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc