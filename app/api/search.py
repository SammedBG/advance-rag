import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.config import settings
from app.core.rate_limiter import rate_limit
from app.core.security import UserPrincipal, get_current_user
from app.generation.citation import CitationValidator
from app.models.response import (
    CitationValidationResult,
    ContextStats,
    GenerationStats,
    GroundingResult,
    SearchResponse,
    SearchResultItem,
)
from app.models.search import SearchFilter, SearchRequest
from app.services.container import (
    get_cache_service,
    get_context_compressor,
    get_context_selector,
    get_generation_service,
    get_grounding_validator,
    get_hybrid_search,
    get_parent_expander,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/search",
    tags=["search"],
    dependencies=[Depends(rate_limit())],
)


def execute_search(
    query: str,
    limit: int = 5,
    filters: SearchFilter | None = None,
    user: UserPrincipal | None = None,
) -> SearchResponse:
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

    # 1. Enforce RBAC on access_scope if not admin and filter not specified
    if user and "admin" not in user.roles and user.scopes:
        if filters is None:
            filters = SearchFilter(access_scope=user.scopes[0])
        elif not filters.access_scope:
            filters.access_scope = user.scopes[0]

    # 2. Check Cache
    cache_service = get_cache_service()
    filter_dict = filters.model_dump() if filters else None

    cached_data = cache_service.get_query_cache(query=query, filters=filter_dict)
    if cached_data is not None:
        logger.info("Search query cache hit: '%s'", query)
        return SearchResponse(**cached_data)

    try:
        logger.info(
            "Search query='%s' limit=%d filters=%s user=%s",
            query,
            limit,
            filters,
            user.user_id if user else "anonymous",
        )

        search_service = get_hybrid_search()

        reranked_results = search_service.search(
            query=query,
            limit=limit,
            retrieval_limit=max(limit * 4, 20),
            filters=filters,
        )

        parent_expander = get_parent_expander()
        expanded_results = parent_expander.expand(reranked_results)

        context_selector = get_context_selector()
        selected_contexts = context_selector.select(expanded_results)

        context_compressor = get_context_compressor()
        compressed_contexts = context_compressor.compress(
            query=query,
            contexts=selected_contexts,
        )

        if not compressed_contexts:
            empty_response = SearchResponse(
                query=query,
                answer="I could not find this information in the provided documentation.",
                citations=[],
                citation_validation=CitationValidationResult(
                    valid=False,
                    invalid_citations=[],
                    missing_citations=True,
                ),
                grounding=GroundingResult(
                    score=0.0,
                    grounded=False,
                    matched_terms=[],
                    unmatched_terms=[],
                ),
                results=[],
                generation=None,
                context_stats=ContextStats(
                    contexts_selected=len(selected_contexts),
                    contexts_compressed=0,
                    selected_tokens=sum(c.token_count for c in selected_contexts),
                    compressed_tokens=0,
                    compression_ratio=0.0,
                    max_tokens=context_compressor.max_tokens,
                ),
            )
            cache_service.set_query_cache(
                query=query,
                data=empty_response.model_dump(),
                filters=filter_dict,
                ttl_seconds=settings.cache_ttl_seconds,
            )
            return empty_response

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

        grounding_validator = get_grounding_validator()
        grounding_validation = grounding_validator.validate(
            answer=generated_answer.answer,
            contexts=[c.content for c in compressed_contexts],
        )

        result_items = [
            SearchResultItem(
                source=idx,
                score=context.retrieval_score,
                chunk_id=context.chunk_id,
                parent_id=context.parent_id,
                document_id=context.document_id,
                title=context.title,
                heading_path=context.heading_path,
                content=context.content,
                dense_rank=context.dense_rank,
                sparse_rank=context.sparse_rank,
                original_token_count=context.original_token_count,
                compressed_token_count=context.compressed_token_count,
            )
            for idx, context in enumerate(compressed_contexts, start=1)
        ]

        selected_tokens = sum(c.token_count for c in selected_contexts)
        compressed_tokens = sum(c.compressed_token_count for c in compressed_contexts)
        ratio = round(compressed_tokens / max(selected_tokens, 1), 3)

        response = SearchResponse(
            query=query,
            answer=generated_answer.answer,
            citations=citation_validation.citations,
            citation_validation=CitationValidationResult(
                valid=citation_validation.valid,
                invalid_citations=citation_validation.invalid_citations,
                missing_citations=citation_validation.missing_citations,
            ),
            grounding=GroundingResult(
                score=grounding_validation.score,
                grounded=grounding_validation.grounded,
                matched_terms=grounding_validation.matched_terms,
                unmatched_terms=grounding_validation.unmatched_terms,
            ),
            results=result_items,
            generation=GenerationStats(
                model=generated_answer.model,
                input_tokens=generated_answer.input_tokens,
                output_tokens=generated_answer.output_tokens,
                total_tokens=generated_answer.total_tokens,
            ),
            context_stats=ContextStats(
                contexts_selected=len(selected_contexts),
                contexts_compressed=len(compressed_contexts),
                selected_tokens=selected_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=ratio,
                max_tokens=context_compressor.max_tokens,
            ),
        )

        # Store in cache
        cache_service.set_query_cache(
            query=query,
            data=response.model_dump(),
            filters=filter_dict,
            ttl_seconds=settings.cache_ttl_seconds,
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Search endpoint failure: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


@router.get("", response_model=SearchResponse)
def search_get(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=100),
    user: UserPrincipal = Depends(get_current_user),
) -> SearchResponse:
    return execute_search(query=query, limit=limit, filters=None, user=user)


@router.post("", response_model=SearchResponse)
def search_post(
    request: SearchRequest,
    user: UserPrincipal = Depends(get_current_user),
) -> SearchResponse:
    return execute_search(
        query=request.query,
        limit=request.limit,
        filters=request.filters,
        user=user,
    )