import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.core.rate_limiter import rate_limit
from app.core.security import UserPrincipal, get_current_user
from app.generation.citation import CitationValidator
from app.models.chat import ChatRequest, ChatResponse
from app.models.response import (
    CitationValidationResult,
    ContextStats,
    GenerationStats,
    GroundingResult,
    SearchResultItem,
)
from app.models.search import SearchFilter
from app.services.container import (
    get_context_compressor,
    get_context_selector,
    get_conversation_repository,
    get_generation_service,
    get_grounding_validator,
    get_hybrid_search,
    get_parent_expander,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(rate_limit())],
)


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: UserPrincipal = Depends(get_current_user),
) -> ChatResponse:
    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    conv_repo = get_conversation_repository()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    # Enforce RBAC access scope if not admin
    filters = request.filters
    if "admin" not in user.roles and user.scopes:
        if filters is None:
            filters = SearchFilter(access_scope=user.scopes[0])
        elif not filters.access_scope:
            filters.access_scope = user.scopes[0]

    try:
        # 1. Fetch prior history from repository if not explicitly passed
        history = request.history
        if not history and conversation_id:
            history = conv_repo.get_history(conversation_id)

        # 2. Record User Message
        conv_repo.add_message(
            conversation_id=conversation_id,
            role="user",
            content=request.query,
        )

        logger.info(
            "Chat request: conversation_id=%s, query='%s', history_len=%d user=%s",
            conversation_id,
            request.query,
            len(history),
            user.user_id,
        )

        effective_query = request.query
        if history:
            effective_query = f"{history[-1].content} {request.query}"

        search_service = get_hybrid_search()

        reranked_results = search_service.search(
            query=effective_query,
            limit=request.limit,
            retrieval_limit=max(request.limit * 4, 20),
            filters=filters,
        )

        parent_expander = get_parent_expander()
        expanded_results = parent_expander.expand(reranked_results)

        context_selector = get_context_selector()
        selected_contexts = context_selector.select(expanded_results)

        context_compressor = get_context_compressor()
        compressed_contexts = context_compressor.compress(
            query=request.query,
            contexts=selected_contexts,
        )

        if not compressed_contexts:
            fallback_answer = "I could not find this information in the provided documentation."
            conv_repo.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=fallback_answer,
                citations=[],
                grounding_score=0.0,
            )
            return ChatResponse(
                conversation_id=conversation_id,
                query=request.query,
                answer=fallback_answer,
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
                sources=[],
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

        generation_service = get_generation_service()
        generated_answer = generation_service.generate(
            query=request.query,
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

        source_items = [
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

        # 3. Record Assistant Message in Repository
        conv_repo.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=generated_answer.answer,
            citations=citation_validation.citations,
            grounding_score=grounding_validation.score,
        )

        return ChatResponse(
            conversation_id=conversation_id,
            query=request.query,
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
            sources=source_items,
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

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Chat endpoint failure: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
