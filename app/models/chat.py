from typing import Literal
from pydantic import BaseModel, Field

from app.models.response import (
    CitationValidationResult,
    ContextStats,
    GenerationStats,
    GroundingResult,
    SearchResultItem,
)
from app.models.search import SearchFilter


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Latest user message or question")
    conversation_id: str | None = Field(default=None, description="Optional conversation tracking ID")
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversational messages for context",
    )
    limit: int = Field(default=5, ge=1, le=100, description="Max contexts to retrieve")
    filters: SearchFilter | None = Field(default=None, description="Metadata filters")


class ChatResponse(BaseModel):
    conversation_id: str
    query: str
    answer: str
    citations: list[int] = Field(default_factory=list)
    citation_validation: CitationValidationResult = Field(
        default_factory=CitationValidationResult
    )
    grounding: GroundingResult = Field(default_factory=GroundingResult)
    sources: list[SearchResultItem] = Field(default_factory=list)
    generation: GenerationStats | None = None
    context_stats: ContextStats = Field(default_factory=ContextStats)
