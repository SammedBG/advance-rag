from typing import Any
from pydantic import BaseModel, Field


class CitationValidationResult(BaseModel):
    valid: bool = False
    invalid_citations: list[int] = Field(default_factory=list)
    missing_citations: bool = False


class GroundingResult(BaseModel):
    score: float = 0.0
    grounded: bool = False
    matched_terms: list[str] = Field(default_factory=list)
    unmatched_terms: list[str] = Field(default_factory=list)


class ContextStats(BaseModel):
    contexts_selected: int = 0
    contexts_compressed: int = 0
    selected_tokens: int = 0
    compressed_tokens: int = 0
    compression_ratio: float = 0.0
    max_tokens: int = 0


class GenerationStats(BaseModel):
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class SearchResultItem(BaseModel):
    source: int
    score: float
    chunk_id: str
    parent_id: str | None = None
    document_id: str
    title: str
    heading_path: list[str] = Field(default_factory=list)
    content: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    original_token_count: int = 0
    compressed_token_count: int = 0


class SearchResponse(BaseModel):
    query: str
    answer: str
    citations: list[int] = Field(default_factory=list)
    citation_validation: CitationValidationResult = Field(
        default_factory=CitationValidationResult
    )
    grounding: GroundingResult = Field(default_factory=GroundingResult)
    results: list[SearchResultItem] = Field(default_factory=list)
    generation: GenerationStats | None = None
    context_stats: ContextStats = Field(default_factory=ContextStats)
