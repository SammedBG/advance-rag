from typing import Any
from pydantic import BaseModel, Field


class SearchFilter(BaseModel):
    document_ids: list[str] | None = None
    technology: str | None = None
    technology_version: str | None = None
    source_types: list[str] | None = None
    access_scope: str | None = None
    custom: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    limit: int = Field(default=5, ge=1, le=100, description="Maximum number of contexts to return")
    filters: SearchFilter | None = Field(default=None, description="Optional metadata filters")
