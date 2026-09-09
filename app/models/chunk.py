from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    tenant_id: str = "default"

    content: str
    title: str

    heading_path: list[str] = Field(default_factory=list)

    chunk_index: int
    chunk_type: str
    parent_id: str | None = None
    token_count: int

    metadata: dict = Field(default_factory=dict)