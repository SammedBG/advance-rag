from datetime import datetime
from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str
    source: str
    title: str
    content: str

    file_type: str
    version: str | None = None
    url: str | None = None

    metadata: dict = Field(default_factory=dict)

    created_at: datetime | None = None
    updated_at: datetime | None = None