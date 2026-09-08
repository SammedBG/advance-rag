from datetime import datetime

from pydantic import BaseModel, Field


class Document(BaseModel):
    document_id: str

    source: str
    source_type: str

    title: str
    content: str

    url: str | None = None

    version: str | None = None

    technology: str | None = None
    technology_version: str | None = None

    access_scope: str = "public"

    metadata: dict = Field(default_factory=dict)

    created_at: datetime | None = None
    updated_at: datetime | None = None