import logging
from typing import Any

from app.models.document import Document

logger = logging.getLogger(__name__)


class DocumentRepository:
    """
    Repository for managing Document entity persistence.
    Provides fast in-memory access with optional database synchronization.
    """

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}

    def save(self, document: Document) -> None:
        self._documents[document.document_id] = document
        logger.debug("Saved document '%s' to repository.", document.document_id)

    def save_many(self, documents: list[Document]) -> None:
        for doc in documents:
            self.save(doc)

    def get(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def get_by_id(self, document_id: str) -> Document | None:
        return self.get(document_id)

    def get_many(self, document_ids: list[str]) -> list[Document]:
        return [
            self._documents[doc_id]
            for doc_id in document_ids
            if doc_id in self._documents
        ]

    def list_all(self) -> list[Document]:
        return list(self._documents.values())

    def delete(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            return True
        return False

    def count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents.clear()
