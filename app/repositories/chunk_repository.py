import logging
from app.models.chunk import DocumentChunk

logger = logging.getLogger(__name__)


class ChunkRepository:
    """
    Repository for managing DocumentChunk entities.
    Provides fast lookup for parent chunk expansion and chunk retrieval.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def save(self, chunk: DocumentChunk) -> None:
        self._chunks[chunk.chunk_id] = chunk

    def save_many(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        logger.debug("Saved %d chunks to repository.", len(chunks))

    def get(self, chunk_id: str) -> DocumentChunk | None:
        return self._chunks.get(chunk_id)

    def get_parent(self, chunk: DocumentChunk) -> DocumentChunk | None:
        if not chunk.parent_id:
            return None
        return self.get(chunk.parent_id)

    def get_many(self, chunk_ids: list[str]) -> list[DocumentChunk]:
        return [
            self._chunks[chunk_id]
            for chunk_id in chunk_ids
            if chunk_id in self._chunks
        ]

    def list_by_document(self, document_id: str) -> list[DocumentChunk]:
        return [
            chunk
            for chunk in self._chunks.values()
            if chunk.document_id == document_id
        ]

    def get_by_document(self, document_id: str) -> list[DocumentChunk]:
        return self.list_by_document(document_id)

    def list_all(self) -> list[DocumentChunk]:
        return list(self._chunks.values())

    def get_all(self) -> list[DocumentChunk]:
        return self.list_all()

    def delete(self, chunk_id: str) -> bool:
        if chunk_id in self._chunks:
            del self._chunks[chunk_id]
            return True
        return False

    def clear(self) -> None:
        self._chunks.clear()

    def count(self) -> int:
        return len(self._chunks)