from app.models.chunk import DocumentChunk


class ChunkRepository:
    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def save_many(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

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

    def clear(self) -> None:
        self._chunks.clear()

    def count(self) -> int:
        return len(self._chunks)