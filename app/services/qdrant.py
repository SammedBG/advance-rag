from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.models.chunk import DocumentChunk


class QdrantService:
    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.vector_size = vector_size

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections()

        collection_names = {
            collection.name
            for collection in collections.collections
        }

        if self.collection_name in collection_names:
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                "Number of chunks and vectors must be identical."
            )

        points: list[PointStruct] = []

        for chunk, vector in zip(chunks, vectors):
            payload = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type,
                "parent_id": chunk.parent_id,
                "title": chunk.title,
                "content": chunk.content,
                "heading_path": chunk.heading_path,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata,
            }

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        vector: list[float],
        limit: int = 10,
    ):
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        return response.points