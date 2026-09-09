from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.loader import DocumentLoader
from app.ingestion.metadata import MetadataExtractor
from app.models.chunk import DocumentChunk
from app.models.document import Document


class IngestionPipeline:
    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 75,
    ):
        self.loader = DocumentLoader()
        self.cleaner = DocumentCleaner()
        self.metadata_extractor = MetadataExtractor()
        self.chunker = StructureAwareChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

    def process(
        self,
        file_path: str,
    ) -> tuple[Document, list[DocumentChunk]]:
        # 1. Load
        document = self.loader.load(file_path)
        return self.process_document(document)

    def process_document(
        self,
        document: Document,
    ) -> tuple[Document, list[DocumentChunk]]:
        # 2. Clean
        document = self.cleaner.clean(document)

        # 3. Extract metadata
        document = self.metadata_extractor.extract(document)

        # 4. Chunk
        chunks = self.chunker.chunk(document)

        return document, chunks