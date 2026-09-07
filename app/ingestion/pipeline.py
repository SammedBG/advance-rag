from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.loader import DocumentLoader
from app.models.chunk import DocumentChunk
from app.models.document import Document


class IngestionPipeline:

    def __init__(self):
        self.loader = DocumentLoader()
        self.cleaner = DocumentCleaner()
        self.chunker = StructureAwareChunker()

    def process(self, file_path: str) -> tuple[Document, list[DocumentChunk]]:
        document = self.loader.load(file_path)

        document = self.cleaner.clean(document)

        chunks = self.chunker.chunk(document)

        return document, chunks