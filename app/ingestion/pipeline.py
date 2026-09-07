from app.ingestion.cleaner import DocumentCleaner
from app.ingestion.loader import DocumentLoader
from app.models.document import Document


class IngestionPipeline:

    def __init__(self):
        self.loader = DocumentLoader()
        self.cleaner = DocumentCleaner()

    def process(self, file_path: str) -> Document:
        document = self.loader.load(file_path)

        document = self.cleaner.clean(document)

        return document