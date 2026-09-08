from pathlib import Path

from app.models.document import Document


class MetadataExtractor:

    def extract(
        self,
        document: Document,
    ) -> Document:

        path = Path(document.source)

        document.source_type = (
            path.suffix.lower().lstrip(".")
        )

        document.metadata.update(
            {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "source": document.source,
                "source_type": document.source_type,
                "document_id": document.document_id,
                "title": document.title,
                "version": document.version,
                "technology": document.technology,
                "technology_version": (
                    document.technology_version
                ),
                "access_scope": document.access_scope,
            }
        )

        return document