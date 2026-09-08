from pathlib import Path

from app.models.document import Document


class DocumentLoader:

    SUPPORTED_EXTENSIONS = {
        ".md",
        ".txt",
    }

    def load(self, file_path: str) -> Document:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        content = path.read_text(
            encoding="utf-8"
        )

        return Document(
            document_id=path.stem,
            source=str(path),
            source_type=extension.lstrip("."),
            title=path.stem,
            content=content,
        )