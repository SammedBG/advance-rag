import re
from typing import overload

from app.models.document import Document


class DocumentCleaner:
    @overload
    def clean(self, document: Document) -> Document: ...

    @overload
    def clean(self, document: str) -> str: ...

    def clean(self, document: Document | str) -> Document | str:
        if isinstance(document, Document):
            raw_text = document.content
        else:
            raw_text = str(document)

        # Normalize line endings
        content = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace
        content = "\n".join(line.rstrip() for line in content.splitlines())

        # Prevent excessive blank lines
        content = re.sub(r"\n{3,}", "\n\n", content).strip()

        if isinstance(document, Document):
            document.content = content
            return document

        return content