import re

from app.models.document import Document


class DocumentCleaner:

    def clean(
        self,
        document: Document,
    ) -> Document:

        content = document.content

        # Normalize line endings
        content = content.replace(
            "\r\n",
            "\n",
        )

        content = content.replace(
            "\r",
            "\n",
        )

        # Remove trailing whitespace
        content = "\n".join(
            line.rstrip()
            for line in content.splitlines()
        )

        # Prevent excessive blank lines
        content = re.sub(
            r"\n{3,}",
            "\n\n",
            content,
        )

        document.content = content.strip()

        return document