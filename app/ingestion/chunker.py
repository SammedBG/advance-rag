import re
from uuid import uuid4

from app.models.chunk import DocumentChunk
from app.models.document import Document


class StructureAwareChunker:

    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")

    def chunk(self, document: Document) -> list[DocumentChunk]:
        lines = document.content.splitlines()

        chunks: list[DocumentChunk] = []

        heading_stack: list[str] = []
        current_content: list[str] = []

        def flush_chunk():
            if not current_content:
                return

            content = "\n".join(current_content).strip()

            if not content:
                return

            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid4()),
                    document_id=document.document_id,
                    content=content,
                    title=document.title,
                    heading_path=heading_stack.copy(),
                    chunk_index=len(chunks),
                )
            )

            current_content.clear()

        for line in lines:

            match = self.HEADING_PATTERN.match(line)

            if match:
                flush_chunk()

                level = len(match.group(1))
                heading = match.group(2).strip()

                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading)

                continue

            current_content.append(line)

        flush_chunk()

        return chunks