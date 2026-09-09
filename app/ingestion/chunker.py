import re
from uuid import uuid5, NAMESPACE_URL

from app.ingestion.tokenizer import TokenCounter
from app.models.chunk import DocumentChunk
from app.models.document import Document


class StructureAwareChunker:
    MD_HEADING_PATTERN = re.compile(
        r"^(#{1,6})\s+(.+)$"
    )
    KEYWORD_HEADING_PATTERN = re.compile(
        r"^(SECTION|CHAPTER|PART|MODULE|APPENDIX)\s*([0-9A-Za-z\.\-_]*)\s*[:\-–—]\s*(.+)$",
        re.IGNORECASE,
    )
    NUMBERED_HEADING_PATTERN = re.compile(
        r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9\s\-_,:;()'\"]{2,80})$"
    )

    def __init__(
        self,
        max_tokens: int = 500,
        overlap_tokens: int = 75,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self.token_counter = TokenCounter()

    def chunk(
        self,
        document: Document,
    ) -> list[DocumentChunk]:

        sections = self._extract_sections(
            document
        )

        chunks: list[DocumentChunk] = []

        for section_index, section in enumerate(sections):

            # Deterministic parent ID.
            # Same document + same section = same ID.
            parent_id = self._make_id(
                document.document_id,
                "parent",
                section_index,
            )

            parent_metadata = (
                self._build_metadata(
                    document=document,
                    chunk_type="parent",
                    chunk_id=parent_id,
                    parent_id=None,
                    heading_path=section[
                        "heading_path"
                    ],
                )
            )

            parent = DocumentChunk(
                chunk_id=parent_id,
                document_id=document.document_id,
                content=section["content"],
                title=document.title,
                heading_path=section[
                    "heading_path"
                ],
                chunk_index=len(chunks),
                chunk_type="parent",
                parent_id=None,
                token_count=self.token_counter.count(
                    section["content"]
                ),
                metadata=parent_metadata,
            )

            chunks.append(parent)

            child_chunks = (
                self._split_into_children(
                    document=document,
                    content=section["content"],
                    heading_path=section[
                        "heading_path"
                    ],
                    parent_id=parent_id,
                    starting_index=len(chunks),
                    section_index=section_index,
                )
            )

            chunks.extend(child_chunks)

        return chunks

    def _extract_sections(
        self,
        document: Document,
    ) -> list[dict]:

        lines = document.content.splitlines()

        sections: list[dict] = []

        heading_stack: list[str] = []

        current_content: list[str] = []

        def flush():

            if not current_content:
                return

            content = "\n".join(
                current_content
            ).strip()

            if content:
                sections.append(
                    {
                        "content": content,
                        "heading_path": (
                            heading_stack.copy()
                        ),
                    }
                )

            current_content.clear()

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.strip()

            if not stripped:
                current_content.append(line)
                i += 1
                continue

            # Markdown heading (# Heading)
            match = self.MD_HEADING_PATTERN.match(stripped)
            if match:
                flush()
                level = len(match.group(1))
                heading = match.group(2).strip()
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading)
                i += 1
                continue

            # Setext underlines (Title \n === or ---)
            if i + 1 < n and len(stripped) <= 100:
                next_stripped = lines[i + 1].strip()
                if len(next_stripped) >= 3 and all(c == "=" for c in next_stripped):
                    flush()
                    heading_stack = [stripped]
                    i += 2
                    continue
                elif (
                    len(next_stripped) >= 3
                    and all(c == "-" for c in next_stripped)
                    and not stripped.startswith("-")
                ):
                    flush()
                    heading_stack = heading_stack[:1]
                    heading_stack.append(stripped)
                    i += 2
                    continue

            # Keyword heading (SECTION 1: Overview, CHAPTER: Setup)
            kw_match = self.KEYWORD_HEADING_PATTERN.match(stripped)
            if kw_match:
                flush()
                heading = stripped
                heading_stack = heading_stack[:1]
                heading_stack.append(heading)
                i += 1
                continue

            # Numbered heading (1. Introduction, 1.2 System Details)
            num_match = self.NUMBERED_HEADING_PATTERN.match(stripped)
            if num_match:
                flush()
                dots = num_match.group(1).count(".")
                level = min(dots + 1, 6)
                heading = stripped
                heading_stack = heading_stack[: level - 1]
                heading_stack.append(heading)
                i += 1
                continue

            current_content.append(line)
            i += 1

        flush()

        if not sections and current_content:
            content = "\n".join(current_content).strip()
            if content:
                sections.append(
                    {
                        "content": content,
                        "heading_path": (
                            [document.title] if document.title else []
                        ),
                    }
                )

        return sections

    def _split_into_children(
        self,
        document: Document,
        content: str,
        heading_path: list[str],
        parent_id: str,
        starting_index: int,
        section_index: int,
    ) -> list[DocumentChunk]:

        words = content.split()

        children: list[DocumentChunk] = []

        current_words: list[str] = []

        current_tokens = 0

        for word in words:

            word_tokens = (
                self.token_counter.count(word)
            )

            if (
                current_words
                and current_tokens + word_tokens
                > self.max_tokens
            ):

                child_content = (
                    " ".join(current_words)
                )

                child_index = len(children)

                child_id = self._make_id(
                    document.document_id,
                    "child",
                    section_index,
                    child_index,
                )

                child_metadata = (
                    self._build_metadata(
                        document=document,
                        chunk_type="child",
                        chunk_id=child_id,
                        parent_id=parent_id,
                        heading_path=heading_path,
                    )
                )

                children.append(
                    DocumentChunk(
                        chunk_id=child_id,
                        document_id=(
                            document.document_id
                        ),
                        content=child_content,
                        title=document.title,
                        heading_path=(
                            heading_path.copy()
                        ),
                        chunk_index=(
                            starting_index
                            + child_index
                        ),
                        chunk_type="child",
                        parent_id=parent_id,
                        token_count=current_tokens,
                        metadata=child_metadata,
                    )
                )

                overlap_words = (
                    self._get_overlap(
                        current_words
                    )
                )

                current_words = overlap_words

                current_tokens = (
                    self.token_counter.count(
                        " ".join(
                            current_words
                        )
                    )
                )

            current_words.append(word)

            current_tokens += word_tokens

        if current_words:

            child_content = (
                " ".join(current_words)
            )

            child_index = len(children)

            child_id = self._make_id(
                document.document_id,
                "child",
                section_index,
                child_index,
            )

            child_metadata = (
                self._build_metadata(
                    document=document,
                    chunk_type="child",
                    chunk_id=child_id,
                    parent_id=parent_id,
                    heading_path=heading_path,
                )
            )

            children.append(
                DocumentChunk(
                    chunk_id=child_id,
                    document_id=(
                        document.document_id
                    ),
                    content=child_content,
                    title=document.title,
                    heading_path=(
                        heading_path.copy()
                    ),
                    chunk_index=(
                        starting_index
                        + child_index
                    ),
                    chunk_type="child",
                    parent_id=parent_id,
                    token_count=current_tokens,
                    metadata=child_metadata,
                )
            )

        return children

    def _get_overlap(
        self,
        words: list[str],
    ) -> list[str]:

        overlap: list[str] = []

        token_count = 0

        for word in reversed(words):

            tokens = (
                self.token_counter.count(word)
            )

            if (
                token_count + tokens
                > self.overlap_tokens
            ):
                break

            overlap.insert(
                0,
                word,
            )

            token_count += tokens

        return overlap

    @staticmethod
    def _make_id(
        document_id: str,
        chunk_type: str,
        section_index: int,
        child_index: int | None = None,
    ) -> str:

        if child_index is None:

            value = (
                f"{document_id}:"
                f"{chunk_type}:"
                f"{section_index}"
            )

        else:

            value = (
                f"{document_id}:"
                f"{chunk_type}:"
                f"{section_index}:"
                f"{child_index}"
            )

        return str(
            uuid5(
                NAMESPACE_URL,
                value,
            )
        )

    def _build_metadata(
        self,
        document: Document,
        chunk_type: str,
        chunk_id: str,
        parent_id: str | None,
        heading_path: list[str],
    ) -> dict:

        return {
            # Document identity
            "document_id": document.document_id,
            "document_version": document.version,

            # Source
            "source": document.source,
            "source_type": document.source_type,
            "url": document.url,

            # Document information
            "title": document.title,

            # Technology
            "technology": document.technology,
            "technology_version": (
                document.technology_version
            ),

            # Structure
            "heading_path": heading_path,
            "section": (
                heading_path[-1]
                if heading_path
                else None
            ),

            # Chunk relationship
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "parent_id": parent_id,

            # Security
            "access_scope": (
                document.access_scope
            ),

            # Custom metadata
            **document.metadata,
        }