from app.ingestion.chunker import StructureAwareChunker
from app.models.document import Document


def test_markdown_and_plaintext_headings():
    chunker = StructureAwareChunker(max_tokens=200, overlap_tokens=20)

    content = """# Main Title
This is the introduction text.

SECTION 1: Installation Guide
Here are the steps to install the system:
Step 1: run pip install.

2. Architecture Overview
The system consists of three main components:
Ingestion, Retrieval, Generation.
"""

    doc = Document(
        document_id="doc-test-1",
        source="test.md",
        source_type="markdown",
        title="Test Heading Document",
        content=content,
    )

    chunks = chunker.chunk(doc)
    assert len(chunks) > 0

    parents = [c for c in chunks if c.chunk_type == "parent"]
    assert len(parents) == 3

    assert "Main Title" in parents[0].heading_path
    assert any("SECTION 1: Installation Guide" in p.heading_path for p in parents)
    assert any("2. Architecture Overview" in p.heading_path for p in parents)


def test_numbered_subsections():
    chunker = StructureAwareChunker(max_tokens=200, overlap_tokens=20)

    content = """1. First Section
Top level info.

1.1 Subsection A
Details on subsection A.

1.2 Subsection B
Details on subsection B.
"""

    doc = Document(
        document_id="doc-test-2",
        source="test.txt",
        source_type="text",
        title="Numbered Document",
        content=content,
    )

    chunks = chunker.chunk(doc)
    parents = [c for c in chunks if c.chunk_type == "parent"]
    assert len(parents) == 3
