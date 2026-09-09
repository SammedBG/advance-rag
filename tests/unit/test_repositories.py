from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.document_repository import DocumentRepository


def test_document_repository():
    repo = DocumentRepository()

    doc = Document(
        document_id="doc_1",
        source="k8s.md",
        source_type="markdown",
        title="K8s Doc",
        content="Pod troubleshooting.",
    )

    repo.save(doc)
    assert repo.count() == 1
    assert repo.get("doc_1") is not None
    assert repo.get("doc_1").title == "K8s Doc"

    docs = repo.list_all()
    assert len(docs) == 1

    deleted = repo.delete("doc_1")
    assert deleted is True
    assert repo.count() == 0


def test_chunk_repository_parent_expansion():
    repo = ChunkRepository()

    parent = DocumentChunk(
        chunk_id="p1",
        document_id="doc1",
        content="Parent chunk containing full section.",
        title="Doc 1",
        chunk_index=0,
        chunk_type="parent",
        token_count=100,
    )

    child = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        content="Child chunk.",
        title="Doc 1",
        chunk_index=1,
        chunk_type="child",
        parent_id="p1",
        token_count=20,
    )

    repo.save_many([parent, child])
    assert repo.count() == 2

    # Verify parent lookup
    found_parent = repo.get_parent(child)
    assert found_parent is not None
    assert found_parent.chunk_id == "p1"

    # Verify document chunks filter
    doc_chunks = repo.list_by_document("doc1")
    assert len(doc_chunks) == 2


def test_conversation_repository():
    repo = ConversationRepository()

    cid = repo.create_conversation(title="Debugging Pods")
    assert cid is not None
    assert repo.count() == 1

    # Add user message
    msg1 = repo.add_message(
        conversation_id=cid,
        role="user",
        content="Why is my pod crashing?",
    )
    assert msg1["role"] == "user"

    # Add assistant message
    msg2 = repo.add_message(
        conversation_id=cid,
        role="assistant",
        content="Check CrashLoopBackOff causes.",
        citations=[1],
        grounding_score=0.9,
    )
    assert msg2["role"] == "assistant"

    # Fetch history
    history = repo.get_history(cid)
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "assistant"
