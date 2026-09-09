import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import (
    Base,
    ChunkModel,
    ConversationModel,
    DocumentModel,
    MessageModel,
    QueryLogModel,
)


@pytest.mark.anyio
async def test_db_models_crud_in_memory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        # 1. Insert Document
        doc = DocumentModel(
            document_id="doc_test_1",
            source="test.md",
            source_type="markdown",
            title="Database Test Doc",
            content="This is the content of the document.",
            technology="postgresql",
            metadata_json={"env": "test"},
        )
        session.add(doc)

        # 2. Insert Chunk
        chunk = ChunkModel(
            chunk_id="chunk_test_1",
            document_id="doc_test_1",
            chunk_type="child",
            parent_id=None,
            chunk_index=0,
            title="Database Test Doc",
            content="This is the chunk content.",
            heading_path=["Database Test Doc", "Overview"],
            token_count=12,
            metadata_json={"type": "child"},
        )
        session.add(chunk)

        # 3. Insert Conversation & Message
        conv = ConversationModel(conversation_id="conv_1", title="Test Conversation")
        session.add(conv)

        msg = MessageModel(
            message_id="msg_1",
            conversation_id="conv_1",
            role="user",
            content="What is this document about?",
        )
        session.add(msg)

        # 4. Insert Query Log
        qlog = QueryLogModel(
            log_id="log_1",
            query="test query",
            route="rag",
            latency_ms=12.5,
            chunks_retrieved=3,
            grounding_score=0.85,
            tokens_used=150,
        )
        session.add(qlog)

        await session.commit()

    # Query Back & Verify
    async with session_factory() as session:
        result_doc = await session.get(DocumentModel, "doc_test_1")
        assert result_doc is not None
        assert result_doc.title == "Database Test Doc"
        assert result_doc.technology == "postgresql"

        result_chunk = await session.get(ChunkModel, "chunk_test_1")
        assert result_chunk is not None
        assert result_chunk.document_id == "doc_test_1"
        assert result_chunk.heading_path == ["Database Test Doc", "Overview"]

        result_conv = await session.get(ConversationModel, "conv_1")
        assert result_conv is not None
        assert result_conv.title == "Test Conversation"

        result_log = await session.get(QueryLogModel, "log_1")
        assert result_log is not None
        assert result_log.latency_ms == 12.5
        assert result_log.grounding_score == 0.85

    await engine.dispose()
