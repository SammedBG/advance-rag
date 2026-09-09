from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import pytest

from app.core.security import create_jwt_token
from app.generation.llm import GroqLLM, LLMResponse
from app.generation.prompt_builder import PromptBuilder
from app.generation.service import GenerationService
from app.main import app
from app.models.chunk import DocumentChunk
from app.context.compressor import CompressedContext


def test_groq_llm_generate_stream_fallback():
    # Instantiate GroqLLM with dummy credentials
    llm = GroqLLM(api_key="gsk_test123", model="openai/gpt-oss-120b")

    # Mock the client chat completion to simulate streaming
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock(delta=MagicMock(content="World!"))]

    llm.client.chat.completions.create = MagicMock(return_value=[mock_chunk_1, mock_chunk_2])

    chunks = list(llm.generate_stream(system_prompt="System", user_prompt="User"))
    assert len(chunks) == 2
    assert "".join(chunks) == "Hello World!"


def test_generation_service_stream():
    llm = GroqLLM(api_key="gsk_test123", model="openai/gpt-oss-120b")
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="FastAPI dependency injection."))]
    llm.client.chat.completions.create = MagicMock(return_value=[mock_chunk])

    prompt_builder = PromptBuilder()
    service = GenerationService(llm=llm, prompt_builder=prompt_builder)

    context = CompressedContext(
        chunk_id="c1",
        parent_id=None,
        document_id="doc1",
        title="FastAPI Guide",
        heading_path=["DI"],
        content="Depends is used for dependency injection.",
        dense_rank=1,
        sparse_rank=1,
        retrieval_score=0.9,
        original_token_count=10,
        compressed_token_count=10,
    )

    stream_chunks = list(service.generate_stream(
        query="How does DI work?",
        contexts=[context],
    ))
    assert len(stream_chunks) >= 1
    assert "FastAPI dependency injection." in "".join(stream_chunks)


def test_chat_stream_endpoint():
    client = TestClient(app)
    token = create_jwt_token(user_id="user_stream", roles=["user"], scopes=["public"])

    response = client.post(
        "/chat/stream",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "What causes CrashLoopBackOff in Kubernetes?"},
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    content = response.text
    assert "data: " in content
    assert "event" in content
