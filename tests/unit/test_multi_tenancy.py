from fastapi.testclient import TestClient
import pytest

from app.core.security import UserPrincipal, create_jwt_token, get_current_user
from app.main import app
from app.models.chunk import DocumentChunk
from app.models.document import Document
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.container import ServiceContainer


def test_document_repository_tenant_isolation():
    repo = DocumentRepository()

    doc_a = Document(
        document_id="doc_tenant_a",
        source="tenant_a.md",
        source_type="markdown",
        title="Tenant A Secrets",
        content="Confidential info for A.",
        tenant_id="tenant_a",
    )
    doc_b = Document(
        document_id="doc_tenant_b",
        source="tenant_b.md",
        source_type="markdown",
        title="Tenant B Secrets",
        content="Confidential info for B.",
        tenant_id="tenant_b",
    )

    repo.save_many([doc_a, doc_b])

    docs_a = repo.list_by_tenant("tenant_a")
    assert len(docs_a) == 1
    assert docs_a[0].document_id == "doc_tenant_a"
    assert docs_a[0].tenant_id == "tenant_a"

    docs_b = repo.list_by_tenant("tenant_b")
    assert len(docs_b) == 1
    assert docs_b[0].document_id == "doc_tenant_b"
    assert docs_b[0].tenant_id == "tenant_b"

    all_docs = repo.list_all()
    assert len(all_docs) == 2


def test_chunk_repository_tenant_isolation():
    repo = ChunkRepository()

    chunk_a = DocumentChunk(
        chunk_id="chunk_a_1",
        document_id="doc_a",
        content="Tenant A content",
        title="Doc A",
        chunk_index=0,
        chunk_type="child",
        tenant_id="tenant_a",
    )
    chunk_b = DocumentChunk(
        chunk_id="chunk_b_1",
        document_id="doc_b",
        content="Tenant B content",
        title="Doc B",
        chunk_index=0,
        chunk_type="child",
        tenant_id="tenant_b",
    )

    repo.save_many([chunk_a, chunk_b])

    chunks_a = repo.list_by_tenant("tenant_a")
    assert len(chunks_a) == 1
    assert chunks_a[0].chunk_id == "chunk_a_1"

    chunks_b = repo.list_by_tenant("tenant_b")
    assert len(chunks_b) == 1
    assert chunks_b[0].chunk_id == "chunk_b_1"


def test_security_tenant_header_and_jwt():
    # Test x-tenant-id header with API key
    user = get_current_user(x_api_key="dev-api-key-12345", x_tenant_id="acme_corp")
    assert user.tenant_id == "acme_corp"
    assert user.is_authenticated is True

    # Test JWT token with tenant_id claim
    token = create_jwt_token(
        user_id="user_tenant_x",
        roles=["user"],
        scopes=["public"],
        tenant_id="tenant_xyz",
    )
    jwt_user = get_current_user(authorization=f"Bearer {token}")
    assert jwt_user.tenant_id == "tenant_xyz"
    assert jwt_user.user_id == "user_tenant_x"


def test_documents_api_tenant_filtering():
    container = ServiceContainer()
    doc_repo = container.get_document_repository()
    doc_repo.clear()

    doc_a = Document(
        document_id="api_doc_a",
        source="a.md",
        source_type="md",
        title="Tenant A Doc",
        content="A content",
        tenant_id="tenant_alpha",
    )
    doc_b = Document(
        document_id="api_doc_b",
        source="b.md",
        source_type="md",
        title="Tenant B Doc",
        content="B content",
        tenant_id="tenant_beta",
    )
    doc_repo.save_many([doc_a, doc_b])

    client = TestClient(app)

    # Token for Tenant Alpha user
    token_a = create_jwt_token(
        user_id="user_a",
        roles=["user"],
        scopes=["public"],
        tenant_id="tenant_alpha",
    )
    resp_a = client.get("/documents", headers={"Authorization": f"Bearer {token_a}"})
    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert len(data_a) == 1
    assert data_a[0]["document_id"] == "api_doc_a"

    # Token for Tenant Beta user
    token_b = create_jwt_token(
        user_id="user_b",
        roles=["user"],
        scopes=["public"],
        tenant_id="tenant_beta",
    )
    resp_b = client.get("/documents", headers={"Authorization": f"Bearer {token_b}"})
    assert resp_b.status_code == 200
    data_b = resp_b.json()
    assert len(data_b) == 1
    assert data_b[0]["document_id"] == "api_doc_b"
