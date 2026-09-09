from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from app.core.security import create_jwt_token
from app.ingestion.worker import JobManager, JobStatus
from app.main import app
from app.models.chunk import DocumentChunk
from app.models.document import Document


def test_job_manager_lifecycle():
    manager = JobManager()
    job = manager.create_job(title="Kubernetes Guide", tenant_id="acme")

    assert job.job_id.startswith("job_")
    assert job.status == JobStatus.QUEUED
    assert job.tenant_id == "acme"
    assert job.title == "Kubernetes Guide"

    # Simulate running the indexing task
    mock_doc = Document(
        document_id="doc_test_123",
        source="k8s.md",
        source_type="md",
        title="Kubernetes Guide",
        content="# Kubernetes Guide\n\nPod debugging details.",
        tenant_id="acme",
    )
    mock_chunk = DocumentChunk(
        chunk_id="chunk_1",
        document_id="doc_test_123",
        content="Pod debugging details.",
        title="Kubernetes Guide",
        chunk_index=0,
        chunk_type="child",
        tenant_id="acme",
    )

    with patch("app.ingestion.worker.ServiceContainer") as mock_sc:
        mock_indexer = MagicMock()
        mock_indexer.index_document.return_value = (mock_doc, [mock_chunk])
        mock_sc.return_value.get_indexer.return_value = mock_indexer

        manager.run_indexing_task(
            job_id=job.job_id,
            title="Kubernetes Guide",
            content="# Kubernetes Guide\n\nPod debugging details.",
            tenant_id="acme",
        )

    updated_job = manager.get_job(job.job_id)
    assert updated_job is not None
    assert updated_job.status == JobStatus.COMPLETED
    assert updated_job.document_id == "doc_test_123"
    assert updated_job.chunks_indexed == 1
    assert updated_job.completed_at is not None

    job_dict = updated_job.to_dict()
    assert job_dict["job_id"] == job.job_id
    assert job_dict["status"] == "completed"
    assert "duration_seconds" in job_dict


def test_job_manager_failure_handling():
    manager = JobManager()
    job = manager.create_job(title="Faulty Doc", tenant_id="acme")

    with patch("app.ingestion.worker.ServiceContainer") as mock_sc:
        mock_indexer = MagicMock()
        mock_indexer.index_document.side_effect = RuntimeError("Qdrant connection timeout")
        mock_sc.return_value.get_indexer.return_value = mock_indexer

        manager.run_indexing_task(
            job_id=job.job_id,
            title="Faulty Doc",
            content="Some faulty content",
            tenant_id="acme",
        )

    updated_job = manager.get_job(job.job_id)
    assert updated_job is not None
    assert updated_job.status == JobStatus.FAILED
    assert "Qdrant connection timeout" in updated_job.error
    assert updated_job.completed_at is not None


def test_documents_api_async_job_submission_and_polling():
    client = TestClient(app)

    token = create_jwt_token(
        user_id="analyst_1",
        roles=["analyst"],
        scopes=["public", "internal"],
        tenant_id="tenant_gamma",
    )

    # Enqueue async document indexing
    response = client.post(
        "/documents/async",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "title": "Terraform Modules",
            "content": "# Terraform\n\nTerraform modules provide reusable infrastructure patterns.",
            "technology": "terraform",
            "access_scope": "internal",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    job_id = data["job_id"]

    # Poll job status
    poll_resp = client.get(
        f"/documents/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert poll_resp.status_code == 200
    poll_data = poll_resp.json()
    assert poll_data["job_id"] == job_id
    assert poll_data["tenant_id"] == "tenant_gamma"
    assert poll_data["status"] in ["queued", "processing", "completed"]


def test_documents_api_tenant_job_isolation():
    client = TestClient(app)

    token_tenant_a = create_jwt_token(
        user_id="user_a",
        roles=["analyst"],
        scopes=["internal"],
        tenant_id="tenant_a",
    )
    token_tenant_b = create_jwt_token(
        user_id="user_b",
        roles=["analyst"],
        scopes=["internal"],
        tenant_id="tenant_b",
    )

    # Submit job as tenant A
    response = client.post(
        "/documents/async",
        headers={"Authorization": f"Bearer {token_tenant_a}"},
        json={
            "title": "Secret Doc A",
            "content": "Tenant A exclusive data.",
        },
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    # Tenant B tries to poll Tenant A's job -> 403 Forbidden
    poll_resp = client.get(
        f"/documents/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token_tenant_b}"},
    )
    assert poll_resp.status_code == 403
    assert "access denied" in poll_resp.json()["detail"].lower()
