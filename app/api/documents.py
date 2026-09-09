from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.rate_limiter import rate_limit
from app.core.security import UserPrincipal, get_current_user, require_scope
from app.ingestion.worker import global_job_manager
from app.services.container import ServiceContainer

router = APIRouter(prefix="/documents", tags=["documents"])


class AsyncDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    technology: str | None = None
    access_scope: str = "public"


class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str


@router.post(
    "/async",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobResponse,
    dependencies=[Depends(rate_limit(requests_per_minute=30))],
)
def enqueue_document_indexing(
    request: AsyncDocumentRequest,
    background_tasks: BackgroundTasks,
    current_user: UserPrincipal = Depends(require_scope("internal")),
) -> JobResponse:
    """
    Asynchronously enqueue a document for structure-aware chunking, embedding, and indexing.
    """
    job = global_job_manager.create_job(
        title=request.title,
        tenant_id=current_user.tenant_id,
    )

    background_tasks.add_task(
        global_job_manager.run_indexing_task,
        job_id=job.job_id,
        title=request.title,
        content=request.content,
        tenant_id=current_user.tenant_id,
        technology=request.technology,
        access_scope=request.access_scope,
    )

    return JobResponse(
        job_id=job.job_id,
        status=job.status.value,
        message="Document indexing job submitted successfully.",
    )


@router.get("/jobs/{job_id}")
def get_job_status(
    job_id: str,
    current_user: UserPrincipal = Depends(get_current_user),
) -> dict:
    """
    Poll the status of an asynchronous indexing job.
    """
    job = global_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    # Multi-tenancy check
    if "admin" not in current_user.roles and job.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to job for another tenant.",
        )

    return job.to_dict()


@router.get("")
def list_documents(
    current_user: UserPrincipal = Depends(get_current_user),
) -> list[dict]:
    """
    List all indexed documents for the current user's tenant.
    """
    container = ServiceContainer()
    doc_repo = container.get_document_repository()

    if "admin" in current_user.roles and current_user.tenant_id == "default":
        docs = doc_repo.list_all()
    else:
        docs = doc_repo.list_by_tenant(current_user.tenant_id)

    return [doc.model_dump() for doc in docs]


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    current_user: UserPrincipal = Depends(require_scope("admin")),
) -> dict:
    """
    Delete a document and all its chunks from the repository.
    """
    container = ServiceContainer()
    doc_repo = container.get_document_repository()
    chunk_repo = container.get_chunk_repository()

    doc = doc_repo.get(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    if "admin" not in current_user.roles and doc.tenant_id != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete document belonging to another tenant.",
        )

    # Remove chunks
    chunks = chunk_repo.list_by_document(document_id)
    for c in chunks:
        chunk_repo.delete(c.chunk_id)

    doc_repo.delete(document_id)
    return {
        "deleted": True,
        "document_id": document_id,
        "chunks_removed": len(chunks),
    }
