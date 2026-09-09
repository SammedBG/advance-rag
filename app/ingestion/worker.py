from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import time
import uuid

from app.ingestion.loader import DocumentLoader
from app.services.container import ServiceContainer

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IndexingJob:
    job_id: str
    tenant_id: str
    title: str
    status: JobStatus = JobStatus.QUEUED
    chunks_indexed: int = 0
    document_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "title": self.title,
            "status": self.status.value,
            "document_id": self.document_id,
            "chunks_indexed": self.chunks_indexed,
            "error": self.error,
            "created_at": datetime.fromtimestamp(self.created_at, timezone.utc).isoformat(),
            "completed_at": (
                datetime.fromtimestamp(self.completed_at, timezone.utc).isoformat()
                if self.completed_at
                else None
            ),
            "duration_seconds": (
                round(self.completed_at - self.created_at, 3)
                if self.completed_at
                else round(time.time() - self.created_at, 3)
            ),
        }


class JobManager:
    """
    In-memory and distributed-compatible Indexing Job Manager.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, IndexingJob] = {}

    def create_job(self, title: str, tenant_id: str = "default") -> IndexingJob:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job = IndexingJob(job_id=job_id, tenant_id=tenant_id, title=title)
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> IndexingJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self, tenant_id: str | None = None) -> list[IndexingJob]:
        if tenant_id:
            return [j for j in self._jobs.values() if j.tenant_id == tenant_id]
        return list(self._jobs.values())

    def run_indexing_task(
        self,
        job_id: str,
        title: str,
        content: str,
        tenant_id: str = "default",
        technology: str | None = None,
        access_scope: str = "public",
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        job.status = JobStatus.PROCESSING
        logger.info("Starting background indexing job %s ('%s')", job_id, title)

        try:
            container = ServiceContainer()
            indexer = container.get_indexer()
            loader = DocumentLoader()

            doc = loader.load_text(
                title=title,
                content=content,
                metadata={
                    "tenant_id": tenant_id,
                    "technology": technology,
                    "access_scope": access_scope,
                },
            )
            doc.tenant_id = tenant_id
            doc.technology = technology
            doc.access_scope = access_scope

            indexed_doc, chunks = indexer.index_document(doc)

            job.status = JobStatus.COMPLETED
            job.document_id = indexed_doc.document_id
            job.chunks_indexed = len(chunks)
            job.completed_at = time.time()
            logger.info("Background indexing job %s completed (%d chunks)", job_id, len(chunks))

        except Exception as exc:
            logger.error("Background indexing job %s failed: %s", job_id, exc)
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.completed_at = time.time()


# Global Singleton Job Manager
global_job_manager = JobManager()
