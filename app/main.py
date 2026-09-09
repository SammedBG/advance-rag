from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import time
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.indexing import router as indexing_router
from app.api.metrics import router as metrics_router
from app.api.search import router as search_router
from app.core.logging import setup_logging
from app.core.metrics import metrics_registry
from app.services.container import ServiceContainer

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure BM25 sparse index and vector index are initialized
    logger.info("Initializing Advanced RAG Platform services...")
    container = ServiceContainer()
    bm25 = container.get_bm25_index()
    chunk_repo = container.get_chunk_repository()

    child_chunks = [c for c in chunk_repo.get_all() if c.chunk_type == "child"]
    if child_chunks and bm25.chunk_count == 0:
        logger.info("Synchronizing BM25 sparse index from persistent ChunkRepository (%d chunks)...", len(child_chunks))
        bm25.build(child_chunks)
    elif bm25.chunk_count == 0:
        raw_dir = Path("data/raw")
        if raw_dir.exists():
            indexer = container.get_indexer()
            raw_files = list(raw_dir.glob("*.md")) + list(raw_dir.glob("*.txt"))
            for f in raw_files:
                try:
                    indexer.index_file(str(f))
                except Exception as exc:
                    logger.warning("Startup index error for '%s': %s", f, exc)

    logger.info("Startup complete. BM25 active with %d chunks.", bm25.chunk_count)
    yield
    logger.info("Shutting down Advanced RAG Platform...")


app = FastAPI(
    title="Advanced RAG Platform",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next) -> Response:
    start_time = time.perf_counter()
    endpoint = request.url.path
    method = request.method

    try:
        response = await call_next(request)
        status_code = str(response.status_code)
    except Exception as exc:
        status_code = "500"
        raise exc from None
    finally:
        duration = time.perf_counter() - start_time
        metrics_registry.increment_counter(
            "rag_http_requests_total",
            value=1.0,
            labels={"endpoint": endpoint, "method": method, "status": status_code},
        )
        metrics_registry.observe_histogram(
            "rag_http_request_duration_seconds",
            value=duration,
            labels={"endpoint": endpoint},
        )

    return response


# API Routers
app.include_router(indexing_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(metrics_router)

# Static files mount
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Advanced RAG Platform API is active."}


@app.get("/dashboard")
def dashboard():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Dashboard UI not found."}


@app.get("/health")
def health():
    return {
        "status": "ok",
    }