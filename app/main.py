from fastapi import FastAPI

from app.api.indexing import router as indexing_router
from app.api.search import router as search_router


app = FastAPI(
    title="Advanced RAG",
    version="0.2.0",
)


app.include_router(
    indexing_router
)

app.include_router(
    search_router
)


@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "advanced-rag",
    }