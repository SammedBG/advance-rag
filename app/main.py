from fastapi import FastAPI

app = FastAPI(
    title="Advanced RAG",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "advanced-rag",
    }