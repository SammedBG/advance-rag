from fastapi import APIRouter, HTTPException

from app.services.container import get_rag_agent


router = APIRouter(
    prefix="/agent",
    tags=["agent"],
)


@router.get("")
def run_agent(query: str):
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        agent = get_rag_agent()

        return agent.run(query)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc