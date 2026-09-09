from fastapi import APIRouter, Depends, HTTPException

from app.core.rate_limiter import rate_limit
from app.core.security import UserPrincipal, get_current_user
from app.services.container import get_rag_agent

router = APIRouter(
    prefix="/agent",
    tags=["agent"],
    dependencies=[Depends(rate_limit())],
)


@router.get("")
def run_agent(
    query: str,
    user: UserPrincipal = Depends(get_current_user),
):
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