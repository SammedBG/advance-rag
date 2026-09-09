from datetime import datetime
import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.rate_limiter import rate_limit
from app.core.security import UserPrincipal, get_current_user
from app.services.container import ServiceContainer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversations",
    tags=["conversations"],
    dependencies=[Depends(rate_limit(requests_per_minute=60))],
)


class CreateConversationRequest(BaseModel):
    title: str | None = None
    conversation_id: str | None = None


class AddMessageRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1)
    citations: list[int] = Field(default_factory=list)
    grounding_score: float | None = None


def _serialize_conversation(conv: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(conv)
    for key in ["created_at", "updated_at"]:
        if isinstance(serialized.get(key), datetime):
            serialized[key] = serialized[key].isoformat()
    return serialized


def _serialize_message(msg: dict[str, Any]) -> dict[str, Any]:
    serialized = dict(msg)
    if isinstance(serialized.get("created_at"), datetime):
        serialized["created_at"] = serialized["created_at"].isoformat()
    return serialized


@router.post("", status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: CreateConversationRequest,
    current_user: UserPrincipal = Depends(get_current_user),
) -> dict[str, Any]:
    container = ServiceContainer()
    conv_repo = container.get_conversation_repository()

    cid = conv_repo.create_conversation(
        conversation_id=request.conversation_id,
        title=request.title,
        tenant_id=current_user.tenant_id,
        user_id=current_user.user_id,
    )
    conv = conv_repo.get_conversation(cid)
    return _serialize_conversation(conv) if conv else {"conversation_id": cid}


@router.get("")
def list_conversations(
    current_user: UserPrincipal = Depends(get_current_user),
) -> list[dict[str, Any]]:
    container = ServiceContainer()
    conv_repo = container.get_conversation_repository()

    if "admin" in current_user.roles and current_user.tenant_id == "default":
        convs = conv_repo.list_conversations()
    else:
        convs = conv_repo.list_conversations(tenant_id=current_user.tenant_id)

    return [_serialize_conversation(c) for c in convs]


@router.get("/{conversation_id}")
def get_conversation(
    conversation_id: str,
    current_user: UserPrincipal = Depends(get_current_user),
) -> dict[str, Any]:
    container = ServiceContainer()
    conv_repo = container.get_conversation_repository()

    conv = conv_repo.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    # Multi-tenancy check
    if "admin" not in current_user.roles and conv.get("tenant_id") != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to conversation of another tenant.",
        )

    messages = conv_repo.get_raw_messages(conversation_id)
    serialized_conv = _serialize_conversation(conv)
    serialized_conv["messages"] = [_serialize_message(m) for m in messages]
    return serialized_conv


@router.post("/{conversation_id}/messages", status_code=status.HTTP_201_CREATED)
def add_message(
    conversation_id: str,
    request: AddMessageRequest,
    current_user: UserPrincipal = Depends(get_current_user),
) -> dict[str, Any]:
    container = ServiceContainer()
    conv_repo = container.get_conversation_repository()

    conv = conv_repo.get_conversation(conversation_id)
    if conv and "admin" not in current_user.roles and conv.get("tenant_id") != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot add messages to conversation of another tenant.",
        )

    msg = conv_repo.add_message(
        conversation_id=conversation_id,
        role=request.role,
        content=request.content,
        citations=request.citations,
        grounding_score=request.grounding_score,
    )
    return _serialize_message(msg)


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    current_user: UserPrincipal = Depends(get_current_user),
) -> dict[str, Any]:
    container = ServiceContainer()
    conv_repo = container.get_conversation_repository()

    conv = conv_repo.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found.",
        )

    if "admin" not in current_user.roles and conv.get("tenant_id") != current_user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete conversation of another tenant.",
        )

    deleted = conv_repo.delete_conversation(conversation_id)
    return {"deleted": deleted, "conversation_id": conversation_id}
