from datetime import datetime, timezone
import logging
import uuid
from typing import Any

from app.models.chat import ChatMessage

logger = logging.getLogger(__name__)


class ConversationRepository:
    """
    Repository for managing conversations and message histories.
    """

    def __init__(self) -> None:
        self._conversations: dict[str, dict[str, Any]] = {}
        self._messages: dict[str, list[dict[str, Any]]] = {}

    def create_conversation(
        self,
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> str:
        cid = conversation_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        self._conversations[cid] = {
            "conversation_id": cid,
            "title": title or f"Conversation {cid[:8]}",
            "created_at": now,
            "updated_at": now,
        }
        if cid not in self._messages:
            self._messages[cid] = []
        return cid

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: list[int] | None = None,
        grounding_score: float | None = None,
    ) -> dict[str, Any]:
        if conversation_id not in self._conversations:
            self.create_conversation(conversation_id=conversation_id)

        now = datetime.now(timezone.utc)
        msg = {
            "message_id": str(uuid.uuid4()),
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "grounding_score": grounding_score,
            "created_at": now,
        }
        self._messages[conversation_id].append(msg)
        self._conversations[conversation_id]["updated_at"] = now
        return msg

    def get_history(self, conversation_id: str) -> list[ChatMessage]:
        raw_msgs = self._messages.get(conversation_id, [])
        return [
            ChatMessage(role=m["role"], content=m["content"])
            for m in raw_msgs
        ]

    def get_raw_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return list(self._messages.get(conversation_id, []))

    def list_conversations(self) -> list[dict[str, Any]]:
        return list(self._conversations.values())

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            self._messages.pop(conversation_id, None)
            return True
        return False

    def count(self) -> int:
        return len(self._conversations)

    def clear(self) -> None:
        self._conversations.clear()
        self._messages.clear()
