from fastapi.testclient import TestClient
import pytest

from app.core.security import create_jwt_token
from app.main import app
from app.services.container import ServiceContainer


def test_conversations_api_crud_flow():
    client = TestClient(app)
    token = create_jwt_token(
        user_id="user_conv_1",
        roles=["user"],
        scopes=["public"],
        tenant_id="tenant_conv",
    )
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create a conversation
    create_resp = client.post(
        "/conversations",
        headers=headers,
        json={"title": "Troubleshooting Redis"},
    )
    assert create_resp.status_code == 201
    conv_data = create_resp.json()
    assert "conversation_id" in conv_data
    assert conv_data["title"] == "Troubleshooting Redis"
    cid = conv_data["conversation_id"]

    # 2. Add message to conversation
    msg_resp = client.post(
        f"/conversations/{cid}/messages",
        headers=headers,
        json={
            "role": "user",
            "content": "Why is Redis memory spiking?",
        },
    )
    assert msg_resp.status_code == 201
    msg_data = msg_resp.json()
    assert msg_data["role"] == "user"
    assert msg_data["content"] == "Why is Redis memory spiking?"

    # 3. Get conversation with messages
    get_resp = client.get(f"/conversations/{cid}", headers=headers)
    assert get_resp.status_code == 200
    fetched_data = get_resp.json()
    assert fetched_data["conversation_id"] == cid
    assert len(fetched_data["messages"]) == 1

    # 4. List conversations
    list_resp = client.get("/conversations", headers=headers)
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert any(c["conversation_id"] == cid for c in list_data)

    # 5. Delete conversation
    del_resp = client.delete(f"/conversations/{cid}", headers=headers)
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 6. Verify 404 after deletion
    get_deleted = client.get(f"/conversations/{cid}", headers=headers)
    assert get_deleted.status_code == 404


def test_conversations_api_tenant_isolation():
    client = TestClient(app)

    token_a = create_jwt_token(
        user_id="user_a",
        roles=["user"],
        scopes=["public"],
        tenant_id="tenant_alpha",
    )
    token_b = create_jwt_token(
        user_id="user_b",
        roles=["user"],
        scopes=["public"],
        tenant_id="tenant_beta",
    )

    # Tenant Alpha creates a conversation
    resp_a = client.post(
        "/conversations",
        headers={"Authorization": f"Bearer {token_a}"},
        json={"title": "Alpha Secrets"},
    )
    assert resp_a.status_code == 201
    cid_a = resp_a.json()["conversation_id"]

    # Tenant Beta tries to read Tenant Alpha's conversation -> 403 Forbidden
    resp_b = client.get(
        f"/conversations/{cid_a}",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp_b.status_code == 403
    assert "access denied" in resp_b.json()["detail"].lower()
