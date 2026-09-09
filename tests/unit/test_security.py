from fastapi import HTTPException
import pytest

from app.core.config import settings
from app.core.security import (
    UserPrincipal,
    create_jwt_token,
    decode_jwt_token,
    get_current_user,
    require_scope,
)


def test_jwt_token_generation_and_decoding():
    token = create_jwt_token(
        user_id="user_123",
        roles=["analyst"],
        scopes=["public", "internal"],
        expires_minutes=30,
    )
    assert token is not None

    payload = decode_jwt_token(token)
    assert payload["sub"] == "user_123"
    assert payload["roles"] == ["analyst"]
    assert payload["scopes"] == ["public", "internal"]


def test_jwt_token_expired():
    token = create_jwt_token(
        user_id="user_expired",
        expires_minutes=-5,  # already expired
    )
    with pytest.raises(HTTPException) as exc_info:
        decode_jwt_token(token)
    assert exc_info.value.status_code == 401
    assert "expired" in exc_info.value.detail.lower()


def test_api_key_authentication():
    # Valid key
    user = get_current_user(x_api_key="dev-api-key-12345")
    assert user.is_authenticated is True
    assert "admin" in user.scopes

    # Invalid key
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(x_api_key="invalid-bogus-key")
    assert exc_info.value.status_code == 401


def test_bearer_token_authentication():
    token = create_jwt_token(
        user_id="jwt_user_1",
        roles=["reader"],
        scopes=["public"],
    )
    user = get_current_user(authorization=f"Bearer {token}")
    assert user.user_id == "jwt_user_1"
    assert user.roles == ["reader"]
    assert user.scopes == ["public"]


def test_scope_enforcement():
    checker = require_scope("confidential")

    # User with required scope
    user_ok = UserPrincipal(user_id="u1", roles=["user"], scopes=["public", "confidential"])
    res = checker(user_ok)
    assert res == user_ok

    # Admin user passes all scopes
    admin_user = UserPrincipal(user_id="admin_1", roles=["admin"], scopes=["public"])
    assert checker(admin_user) == admin_user

    # User missing scope -> 403 Forbidden
    user_denied = UserPrincipal(user_id="u2", roles=["user"], scopes=["public"])
    with pytest.raises(HTTPException) as exc_info:
        checker(user_denied)
    assert exc_info.value.status_code == 403
