from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Callable

from fastapi import Depends, Header, HTTPException, status
import jwt
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)


class UserPrincipal(BaseModel):
    user_id: str
    tenant_id: str = "default"
    roles: list[str] = Field(default_factory=lambda: ["user"])
    scopes: list[str] = Field(default_factory=lambda: ["public"])
    is_authenticated: bool = True


def create_jwt_token(
    user_id: str,
    tenant_id: str = "default",
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
    expires_minutes: int | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    delta = timedelta(minutes=expires_minutes or settings.jwt_expiration_minutes)
    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "roles": roles or ["user"],
        "scopes": scopes or ["public"],
        "iat": now,
        "exp": now + delta,
    }
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def decode_jwt_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
        )


def get_current_user(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> UserPrincipal:
    api_key = x_api_key if isinstance(x_api_key, str) else None
    tenant = x_tenant_id if isinstance(x_tenant_id, str) else "default"
    auth = authorization if isinstance(authorization, str) else None

    # If security is disabled and no credentials provided, grant development principal
    if not settings.security_enabled and not api_key and not auth:
        return UserPrincipal(
            user_id="dev-user",
            tenant_id=tenant,
            roles=["admin"],
            scopes=["public", "internal", "confidential", "admin"],
            is_authenticated=False,
        )

    # 1. Verify API Key header
    if api_key:
        if api_key in settings.api_keys:
            return UserPrincipal(
                user_id=f"api-key-user-{api_key[:6]}",
                tenant_id=tenant,
                roles=["service"],
                scopes=["public", "internal", "confidential", "admin"],
                is_authenticated=True,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key.",
        )

    # 2. Verify Bearer JWT
    if auth:
        parts = auth.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1]
            payload = decode_jwt_token(token)
            return UserPrincipal(
                user_id=payload.get("sub", "unknown"),
                tenant_id=payload.get("tenant_id", tenant),
                roles=payload.get("roles", ["user"]),
                scopes=payload.get("scopes", ["public"]),
                is_authenticated=True,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must follow format: Bearer <token>",
        )

    # No credentials provided when security is enabled
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide 'X-API-Key' or 'Authorization: Bearer <token>'.",
    )


def require_scope(required_scope: str) -> Callable:
    def scope_checker(user: UserPrincipal = Depends(get_current_user)) -> UserPrincipal:
        if "admin" in user.roles or required_scope in user.scopes:
            return user
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Permission denied: Missing required scope '{required_scope}'.",
        )

    return scope_checker
