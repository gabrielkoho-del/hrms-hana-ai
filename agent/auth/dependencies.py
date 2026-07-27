# agent/auth/dependencies.py
"""FastAPI dependency for token verification."""
import os
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .models import AuthContext, validate_jwt
from ..config import DEV_TEST_TOKEN, AUTH_MODE
import logging

logger = logging.getLogger("hr_agent")
security = HTTPBearer(auto_error=False)


async def extract_token(request: Request) -> Optional[str]:
    """Extract Bearer token from Authorization header, or None."""
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.replace("Bearer ", "")
    return None


def get_fallback_token() -> Optional[str]:
    """Return dev test token for local development when no header is present."""
    return DEV_TEST_TOKEN or None


async def verify_token(request: Request) -> AuthContext:
    """
    Dual-mode auth dependency for FastAPI endpoints.

    - Production (AUTH_MODE=production): Requires valid Authorization header.
    - Test (AUTH_MODE=test): Falls back to DEV_TEST_TOKEN from .env if no header.
    - Disabled (AUTH_MODE=disabled): Returns anonymous context with NO permissions.
    """
    if AUTH_MODE == "disabled":
        ctx = AuthContext()
        ctx.authenticated = True
        ctx.tenant_id = "anonymous"
        ctx.email = "anonymous"
        # NO permissions granted — user must be explicitly granted access
        return ctx

    token = await extract_token(request)

    if not token and AUTH_MODE == "test":
        token = get_fallback_token()
        if token:
            logger.info("[Auth] Using DEV_TEST_TOKEN fallback")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header or DEV_TEST_TOKEN not configured",
            headers={"WWW-Authenticate": "Bearer"},
        )

    ctx = validate_jwt(token)

    if not ctx.authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("[Auth] %s @ %s → roles %s → %d perms",
                ctx.email, ctx.tenant_id, ctx.internal_roles, len(ctx.permissions))

    return ctx