# agent/auth/models.py
"""Authentication models and JWT validation logic."""
import os
import jwt
from typing import Optional, List, Set
from ..config import JWT_SECRET, JWT_ALGORITHM, JWT_AUDIENCE, JWT_ISSUERS


class AuthContext:
    """Holds resolved authentication state for a request."""

    def __init__(self):
        self.authenticated: bool = False
        self.tenant_id: Optional[str] = None
        self.user_id: Optional[str] = None
        self.email: Optional[str] = None
        self.emp_id: Optional[str] = None
        self.internal_roles: List[str] = []
        self.permissions: Set[str] = set()
        self.raw_groups: List[str] = []

    def __repr__(self):
        return (
            f"AuthContext(authenticated={self.authenticated}, tenant={self.tenant_id}, "
            f"email={self.email}, emp_id={self.emp_id}, roles={self.internal_roles}, perms={len(self.permissions)})"
        )


def validate_jwt(token: str) -> AuthContext:
    """
    Validate a JWT and resolve tenant/roles/permissions.
    Supports HS256 (local dev) and RS256 (production).
    """
    ctx = AuthContext()

    if not token or token == "Bearer":
        return ctx

    try:
        # Inspect header without verification to determine algorithm and tenant
        unverified = jwt.decode(token, options={"verify_signature": False})
        tenant_id = unverified.get("tenant_id") or unverified.get("tid") or unverified.get("org_id")

        if not tenant_id:
            return ctx

        algo = unverified.get("alg", JWT_ALGORITHM)

        if algo == "HS256":
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUDIENCE)
        else:
            issuer = unverified.get("iss", "")
            if JWT_ISSUERS and issuer not in JWT_ISSUERS:
                return ctx
            # RS256 path: in production, replace JWT_SECRET with JWKS key fetch
            payload = jwt.decode(
                token,
                key=JWT_SECRET,
                algorithms=["RS256"],
                audience=JWT_AUDIENCE,
                issuer=issuer if issuer else None
            )

        # Resolve groups to roles to permissions via tenant_resolver
        groups = payload.get("groups", []) or payload.get("roles", [])

        # Import here to avoid circular dependency at module level
        from agent.auth.tenant_resolver import resolve_tenant_groups, get_role_permissions

        internal_roles = resolve_tenant_groups(tenant_id, groups)
        permissions = get_role_permissions(internal_roles)

        ctx.authenticated = True
        ctx.tenant_id = tenant_id
        ctx.user_id = payload.get("oid") or payload.get("sub")
        ctx.email = payload.get("email") or payload.get("upn") or payload.get("preferred_username")
        # EMPLOYEE_ID is stored as string in JWT to preserve leading zeros (e.g., '000024')
        # RDEMOROCKFORT's V_EMP view uses EMPLOYEE_NO as the key field, not EMPLOYEE_ID
        ctx.emp_id = payload.get("emp_id")
        ctx.raw_groups = groups
        ctx.internal_roles = internal_roles
        ctx.permissions = permissions

    except jwt.ExpiredSignatureError:
        pass
    except jwt.InvalidTokenError:
        pass
    except Exception:
        pass

    return ctx