# Re-export so callers can do:
#   from agent.auth import resolve_tenant_groups
# Instead of:
#   from agent.auth.tenant_resolver import resolve_tenant_groups

from .tenant_resolver import (
    resolver,
    resolve_tenant_groups,
    get_role_permissions,
    get_tenant_info,
    list_tenants,
)
"""Authentication submodule exports."""
from .models import AuthContext, validate_jwt
from .dependencies import verify_token