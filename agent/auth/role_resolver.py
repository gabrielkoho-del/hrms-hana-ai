"""
Auth role resolution and data-access guards.

Encapsulates:
  - Mapping permissions / internal roles to a canonical auth_role
    ("admin", "manager", "employee")
  - Row-level self-access check: verify that every returned record belongs
    to the requesting user.
"""
from typing import Any, Dict, List, Optional, Tuple


def resolve_auth_role(auth_context: Any) -> Tuple[str, bool, Optional[Any], Optional[Any]]:
    """Resolve the canonical auth_role from permissions and internal roles.

    Returns:
        (auth_role, fallback_used, permissions, internal_roles)

    Priority:
      1. permissions.read:all_employees  -> admin
      2. permissions.read:subordinates   -> manager
      3. internal_roles / role fallback   -> admin / manager if role matches
      4. otherwise                       -> employee
    """
    auth_role = "employee"
    fallback_used = False
    perms = getattr(auth_context, "permissions", None)
    roles = getattr(auth_context, "internal_roles", None)

    if perms:
        if "read:all_employees" in perms:
            auth_role = "admin"
        elif "read:subordinates" in perms:
            auth_role = "manager"

    if roles and auth_role == "employee":
        fallback_used = True
        role_set = (
            {r.lower() for r in roles}
            if isinstance(roles, (list, set, tuple))
            else {str(roles).lower()}
        )
        if any(r in role_set for r in ("hrms_hr", "admin", "hr")):
            auth_role = "admin"
        elif any(r in role_set for r in ("hrms_manager", "manager", "supervisor")):
            auth_role = "manager"

    if auth_role == "employee" and getattr(auth_context, "role", None):
        fallback_used = True
        role = auth_context.role.lower()
        if role in ("admin", "hrms_hr", "hr"):
            auth_role = "admin"
        elif role in ("manager", "supervisor", "hrms_manager"):
            auth_role = "manager"

    return auth_role, fallback_used, perms, roles


def is_data_about_user(items: List[Dict], auth_context: Any) -> bool:
    """Return True if EVERY returned data item belongs to the requesting user."""
    if not auth_context or (not getattr(auth_context, "email", None) and not getattr(auth_context, "emp_id", None)):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        is_self = False
        if getattr(auth_context, "email", None):
            item_email = str(item.get("EMAIL", item.get("email", ""))).lower()
            if item_email == auth_context.email.lower():
                is_self = True
        if getattr(auth_context, "emp_id", None):
            item_emp_id = str(item.get("EMPLOYEE_NO", item.get("emp_id", "")))
            if item_emp_id == str(getattr(auth_context, "emp_id", "")):
                is_self = True
        if not is_self:
            return False
    return True
