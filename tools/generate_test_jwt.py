# tools/generate_test_jwt.py
"""
Generate test JWT tokens for local development with auto-verification.
Usage:
    # Preset users (quickest)
    python tools/generate_test_jwt.py --user lsoto        # HR role, lsoto@hr.com
    python tools/generate_test_jwt.py --user patrickr     # Employee role, patrickr@employee.io

    # Manual role + email
    python tools/generate_test_jwt.py --role hr --email lsoto@hr.com --emp-id 1
    python tools/generate_test_jwt.py --role employee --email patrickr@employee.io --emp-id 42
    python tools/generate_test_jwt.py --role manager --email boss@local.dev --emp-id 10
    
    # With specific tenant
    python tools/generate_test_jwt.py --tenant RDEMOROCKFORT --role hr --email test@local.dev --emp-id 1
"""
import jwt
import datetime
import argparse
import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import os

load_dotenv(project_root / "config" / ".env")

JWT_SECRET = os.getenv("JWT_SECRET", "local-dev-secret-change-me")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "hrms-api")


def load_tenant_group_mappings(tenant_id: str) -> dict:
    """Load group mappings from tenant_mappings.yaml for the specified tenant."""
    mappings_path = project_root / "config" / "tenant_mappings.yaml"
    with open(mappings_path, "r") as f:
        config = yaml.safe_load(f)
    
    tenant = config.get("tenants", {}).get(tenant_id, {})
    group_mappings = tenant.get("group_mappings", {})
    
    role_to_group = {}
    for ext_group, int_role in group_mappings.items():
        ext_lower = ext_group.lower()
        # Prefer group names that match the role pattern (e.g., "rockfortHR" for "hr")
        if int_role == "HRMS_HR":
            if "hr" in ext_lower and "admin" not in ext_lower:
                role_to_group["hr"] = ext_group
            elif "hr" not in role_to_group:
                # Fallback if no matching name found
                role_to_group["hr"] = ext_group
        elif int_role == "HRMS_EMPLOYEE":
            if "employee" in ext_lower:
                role_to_group["employee"] = ext_group
            elif "employee" not in role_to_group:
                role_to_group["employee"] = ext_group
        elif int_role == "HRMS_MANAGER":
            if "manager" in ext_lower:
                role_to_group["manager"] = ext_group
            elif "manager" not in role_to_group:
                role_to_group["manager"] = ext_group
    
    return role_to_group


def make_test_token(tenant_id: str, groups: list, user_email: str = "test@local.dev", emp_id: str = None) -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Resolve external IDP groups to internal DAB roles using the same
    # tenant resolver the agent uses at runtime. DAB's JWT provider reads
    # the `roles` claim (not `groups`), so we emit both: `groups` for
    # audit/logging and `roles` for DAB authorization.
    from agent.auth.tenant_resolver import resolve_tenant_groups
    internal_roles = resolve_tenant_groups(tenant_id, groups)
    
    payload = {
        "tenant_id": tenant_id,
        "groups": groups,
        "roles": internal_roles,
        "email": user_email,
        "sub": f"test-user-{tenant_id}",
        "iat": now,
        "exp": now + datetime.timedelta(hours=24),
        "iss": "local-dev-issuer",
        "aud": JWT_AUDIENCE,
    }
    if emp_id is not None:
        payload["emp_id"] = emp_id
    
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return token


def verify_token_locally(token: str) -> dict:
    """Verify token and simulate what mcp_server.py does."""
    from agent.auth.tenant_resolver import resolve_tenant_groups, get_role_permissions
    
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUDIENCE)
        
        tenant_id = decoded.get("tenant_id")
        groups = decoded.get("groups", [])
        
        internal_roles = resolve_tenant_groups(tenant_id, groups)
        permissions = get_role_permissions(internal_roles)
        
        return {
            "valid": True,
            "decoded": decoded,
            "tenant_id": tenant_id,
            "internal_roles": internal_roles,
            "permissions": sorted(permissions),
            "raw_groups": groups,
        }
    
    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "Token expired"}
    except jwt.InvalidTokenError as e:
        return {"valid": False, "error": str(e)}
    except Exception as e:
        return {"valid": False, "error": f"Verification failed: {e}"}


def generate_and_print_token(tenant_id, role, email, emp_id, group_name, label=""):
    """Generate a single token, verify it, and print formatted output."""
    groups = [group_name]
    token = make_test_token(tenant_id, groups, email, emp_id)
    verification = verify_token_locally(token)
    
    label_str = f" [{label}]" if label else ""
    print("=" * 60)
    print(f"  Test JWT Generated{label_str}")
    print(f"  Role: {role.upper()}  |  Tenant: {tenant_id}")
    print(f"  Email: {email}")
    print(f"  Emp ID: {emp_id}")
    print(f"  Groups: {groups}")
    print("=" * 60)
    print(f"{token}")
    
    print("=" * 60)
    print("  LOCAL VERIFICATION")
    print("=" * 60)
    
    if verification["valid"]:
        d = verification["decoded"]
        print(f"  Token: VALID")
        print(f"  Tenant ID:    {verification['tenant_id']}")
        print(f"  Email:        {d.get('email')}")
        print(f"  Emp ID:       {d.get('emp_id')}")
        print(f"  Raw Groups:   {verification['raw_groups']}")
        print(f"  Resolved:     {verification['internal_roles']}")
        print(f"  Permissions ({len(verification['permissions'])}):")
        for perm in verification["permissions"]:
            print(f"    • {perm}")
        print(f"  Expires:      {datetime.datetime.fromtimestamp(d['exp'], tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    else:
        print(f"  Token: INVALID")
        print(f"  Error: {verification['error']}")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Generate test JWT for HRMS")
    
    parser.add_argument(
        "--user",
        choices=["lsoto", "patrickr"],
        default=None,
        help="Preset user (auto-sets role + email): lsoto=HR, patrickr=Employee",
    )
    parser.add_argument(
        "--role",
        default=None,
        help="Which test role to generate token for. Accepts abstract roles (hr/manager/employee, auto-mapped via tenant_mappings.yaml) or the raw external group name (e.g. rockfortHR).",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="User email in token",
    )
    parser.add_argument(
        "--emp-id",
        type=str,
        default=None,
        help="Employee ID to embed in the test token",
    )
    parser.add_argument(
        "--tenant",
        default="LOCALDEV",
        help="Tenant ID in tenant_mappings.yaml",
    )
    args = parser.parse_args()
    
    PRESETS = {
        "lsoto":     {"role": "hr",       "email": "gabrielho@hr.com",         "emp_id": "1"},
        "patrickr":  {"role": "employee", "email": "patrickr@employee.io", "emp_id": "2"},
    }
    
    # Resolve tenant-specific group mappings
    group_map = load_tenant_group_mappings(args.tenant)
    
    # CASE 1: No args at all → generate ALL presets
    if not args.user and not args.role and not args.email and args.emp_id is None:
        print("=" * 60)
        print("  GENERATING ALL PRESET TOKENS")
        print("=" * 60)
        print()
        
        for preset_name, preset in PRESETS.items():
            role = preset["role"]
            # If the user passed an abstract role (hr/manager/employee) and the tenant
            # has a group mapping, use the mapped group name. Otherwise treat the
            # argument as the actual external group name (e.g. "rockfortHR").
            group_name = group_map.get(role, role)
            generate_and_print_token(
                args.tenant,
                role,
                preset["email"],
                preset["emp_id"],
                group_name,
                label=preset_name
            )
        
        print("=" * 60)
        print("  USAGE")
        print("=" * 60)
        print("  1. Copy a token above into librechat.yaml apiKey field")
        print("  2. Start Agent:      python -m agent.main")
        print("  3. Restart LibreChat and test!")
        print("=" * 60)
        return
    
    # CASE 2: Single preset user
    if args.user:
        preset = PRESETS[args.user]
        role = preset["role"]
        email = preset["email"]
        emp_id = preset["emp_id"]
    
    # CASE 3: Manual args
    else:
        role = args.role or "hr"
        email = args.email or "test@local.dev"
        emp_id = args.emp_id if args.emp_id is not None else "1"
    
    group_name = group_map.get(role, f"Test{role.capitalize()}")
    generate_and_print_token(args.tenant, role, email, emp_id, group_name)
    
    print("=" * 60)
    print("  USAGE")
    print("=" * 60)
    print("  1. Start MCP server: python mcp_server.py")
    print("  2. Test via MCP client (see test_mcp_client.py)")
    print("  3. Or use this token in your agent/main.py flow")
    print("=" * 60)


if __name__ == "__main__":
    main()