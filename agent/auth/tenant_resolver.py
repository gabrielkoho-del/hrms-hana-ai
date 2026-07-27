# agent/auth/tenant_resolver.py
import os
import yaml
from pathlib import Path
from typing import Dict, List, Set, Optional

DEFAULT_MAPPINGS_PATH = Path(__file__).parent.parent.parent / "config" / "tenant_mappings.yaml"


class TenantResolver:
    def __init__(self, mappings_path: Optional[Path] = None):
        self.mappings_path = mappings_path or DEFAULT_MAPPINGS_PATH
        self._cache: Optional[Dict] = None
        self._mtime: float = 0

    def _load(self) -> Dict:
        if not self.mappings_path.exists():
            raise FileNotFoundError(
                f"Tenant mappings not found: {self.mappings_path}\n"
                f"Expected at: {self.mappings_path.resolve()}"
            )

        current_mtime = self.mappings_path.stat().st_mtime
        if self._cache is None or current_mtime != self._mtime:
            with open(self.mappings_path, "r", encoding="utf-8") as f:
                self._cache = yaml.safe_load(f)
            self._mtime = current_mtime
            print(f"[TenantResolver] Loaded mappings from {self.mappings_path}")

        return self._cache

    def resolve_groups(self, tenant_id: str, external_groups: List[str]) -> List[str]:
        data = self._load()
        tenant = data.get("tenants", {}).get(tenant_id)

        if not tenant:
            print(f"[TenantResolver] Unknown tenant: '{tenant_id}'")
            return []

        mappings = tenant.get("group_mappings", {})
        resolved = []

        for ext_group in external_groups:
            for mapped_group, internal_role in mappings.items():
                if ext_group.lower() == mapped_group.lower():
                    resolved.append(internal_role)
                    break

        seen: Set[str] = set()
        return [r for r in resolved if not (r in seen or seen.add(r))]

    def get_role_permissions(self, internal_roles: List[str]) -> Set[str]:
        """
        Get aggregated permissions for a list of internal roles.
        YAML structure: roles.<role_name>.permissions (list)
        """
        data = self._load()
        # FIX: Use 'roles' key, not 'role_permissions'
        roles_def = data.get("roles", {})

        permissions: Set[str] = set()
        for role in internal_roles:
            role_data = roles_def.get(role, {})
            # FIX: Extract nested 'permissions' list from role definition
            role_perms = role_data.get("permissions", [])
            permissions.update(role_perms)
        return permissions

    def get_tenant_info(self, tenant_id: str) -> Optional[Dict]:
        data = self._load()
        return data.get("tenants", {}).get(tenant_id)

    def get_tenant_dab_url(self, tenant_id: str) -> Optional[str]:
        tenant = self.get_tenant_info(tenant_id)
        if not tenant:
            return None
        return tenant.get("dab_url")

    def list_tenants(self) -> List[str]:
        data = self._load()
        return list(data.get("tenants", {}).keys())


# Singleton instance
resolver = TenantResolver()

# Convenience functions
def resolve_tenant_groups(tenant_id: str, external_groups: List[str]) -> List[str]:
    return resolver.resolve_groups(tenant_id, external_groups)

def get_role_permissions(internal_roles: List[str]) -> Set[str]:
    return resolver.get_role_permissions(internal_roles)

def get_tenant_info(tenant_id: str) -> Optional[Dict]:
    return resolver.get_tenant_info(tenant_id)

def list_tenants() -> List[str]:
    return resolver.list_tenants()


# ─── Self-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Tenants:", list_tenants())
    print("XYZ HR:", resolve_tenant_groups("XYZ", ["Human Resources Team"]))
    print("LOCALDEV Employee:", resolve_tenant_groups("LOCALDEV", ["TestEmployee"]))
    
    # Test permissions
    hr_perms = get_role_permissions(["HRMS_HR"])
    print(f"\nHRMS_HR permissions ({len(hr_perms)}):")
    for p in sorted(hr_perms):
        print(f"  • {p}")
    
    mgr_perms = get_role_permissions(["HRMS_MANAGER"])
    print(f"\nHRMS_MANAGER permissions ({len(mgr_perms)}):")
    for p in sorted(mgr_perms):
        print(f"  • {p}")