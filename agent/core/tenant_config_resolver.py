"""Multi-tenant configuration resolver.

Provides per-tenant configuration loading with TTL caching, following the
same pattern as CodeResolver. Tenant-specific configs override global defaults.
"""
import logging
import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("hr_agent")

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))


class _TTLCacheEntry:
    def __init__(self, value: any, ttl_seconds: int = 300):
        self.value = value
        self.expires_at = time.time() + ttl_seconds

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class TenantConfigResolver:
    """Per-tenant singleton that caches configuration with TTL.

    Multi-tenant architecture:
    - Each tenant can have config overrides in tenant-specific YAML files
    - Falls back to global configs when tenant-specific ones don't exist
    - Instance keyed by tenant_id ensures isolation

    Supported configs:
    - dimension_mappings
    - leave_entities
    - chart_patterns
    - dashboards
    """

    _instances: Dict[str, "TenantConfigResolver"] = {}
    _lock = Lock()

    def __new__(cls, tenant_id: str) -> "TenantConfigResolver":
        with cls._lock:
            if tenant_id not in cls._instances:
                instance = super().__new__(cls)
                instance.tenant_id = tenant_id
                instance._cache_entry: Optional[_TTLCacheEntry] = None
                instance._config: Dict[str, Any] = {}
                instance._initialized = False
                cls._instances[tenant_id] = instance
            return cls._instances[tenant_id]

    def _tenant_config_path(self, filename: str) -> str:
        """Return path to tenant-specific config file, if it exists."""
        tenant_filename = f"{filename.rsplit('.', 1)[0]}.{self.tenant_id}.yaml"
        path = os.path.join(_CONFIG_DIR, tenant_filename)
        if os.path.exists(path):
            return path
        return os.path.join(_CONFIG_DIR, filename)

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _merge_dicts(self, base: Dict, override: Dict) -> Dict:
        """Deep merge override into base."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    def _load_config(self) -> Dict[str, Any]:
        """Load merged config for this tenant."""
        config: Dict[str, Any] = {}

        # Load dimension mappings
        dim_path = self._tenant_config_path("dimension_mappings.yaml")
        dim_data = self._load_yaml(dim_path)
        if dim_data:
            config["dimension_map"] = dim_data.get("dimension_map", {})
            config["all_dimension_fields"] = set(dim_data.get("all_dimension_fields") or [])

        # Load leave entities
        leave_path = self._tenant_config_path("leave_entities.yaml")
        leave_data = self._load_yaml(leave_path)
        if leave_data:
            config["leave_entitlement_entity_preference"] = leave_data.get("entitlement_entity_preference", [])
            config["leave_entitlement_entities"] = frozenset(leave_data.get("entitlement_entities") or [])
            config["leave_entitlement_keywords"] = tuple(leave_data.get("entitlement_keywords") or [])
            config["action_keywords"] = tuple(leave_data.get("action_keywords") or [])

        # Load chart patterns
        chart_path = self._tenant_config_path("chart_patterns.yaml")
        chart_data = self._load_yaml(chart_path)
        if chart_data:
            config["chart_type_patterns"] = chart_data.get("chart_type_patterns", [])
            config["general_chart_pattern"] = chart_data.get("general_chart_pattern", "")
            config["multi_chart_patterns"] = chart_data.get("multi_chart_patterns", [])
            config["rag_keywords"] = tuple(chart_data.get("rag_keywords") or [])
            config["export_keywords"] = tuple(chart_data.get("export_keywords") or [])

        # Load dashboards
        dash_path = self._tenant_config_path("dashboards.yaml")
        dash_data = self._load_yaml(dash_path)
        if dash_data:
            config["dashboards"] = dash_data.get("dashboards", {})

        return config

    async def get_config(self) -> Dict[str, Any]:
        """Get config for this tenant, refreshing if expired."""
        if self._cache_entry and not self._cache_entry.is_expired():
            return self._config

        await self._refresh_cache()
        return self._config

    async def _refresh_cache(self):
        """Refresh config cache for this tenant."""
        try:
            self._config = self._load_config()
            self._cache_entry = _TTLCacheEntry(self._config, ttl_seconds=300)
            self._initialized = True
            logger.info(
                "TenantConfigResolver: refreshed %s config with %d keys",
                self.tenant_id,
                len(self._config),
            )
        except Exception as e:
            logger.warning(
                "TenantConfigResolver: failed to refresh cache for %s: %s. Using stale data if available.",
                self.tenant_id,
                e,
            )
            if not self._config:
                self._config = {}

    def _ensure_config_loaded(self) -> None:
        """Ensure config is loaded synchronously."""
        if not self._initialized and not self._config:
            try:
                self._config = self._load_config()
                self._cache_entry = _TTLCacheEntry(self._config, ttl_seconds=300)
                self._initialized = True
            except Exception as e:
                logger.warning(
                    "TenantConfigResolver: failed to load config for %s: %s",
                    self.tenant_id,
                    e,
                )
                self._config = {}

    def get_dimension_mappings(self) -> Dict[str, List[str]]:
        """Get dimension mappings for this tenant."""
        self._ensure_config_loaded()
        return dict(self._config.get("dimension_map", {}))

    def get_all_dimension_fields(self) -> set:
        """Get all dimension fields for this tenant."""
        self._ensure_config_loaded()
        return set(self._config.get("all_dimension_fields", []))

    def get_leave_entitlement_entity_preference(self) -> List[str]:
        """Get leave entitlement entity preference for this tenant."""
        self._ensure_config_loaded()
        return list(self._config.get("leave_entitlement_entity_preference", []))

    def get_leave_entitlement_entities(self) -> frozenset:
        """Get leave entitlement entities for this tenant."""
        self._ensure_config_loaded()
        return frozenset(self._config.get("leave_entitlement_entities", []))

    def get_leave_entitlement_keywords(self) -> tuple:
        """Get leave entitlement keywords for this tenant."""
        self._ensure_config_loaded()
        return tuple(self._config.get("leave_entitlement_keywords", []))

    def get_action_keywords(self) -> tuple:
        """Get action keywords for this tenant."""
        self._ensure_config_loaded()
        return tuple(self._config.get("action_keywords", []))

    def get_chart_type_patterns(self) -> List[Dict[str, str]]:
        """Get chart type patterns for this tenant."""
        self._ensure_config_loaded()
        return list(self._config.get("chart_type_patterns", []))

    def get_general_chart_pattern(self) -> str:
        """Get general chart pattern for this tenant."""
        self._ensure_config_loaded()
        return self._config.get("general_chart_pattern", "")

    def get_multi_chart_patterns(self) -> List[str]:
        """Get multi-chart patterns for this tenant."""
        self._ensure_config_loaded()
        return list(self._config.get("multi_chart_patterns", []))

    def get_rag_keywords(self) -> tuple:
        """Get RAG keywords for this tenant."""
        self._ensure_config_loaded()
        return tuple(self._config.get("rag_keywords", []))

    def get_export_keywords(self) -> tuple:
        """Get export keywords for this tenant."""
        self._ensure_config_loaded()
        return tuple(self._config.get("export_keywords", []))

    def get_named_dashboards(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get named dashboards for this tenant."""
        self._ensure_config_loaded()
        return dict(self._config.get("dashboards", {}))
