"""Configuration loader for YAML-based planner configs.

Provides cached access to dimension mappings, leave entities, chart patterns,
and dashboard configurations. Supports multi-tenant config overrides via
TenantConfigResolver.
"""
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

import yaml

from agent.core.tenant_config_resolver import TenantConfigResolver

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))


def _load_yaml(filename: str) -> Dict[str, Any]:
    path = os.path.join(_CONFIG_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────
# Global config functions (fallback, no tenant)
# ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_dimension_mappings() -> Dict[str, List[str]]:
    data = _load_yaml("dimension_mappings.yaml")
    return {k: v for k, v in (data.get("dimension_map") or {}).items()}


@lru_cache(maxsize=1)
def get_all_dimension_fields() -> set:
    data = _load_yaml("dimension_mappings.yaml")
    return set(data.get("all_dimension_fields") or [])


@lru_cache(maxsize=1)
def get_leave_entitlement_entity_preference() -> List[str]:
    data = _load_yaml("leave_entities.yaml")
    return list(data.get("entitlement_entity_preference") or [])


@lru_cache(maxsize=1)
def get_leave_entitlement_entities() -> frozenset:
    data = _load_yaml("leave_entities.yaml")
    return frozenset(data.get("entitlement_entities") or [])


@lru_cache(maxsize=1)
def get_leave_entitlement_keywords() -> tuple:
    data = _load_yaml("leave_entities.yaml")
    return tuple(data.get("entitlement_keywords") or [])


@lru_cache(maxsize=1)
def get_action_keywords() -> tuple:
    data = _load_yaml("leave_entities.yaml")
    return tuple(data.get("action_keywords") or [])


@lru_cache(maxsize=1)
def get_chart_type_patterns() -> List[Dict[str, str]]:
    data = _load_yaml("chart_patterns.yaml")
    return list(data.get("chart_type_patterns") or [])


@lru_cache(maxsize=1)
def get_general_chart_pattern() -> str:
    data = _load_yaml("chart_patterns.yaml")
    return data.get("general_chart_pattern", r"\b(chart|graph|visualize|visualization|plot|dashboard|kpi)\b")


@lru_cache(maxsize=1)
def get_multi_chart_patterns() -> List[str]:
    data = _load_yaml("chart_patterns.yaml")
    return list(data.get("multi_chart_patterns") or [])


@lru_cache(maxsize=1)
def get_rag_keywords() -> tuple:
    data = _load_yaml("chart_patterns.yaml")
    return tuple(data.get("rag_keywords") or [])


@lru_cache(maxsize=1)
def get_export_keywords() -> tuple:
    data = _load_yaml("chart_patterns.yaml")
    return tuple(data.get("export_keywords") or [])


@lru_cache(maxsize=1)
def get_named_dashboards() -> Dict[str, List[Dict[str, Any]]]:
    data = _load_yaml("dashboards.yaml")
    return dict(data.get("dashboards") or {})


# ─────────────────────────────────────────────────────────
# Tenant-aware config functions
# ─────────────────────────────────────────────────────────

def get_dimension_mappings_for_tenant(tenant_id: str) -> Dict[str, List[str]]:
    """Get dimension mappings for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_dimension_mappings()


def get_all_dimension_fields_for_tenant(tenant_id: str) -> set:
    """Get all dimension fields for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_all_dimension_fields()


def get_leave_entitlement_entity_preference_for_tenant(tenant_id: str) -> List[str]:
    """Get leave entitlement entity preference for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_leave_entitlement_entity_preference()


def get_leave_entitlement_entities_for_tenant(tenant_id: str) -> frozenset:
    """Get leave entitlement entities for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_leave_entitlement_entities()


def get_leave_entitlement_keywords_for_tenant(tenant_id: str) -> tuple:
    """Get leave entitlement keywords for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_leave_entitlement_keywords()


def get_action_keywords_for_tenant(tenant_id: str) -> tuple:
    """Get action keywords for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_action_keywords()


def get_chart_type_patterns_for_tenant(tenant_id: str) -> List[Dict[str, str]]:
    """Get chart type patterns for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_chart_type_patterns()


def get_general_chart_pattern_for_tenant(tenant_id: str) -> str:
    """Get general chart pattern for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_general_chart_pattern()


def get_multi_chart_patterns_for_tenant(tenant_id: str) -> List[str]:
    """Get multi-chart patterns for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_multi_chart_patterns()


def get_rag_keywords_for_tenant(tenant_id: str) -> tuple:
    """Get RAG keywords for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_rag_keywords()


def get_export_keywords_for_tenant(tenant_id: str) -> tuple:
    """Get export keywords for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_export_keywords()


def get_named_dashboards_for_tenant(tenant_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get named dashboards for a specific tenant."""
    resolver = TenantConfigResolver(tenant_id)
    return resolver.get_named_dashboards()


def clear_config_cache() -> None:
    """Clear all cached config values (useful in tests)."""
    get_dimension_mappings.cache_clear()
    get_all_dimension_fields.cache_clear()
    get_leave_entitlement_entity_preference.cache_clear()
    get_leave_entitlement_entities.cache_clear()
    get_leave_entitlement_keywords.cache_clear()
    get_action_keywords.cache_clear()
    get_chart_type_patterns.cache_clear()
    get_general_chart_pattern.cache_clear()
    get_multi_chart_patterns.cache_clear()
    get_rag_keywords.cache_clear()
    get_export_keywords.cache_clear()
    get_named_dashboards.cache_clear()
