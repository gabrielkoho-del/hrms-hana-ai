"""
agent/integrations/schema_registry.py

Per-tenant HANA schema registry service.

Replaces ad-hoc file I/O in agentic_executor with a single
startup-loaded, in-memory registry that mirrors the production
architecture while using local JSON cache files instead of Redis.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.integrations.hana_client import is_hana_available

logger = logging.getLogger("hr_agent")


class SchemaRegistryService:
    """In-memory per-tenant HANA schema registry with JSON cache fallback.

    Loads all discovered tenant registries at startup and serves them
    from memory during request execution. No file I/O occurs on the
    hot path.

    Renewal policy:
      - TTL-based reload: ``get_registry()`` reloads a tenant when its
        in-memory cache is older than ``ttl_seconds``.
      - Background refresh: ``start_background_refresh()`` launches an
        ``asyncio`` task that periodically diffs the live HANA registry
        against the cached one and reloads on mismatch.
    """

    def __init__(self, cache_dir: Optional[str] = None, ttl_seconds: Optional[int] = None) -> None:
        self._cache_dir = Path(cache_dir or "data/hana_schema_cache")
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else int(
            __import__("os").getenv("SCHEMA_REGISTRY_TTL_SECONDS", "3600")
        )
        # Background refresh runs periodically to catch drift.
        # Default 3600s (1 hr); override with SCHEMA_REGISTRY_REFRESH_INTERVAL_SECONDS.
        self._refresh_interval_seconds = int(
            __import__("os").getenv("SCHEMA_REGISTRY_REFRESH_INTERVAL_SECONDS", "3600")
        )
        self._registries: Dict[str, Dict[str, List[str]]] = {}
        self._enriched: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._last_loaded_at: Dict[str, float] = {}
        self._refresh_task: Optional[asyncio.Task] = None
        self._refresh_lock = asyncio.Lock()

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def ttl_seconds(self) -> int:
        return self._ttl_seconds

    def _tenant_cache_path(self, tenant_id: str) -> Path:
        return self._cache_dir / f"{tenant_id.upper()}.json"

    def _now(self) -> float:
        import time as _time
        return _time.time()

    def _is_stale(self, tenant_key: str) -> bool:
        """Return True if the tenant's in-memory cache exceeds the TTL."""
        last = self._last_loaded_at.get(tenant_key)
        if last is None:
            return True
        return (self._now() - last) > self._ttl_seconds

    def _load_tenant_from_cache(self, tenant_id: str) -> Dict[str, List[str]]:
        """Load a single tenant registry from its JSON cache file.

        Returns empty dict if the file is missing or invalid.
        """
        cache_file = self._tenant_cache_path(tenant_id)
        if not cache_file.exists():
            return {}

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached.get("schemas"):
                schemas = {
                    k: [str(t).upper() for t in v if isinstance(v, list)]
                    for k, v in cached["schemas"].items()
                }
                logger.info(
                    "SchemaRegistry: loaded %d schemas for tenant=%s from %s",
                    len(schemas),
                    tenant_id,
                    cache_file.name,
                )
                return schemas
            logger.warning(
                "SchemaRegistry: cache file exists but has no schemas: %s",
                cache_file,
            )
        except Exception as e:
            logger.warning(
                "SchemaRegistry: cache load failed for tenant=%s at %s: %s",
                tenant_id,
                cache_file,
                e,
            )
        return {}

    def _load_tenant_from_fallbacks(self, tenant_id: str) -> Dict[str, List[str]]:
        """Try common fallback paths for tenant schema configs."""
        tenant_key = tenant_id.upper()
        fallback_paths = [
            Path("data") / f"{tenant_key}.json",
            Path("config") / f"{tenant_key}.json",
            Path(f"{tenant_key}.json"),
        ]
        for fallback in fallback_paths:
            if fallback.exists():
                try:
                    with open(fallback, "r", encoding="utf-8") as f:
                        cached = json.load(f)
                    if isinstance(cached, dict) and cached.get("schemas"):
                        schemas = {
                            k: [str(t).upper() for t in v if isinstance(t, str)]
                            for k, v in cached["schemas"].items()
                        }
                        logger.info(
                            "SchemaRegistry: loaded %d schemas for tenant=%s from fallback %s",
                            len(schemas),
                            tenant_id,
                            fallback,
                        )
                        return schemas
                except Exception as e:
                    logger.warning(
                        "SchemaRegistry: fallback load failed for tenant=%s at %s: %s",
                        tenant_id,
                        fallback,
                        e,
                    )
        return {}

    def load_tenant(self, tenant_id: str) -> Dict[str, List[str]]:
        """Load registry for a single tenant, preferring cache then fallbacks."""
        tenant_key = tenant_id.upper()
        if tenant_key in self._registries and not self._is_stale(tenant_key):
            return self._registries[tenant_key]

        schemas = self._load_tenant_from_cache(tenant_id)
        if not schemas:
            schemas = self._load_tenant_from_fallbacks(tenant_id)

        self._registries[tenant_key] = schemas
        self._last_loaded_at[tenant_key] = self._now()
        if schemas:
            logger.info(
                "SchemaRegistry: loaded %d schemas for tenant=%s",
                len(schemas),
                tenant_id,
            )
        else:
            logger.warning(
                "SchemaRegistry: no schema cache found for tenant=%s",
                tenant_id,
            )
        return schemas

    def load_all_tenants(self, tenant_ids: Optional[List[str]] = None) -> Dict[str, int]:
        """Load registries for all known tenants.

        If tenant_ids is None, scans the cache directory for JSON files.

        Returns a summary dict mapping tenant_id to schema count.
        """
        if tenant_ids is None:
            tenant_ids = self._discover_tenant_ids()

        summary: Dict[str, int] = {}
        for tenant_id in tenant_ids:
            schemas = self.load_tenant(tenant_id)
            summary[tenant_id] = len(schemas)

        total_tenants = len(summary)
        total_tables = sum(
            len(tables)
            for schemas in self._registries.values()
            for tables in schemas.values()
        )
        logger.info(
            "SchemaRegistry: startup complete — tenants=%d total_tables=%d",
            total_tenants,
            total_tables,
        )
        return summary

    def _discover_tenant_ids(self) -> List[str]:
        """Scan the cache directory for tenant JSON files."""
        tenant_ids: List[str] = []
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            for path in self._cache_dir.glob("*.json"):
                tenant_id = path.stem
                if tenant_id:
                    tenant_ids.append(tenant_id)
        except Exception as e:
            logger.warning("SchemaRegistry: tenant discovery failed: %s", e)
        return sorted(tenant_ids)

    def get_registry(self, tenant_id: str) -> Dict[str, List[str]]:
        """Return the registry for the given tenant, loading/reloading on demand if needed."""
        tenant_key = tenant_id.upper()
        if tenant_key not in self._registries or self._is_stale(tenant_key):
            # If HANA is known to be down, don't attempt a live discovery -- just
            # serve whatever is cached (possibly empty) to avoid connection
            # retries on every user query.
            if not is_hana_available():
                logger.info(
                    "SchemaRegistry: HANA unavailable -- skipping live discovery for tenant=%s",
                    tenant_id,
                )
                return self._registries.get(tenant_key, {})
            return self.load_tenant(tenant_id)
        return self._registries[tenant_key]

    def get_all_tenants(self) -> List[str]:
        """Return all tenant IDs that have been loaded."""
        return sorted(self._registries.keys())

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of all loaded registries for health checks."""
        total_schemas = 0
        total_tables = 0
        tenant_schemas: Dict[str, int] = {}
        for tenant_key, schemas in self._registries.items():
            tenant_tables = sum(len(tables) for tables in schemas.values())
            total_schemas += len(schemas)
            total_tables += tenant_tables
            tenant_schemas[tenant_key] = tenant_tables

        age = {}
        for tenant_key, ts in self._last_loaded_at.items():
            age[tenant_key] = round(self._now() - ts, 1)

        return {
            "tenants_loaded": len(self._registries),
            "total_schemas": total_schemas,
            "total_tables": total_tables,
            "tenant_schemas": tenant_schemas,
            "tenant_age_seconds": age,
            "ttl_seconds": self._ttl_seconds,
            "cache_dir": str(self._cache_dir),
        }

    def invalidate_tenant(self, tenant_id: str) -> None:
        """Remove a tenant from the in-memory cache."""
        tenant_key = tenant_id.upper()
        self._registries.pop(tenant_key, None)
        self._enriched.pop(tenant_key, None)
        self._last_loaded_at.pop(tenant_key, None)
        logger.info("SchemaRegistry: invalidated cache for tenant=%s", tenant_id)

    def invalidate_all(self) -> None:
        """Clear all in-memory registries."""
        self._registries.clear()
        self._enriched.clear()
        self._last_loaded_at.clear()
        logger.info("SchemaRegistry: invalidated all caches")

    def enrich_schema_with_semantics(
        self,
        tenant_id: str,
        schemas: Dict[str, List[str]],
        hana_client: Any,
        max_tables: int = 50,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Enrich schema registry with business semantics from HANA explain_table.

        Calls hana_explain_table for tables that do not yet have cached
        descriptions, up to max_tables total calls to avoid startup blowups.
        Returns a map of schema -> table -> {description, ...}.
        """
        tenant_key = tenant_id.upper()
        enriched: Dict[str, Dict[str, Dict[str, Any]]] = {}
        remaining = max_tables
        for schema_name, tables in schemas.items():
            table_meta: Dict[str, Dict[str, Any]] = {}
            for table in tables:
                if remaining <= 0:
                    break
                try:
                    result = hana_client.call_tool(
                        "hana_explain_table",
                        {"table_name": table, "schema_name": schema_name},
                    )
                    normalized = result
                    if isinstance(result, dict):
                        normalized = result.get("result", result)
                    description = ""
                    columns: List[Dict[str, Any]] = []
                    if isinstance(normalized, dict):
                        description = str(normalized.get("description") or normalized.get("business_description") or "").strip()
                        raw_cols = normalized.get("columns") or normalized.get("fields") or []
                        if isinstance(raw_cols, list):
                            columns = [c for c in raw_cols if isinstance(c, dict)]
                    elif isinstance(normalized, list):
                        for item in normalized:
                            if isinstance(item, dict):
                                desc = str(item.get("description") or "").strip()
                                if desc:
                                    description = desc
                                    break
                    if description or columns:
                        table_meta[table] = {
                            "description": description,
                            "columns": columns,
                        }
                        remaining -= 1
                except Exception as e:
                    logger.debug("SchemaRegistry: explain_table failed for %s.%s: %s", schema_name, table, e)
            if table_meta:
                enriched[schema_name] = table_meta
        if enriched:
            self._enriched[tenant_key] = enriched
            logger.info("SchemaRegistry: enriched %d tables with semantics for tenant=%s", sum(len(v) for v in enriched.values()), tenant_id)
        return enriched

    def get_enriched_schema(self, tenant_id: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Return enriched schema metadata for the given tenant."""
        return self._enriched.get(tenant_id.upper(), {})

    async def start_background_refresh(
        self,
        tenant_ids: List[str],
        hana_manager: Any,
        refresh_interval_seconds: Optional[int] = None,
    ) -> None:
        """Start a background task that periodically refreshes tenant registries.

        The task compares the live HANA registry against the cached one and
        reloads when they differ.  It never cancels an in-flight refresh.
        """
        if self._refresh_task is not None:
            return

        interval = refresh_interval_seconds if refresh_interval_seconds is not None else int(
            __import__("os").getenv("SCHEMA_REGISTRY_REFRESH_INTERVAL_SECONDS", str(self._ttl_seconds))
        )

        async def _refresh_loop() -> None:
            while True:
                await asyncio.sleep(interval)
                # If HANA is known to be down, skip the live diff entirely so we
                # don't retry connections against a dead server every interval.
                if not is_hana_available():
                    continue
                for tenant_id in tenant_ids:
                    tenant_key = tenant_id.upper()
                    async with self._refresh_lock:
                        try:
                            client = hana_manager.get_client(tenant_id)
                            raw_schemas = client.call_tool("hana_list_schemas", {})
                            try:
                                from agent.main import _normalize_name_list
                                normalized = _normalize_name_list(raw_schemas)
                            except Exception:
                                normalized = []
                            
                            live_schemas: Dict[str, List[str]] = {}
                            for schema_name in normalized:
                                try:
                                    raw_tables = client.call_tool("hana_list_tables", {"schema_name": schema_name})
                                    try:
                                        from agent.main import _normalize_name_list
                                        tables = _normalize_name_list(raw_tables)
                                    except Exception:
                                        tables = []
                                    if tables:
                                        live_schemas[schema_name] = tables
                                except Exception as e:
                                    logger.warning("SchemaRegistry: background list_tables failed for %s: %s", schema_name, e)
                            
                            cached_schemas = self._registries.get(tenant_key, {})
                            if live_schemas != cached_schemas:
                                logger.info(
                                    "SchemaRegistry: background refresh triggered reload for tenant=%s (cached=%d schemas, live=%d schemas)",
                                    tenant_id, len(cached_schemas), len(live_schemas),
                                )
                                self._registries[tenant_key] = live_schemas
                                self._last_loaded_at[tenant_key] = self._now()
                                try:
                                    cache_file = self._tenant_cache_path(tenant_id)
                                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                                    with open(cache_file, "w", encoding="utf-8") as f:
                                        json.dump({"schemas": live_schemas, "updated_at": self._now()}, f)
                                except Exception as e:
                                    logger.warning("SchemaRegistry: background cache write failed for tenant=%s: %s", tenant_id, e)
                        except Exception as e:
                            logger.warning("SchemaRegistry: background refresh failed for tenant=%s: %s", tenant_id, e)

        self._refresh_task = asyncio.create_task(_refresh_loop())
        logger.info(
            "SchemaRegistry: background refresh started (interval=%ds, tenants=%s)",
            interval,
            ", ".join(tenant_ids),
        )


# Global singleton (must be after class definitions)
schema_registry_service = SchemaRegistryService()