"""
agent/forecasting/external_data_fetcher.py
External data fetchers for the Forecasting Engine.
Fetches market data from OpenDOSM, World Bank, and Yahoo Finance,
and internal HR data from DAB. Merges everything into a single
enriched JSON dataset for the sandbox.

Design:
  - TTL cache per source; cache keys are namespaced by source so a
    single source can be invalidated without touching the others.
  - Each fetcher is fail-soft: any exception degrades to an empty
    result list so the rest of the enriched dataset is unaffected.
  - Cross-platform temp paths via FORECAST_TEMP_DIR env var or tempfile.
  - Decoupled from DAB client via injectable query_fn.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests

logger = logging.getLogger("hr_agent.forecasting")

# yfinance is optional at import time (install via requirements.txt)
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore
    _YFINANCE_AVAILABLE = False
    logger.info("yfinance not installed; Yahoo Finance fetchers disabled")

# ============================================================================
# CONFIG
# ============================================================================

_DEFAULT_TTLS = {
    "dosm": 86400,           # 1 day
    "worldbank": 604800,     # 7 days
    "yfinance": 14400,       # 4 hours
    "internal_hr": 3600,     # 1 hour
}

_FORECAST_TEMP_DIR = os.getenv("FORECAST_TEMP_DIR", None)
_FORECAST_DATA_DIR = os.getenv("FORECAST_DATA_DIR", None)

# DOSM API base
DOSM_BASE_URL = os.getenv("DOSM_BASE_URL", "https://api.data.gov.my/opendosm")
# World Bank API base
WORLD_BANK_BASE_URL = os.getenv("WORLD_BANK_BASE_URL", "https://api.worldbank.org/v2")

# ============================================================================
# TTL CACHE
# ============================================================================

class TTLCache:
    """Simple in-memory TTL cache keyed by cache_key."""

    def __init__(self):
        self._store: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str, ttl: int) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time())

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


# ============================================================================
# HELPERS
# ============================================================================

def _cache_key(source: str, *parts: Any) -> str:
    """Build a namespaced cache key.

    The first argument is a logical source name (e.g. "dosm", "worldbank",
    "yfinance", "internal_hr") so the caller can invalidate all entries for
    one source via ``invalidate_cache(source)``. The hash makes the final key
    fixed-length and safe to log.
    """
    raw = json.dumps((source, *parts), sort_keys=True, default=str)
    return f"{source}:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _month_key(dt: Any) -> str:
    if isinstance(dt, str):
        # Try parse ISO date
        try:
            return datetime.strptime(dt[:10], "%Y-%m-%d").strftime("%Y-%m")
        except Exception:
            return dt[:7]
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m")
    return str(dt)[:7]


def _ensure_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


# ============================================================================
# EXTERNAL DATA FETCHER
# ============================================================================

class ExternalDataFetcher:
    """Fetches external market data and internal HR data, then merges them.

    Args:
        dab_query_fn: Callable[[str, Dict], Any] that invokes a DAB tool.
            Signature: async dab_query_fn(tool_name: str, args: Dict) -> Dict
        http_session: Optional requests.Session for external HTTP calls.
        ttl_overrides: Optional dict overriding default TTLs per source.
    """

    def __init__(
        self,
        dab_query_fn: Optional[Callable[[str, Dict], Any]] = None,
        http_session: Optional[requests.Session] = None,
        ttl_overrides: Optional[Dict[str, int]] = None,
    ):
        self._dab_query_fn = dab_query_fn
        self._session = http_session or requests.Session()
        self._session.headers.update({"User-Agent": "HR-Forecasting-Engine/1.0"})
        self._ttls = {**_DEFAULT_TTLS, **(ttl_overrides or {})}
        self._cache = TTLCache()

    # ------------------------------------------------------------------
    # DOSM fetchers
    # ------------------------------------------------------------------

    def fetch_dosm_unemployment(self, months: int = 24) -> List[Dict[str, Any]]:
        """Fetch Malaysia unemployment rate from OpenDOSM (lfs_month).

        External API (DOSM) — kept isolated from internal DAB data.
        """
        cache_key = _cache_key("dosm", "unemployment", months)
        cached = self._cache.get(cache_key, self._ttls["dosm"])
        if cached is not None:
            return cached

        # OpenDOSM dataset: Monthly Principal Labour Force Statistics
        url = f"{DOSM_BASE_URL}?id=lfs_month&limit={months}"
        results: List[Dict[str, Any]] = []

        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            rows = _ensure_list(payload.get("data", payload) if isinstance(payload, dict) else payload)
            for row in rows[:months]:
                period = row.get("date", row.get("period", ""))
                value = _safe_float(row.get("u_rate"))
                if period and value is not None:
                    results.append({"period": str(period)[:7], "unemployment_rate": value})
            logger.info("DOSM unemployment fetched %d rows", len(results))
        except Exception as exc:
            logger.warning("DOSM unemployment fetch failed: %s", exc)

        self._cache.set(cache_key, results)
        return results

    def fetch_dosm_labour_force(self, months: int = 24) -> List[Dict[str, Any]]:
        """Fetch Malaysia labour force survey data (LFPR, employed, unemployed).

        External API (DOSM) — kept isolated from internal DAB data.
        Source dataset: lfs_month (columns p_rate, lf_employed, lf_unemployed).
        """
        cache_key = _cache_key("dosm", "labour_force", months)
        cached = self._cache.get(cache_key, self._ttls["dosm"])
        if cached is not None:
            return cached

        url = f"{DOSM_BASE_URL}?id=lfs_month&limit={months}"
        results: List[Dict[str, Any]] = []

        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            rows = _ensure_list(payload.get("data", payload) if isinstance(payload, dict) else payload)
            for row in rows[:months]:
                period = row.get("date", row.get("period", ""))
                lfpr = _safe_float(row.get("p_rate"))
                employed = _safe_float(row.get("lf_employed"))
                unemployed = _safe_float(row.get("lf_unemployed"))
                if period:
                    entry: Dict[str, Any] = {"period": str(period)[:7]}
                    if lfpr is not None:
                        entry["lfpr"] = lfpr
                    if employed is not None:
                        entry["employed"] = employed
                    if unemployed is not None:
                        entry["unemployed"] = unemployed
                    results.append(entry)
            logger.info("DOSM labour force fetched %d rows", len(results))
        except Exception as exc:
            logger.warning("DOSM labour force fetch failed: %s", exc)

        self._cache.set(cache_key, results)
        return results

    def fetch_dosm_wages(self) -> List[Dict[str, Any]]:
        """Fetch formal sector wages (quarterly).

        External API (DOSM) — kept isolated from internal DAB data.
        NOTE: the exact OpenDOSM dataset id for formal-sector wages could not be
        verified against the public catalogue; ``formal-sector-wages`` is a
        best-effort id. The try/except degrades gracefully (returns []) if the
        id is wrong, so the rest of the enriched dataset is unaffected.
        """
        cache_key = _cache_key("dosm", "wages")
        cached = self._cache.get(cache_key, self._ttls["dosm"])
        if cached is not None:
            return cached

        url = f"{DOSM_BASE_URL}?id=formal-sector-wages&limit=100"
        results: List[Dict[str, Any]] = []

        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            rows = _ensure_list(payload.get("data", payload) if isinstance(payload, dict) else payload)
            for row in rows:
                period = row.get("date", row.get("period", ""))
                value = _safe_float(row.get("value", row.get("wages")))
                if period and value is not None:
                    results.append({"period": str(period)[:7], "wages": value})
            logger.info("DOSM wages fetched %d rows", len(results))
        except Exception as exc:
            logger.warning("DOSM wages fetch failed: %s", exc)

        self._cache.set(cache_key, results)
        return results

    def fetch_dosm_youth_unemployment(self, months: int = 24) -> List[Dict[str, Any]]:
        """Fetch youth unemployment (15-24 and 15-30).

        External API (DOSM) — kept isolated from internal DAB data.
        Source dataset: lfs_month_youth (columns u_rate_15_24, u_rate_15_30).
        """
        cache_key = _cache_key("dosm", "youth_unemployment", months)
        cached = self._cache.get(cache_key, self._ttls["dosm"])
        if cached is not None:
            return cached

        results: List[Dict[str, Any]] = []
        url = f"{DOSM_BASE_URL}?id=lfs_month_youth&limit={months}"
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            rows = _ensure_list(payload.get("data", payload) if isinstance(payload, dict) else payload)
            for row in rows[:months]:
                period = row.get("date", row.get("period", ""))
                if not period:
                    continue
                for age_group, col in [("15-24", "u_rate_15_24"), ("15-30", "u_rate_15_30")]:
                    value = _safe_float(row.get(col))
                    if value is not None:
                        results.append({
                            "period": str(period)[:7],
                            f"youth_unemployment_{age_group}": value,
                        })
        except Exception as exc:
            logger.warning("DOSM youth unemployment fetch failed: %s", exc)

        self._cache.set(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # World Bank fetchers
    # ------------------------------------------------------------------

    def fetch_worldbank_indicator(
        self,
        indicator: str,
        country: str = "MYS",
        years: int = 10,
    ) -> List[Dict[str, Any]]:
        """Fetch a World Bank indicator for Malaysia."""
        cache_key = _cache_key("worldbank", indicator, country, years)
        cached = self._cache.get(cache_key, self._ttls["worldbank"])
        if cached is not None:
            return cached

        end_year = datetime.now().year
        start_year = end_year - years
        url = (
            f"{WORLD_BANK_BASE_URL}/country/{country}/indicator/{indicator}"
            f"?format=json&date={start_year}:{end_year}&per_page=500"
        )
        results: List[Dict[str, Any]] = []

        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            # World Bank returns [metadata, data_array]
            if isinstance(payload, list) and len(payload) >= 2:
                rows = payload[1] or []
            else:
                rows = _ensure_list(payload)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                period = row.get("date", "")
                value = _safe_float(row.get("value"))
                if period and value is not None:
                    results.append({"period": str(period)[:7], indicator: value})
            logger.info("World Bank %s fetched %d rows", indicator, len(results))
        except Exception as exc:
            logger.warning("World Bank %s fetch failed: %s", indicator, exc)

        self._cache.set(cache_key, results)
        return results

    def fetch_malaysia_macro_bundle(self, years: int = 10) -> List[Dict[str, Any]]:
        """Fetch GDP, inflation, unemployment, population from World Bank."""
        indicators = {
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "FP.CPI.TOTL.ZG": "inflation",
            "SL.UEM.TOTL.ZS": "unemployment_rate_wb",
            "SP.POP.TOTL": "population",
        }
        merged: Dict[str, Dict[str, Any]] = {}
        for wb_code, label in indicators.items():
            rows = self.fetch_worldbank_indicator(wb_code, years=years)
            for row in rows:
                period = row["period"]
                if period not in merged:
                    merged[period] = {"period": period}
                merged[period][label] = row.get(wb_code)

        # Sort by period
        results = [merged[k] for k in sorted(merged.keys())]
        logger.info("Malaysia macro bundle fetched %d periods", len(results))
        return results

    # ------------------------------------------------------------------
    # Yahoo Finance fetchers
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, ticker: str, period: str = "2y", interval: str = "1mo") -> List[Dict[str, Any]]:
        """Generic yfinance fetcher."""
        if not _YFINANCE_AVAILABLE:
            logger.warning("yfinance not available; skipping %s", ticker)
            return []

        cache_key = _cache_key("yfinance", ticker, period, interval)
        cached = self._cache.get(cache_key, self._ttls["yfinance"])
        if cached is not None:
            return cached

        results: List[Dict[str, Any]] = []
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period, interval=interval, auto_adjust=True)
            if hist is not None and not hist.empty:
                hist.index = pd.to_datetime(hist.index)
                for ts, row in hist.iterrows():
                    period_str = ts.strftime("%Y-%m")
                    close = _safe_float(row.get("Close"))
                    if close is not None:
                        results.append({"period": period_str, ticker: close})
            logger.info("yfinance %s fetched %d rows", ticker, len(results))
        except Exception as exc:
            logger.warning("yfinance %s fetch failed: %s", ticker, exc)

        self._cache.set(cache_key, results)
        return results

    def fetch_klci_monthly(self, months: int = 24) -> List[Dict[str, Any]]:
        """Fetch KLCI monthly closing prices."""
        rows = self._fetch_yfinance("^KLSE", period="2y", interval="1mo")
        return rows[-months:] if len(rows) > months else rows

    def fetch_myr_usd(self, months: int = 24) -> List[Dict[str, Any]]:
        """Fetch MYR/USD exchange rate monthly."""
        rows = self._fetch_yfinance("MYRUSD=X", period="2y", interval="1mo")
        return rows[-months:] if len(rows) > months else rows

    # ------------------------------------------------------------------
    # Internal HR data fetcher (via DAB)
    # ------------------------------------------------------------------

    async def fetch_internal_hr_data(
        self,
        months: int = 24,
        tenant_id: str = "LOCALDEV",
    ) -> List[Dict[str, Any]]:
        """Fetch internal HR time-series from DAB.

        Queries headcount, hires, and terminations by month.
        Falls back gracefully if DAB is unavailable.
        """
        if self._dab_query_fn is None:
            logger.warning("No DAB query_fn provided; skipping internal HR data fetch")
            return []

        cache_key = _cache_key("internal_hr", tenant_id, months)
        cached = self._cache.get(cache_key, self._ttls["internal_hr"])
        if cached is not None:
            return cached

        results: List[Dict[str, Any]] = []
        now = datetime.now()

        # We will build monthly buckets for the last `months` months
        month_buckets: Dict[str, Dict[str, Any]] = {}
        for i in range(months):
            d = now - timedelta(days=30 * i)
            key = d.strftime("%Y-%m")
            month_buckets[key] = {"period": key, "headcount": 0, "hires": 0, "terminations": 0}

        try:
            # 1. Headcount: count active employees as of each month
            # Try common entity/field combinations
            headcount_queries = [
                {"tool": "aggregate_records", "args": {"entity": "V_EMP", "function": "count", "field": "EMPLOYEE_NO", "filter": "EMPLOYEE_STATUS eq 'Active'"}},
                {"tool": "aggregate_records", "args": {"entity": "Employees", "function": "count", "field": "emp_id", "filter": "employment_status eq 'Active'"}},
            ]
            for hq in headcount_queries:
                try:
                    resp = await self._dab_query_fn(hq["tool"], hq["args"])
                    count = resp.get("count", resp.get("value", 0))
                    if count:
                        # Distribute across months as a snapshot
                        for key in month_buckets:
                            month_buckets[key]["headcount"] = count
                        break
                except Exception:
                    continue

            # 2. Hires: count by hire_date / DATE_JOINED per month
            hire_queries = [
                {"tool": "aggregate_records", "args": {"entity": "V_EMP", "function": "count", "field": "EMPLOYEE_NO", "groupby": "POPER", "filter": "DATE_JOINED ge datetime'2024-01-01'"}},
                {"tool": "aggregate_records", "args": {"entity": "Employees", "function": "count", "field": "emp_id", "groupby": "hire_year_month", "filter": "hire_date ge '2024-01-01'"}},
            ]
            for hq in hire_queries:
                try:
                    resp = await self._dab_query_fn(hq["tool"], hq["args"])
                    items = resp.get("items", resp.get("result", []))
                    if isinstance(items, list):
                        for item in items:
                            period = item.get("POPER", item.get("hire_year_month", ""))
                            count = item.get("count", item.get("value", 0))
                            if period and str(period) in month_buckets:
                                month_buckets[str(period)]["hires"] = count
                        break
                except Exception:
                    continue

            # 3. Terminations: count by resignation date per month
            term_queries = [
                {"tool": "aggregate_records", "args": {"entity": "V_EMP", "function": "count", "field": "EMPLOYEE_NO", "groupby": "POPER", "filter": "DATE_RESIGNED ge datetime'2024-01-01'"}},
                {"tool": "aggregate_records", "args": {"entity": "Employees", "function": "count", "field": "emp_id", "groupby": "resignation_year_month", "filter": "employment_status eq 'Inactive'"}},
            ]
            for tq in term_queries:
                try:
                    resp = await self._dab_query_fn(tq["tool"], tq["args"])
                    items = resp.get("items", resp.get("result", []))
                    if isinstance(items, list):
                        for item in items:
                            period = item.get("POPER", item.get("resignation_year_month", ""))
                            count = item.get("count", item.get("value", 0))
                            if period and str(period) in month_buckets:
                                month_buckets[str(period)]["terminations"] = count
                        break
                except Exception:
                    continue

            results = [month_buckets[k] for k in sorted(month_buckets.keys())]
            logger.info("Internal HR data fetched %d monthly rows", len(results))
        except Exception as exc:
            logger.warning("Internal HR data fetch failed: %s", exc)

        self._cache.set(cache_key, results)
        return results

    # ------------------------------------------------------------------
    # Enriched dataset builder
    # ------------------------------------------------------------------

    async def build_enriched_dataset(
        self,
        tenant_id: str = "LOCALDEV",
        forecast_horizon_months: int = 6,
        external_months: int = 24,
    ) -> Dict[str, Any]:
        """Merge internal HR data (DAB) with external market data into one dict.

        This is the single entry point called by the forecasting pipeline.
        Internal and external sources are fetched separately and kept in
        distinct top-level keys so the sandbox can tell them apart:

          * ``hr_series``     — internal HR data from DAB (V_EMP / V_TMS_OVERTIME)
          * ``market_series`` — external APIs (DOSM, World Bank, Yahoo Finance)
          * ``merged_series`` — both joined on ``period`` (YYYY-MM)
          * ``metadata``      — provenance (internal_sources vs external_sources)

        All external data is fetched OUTSIDE the sandbox; only the merged dict
        is passed in as ``_input_data``.
        """
        # 1. Internal HR data
        hr_series = await self.fetch_internal_hr_data(months=external_months, tenant_id=tenant_id)

        # 2. External market data
        market_series: List[Dict[str, Any]] = []

        # DOSM unemployment
        try:
            market_series.extend(self.fetch_dosm_unemployment(months=external_months))
        except Exception as exc:
            logger.warning("DOSM unemployment skipped: %s", exc)

        # DOSM labour force
        try:
            market_series.extend(self.fetch_dosm_labour_force(months=external_months))
        except Exception as exc:
            logger.warning("DOSM labour force skipped: %s", exc)

        # World Bank macro bundle
        try:
            market_series.extend(self.fetch_malaysia_macro_bundle(years=5))
        except Exception as exc:
            logger.warning("World Bank macro skipped: %s", exc)

        # Yahoo Finance KLCI + MYR
        try:
            market_series.extend(self.fetch_klci_monthly(months=external_months))
        except Exception as exc:
            logger.warning("KLCI fetch skipped: %s", exc)

        try:
            market_series.extend(self.fetch_myr_usd(months=external_months))
        except Exception as exc:
            logger.warning("MYR/USD fetch skipped: %s", exc)

        # 3. Merge on period
        hr_map = {row["period"]: row for row in hr_series}
        market_map: Dict[str, Dict[str, Any]] = {}
        for row in market_series:
            p = row["period"]
            if p not in market_map:
                market_map[p] = {}
            market_map[p].update({k: v for k, v in row.items() if k != "period"})

        all_periods = sorted(set(hr_map.keys()) | set(market_map.keys()))
        merged_rows: List[Dict[str, Any]] = []
        for period in all_periods:
            row: Dict[str, Any] = {"period": period}
            row.update(hr_map.get(period, {}))
            row.update(market_map.get(period, {}))
            merged_rows.append(row)

        # 4. Build final JSON
        enriched = {
            "hr_series": hr_series,
            "market_series": market_series,
            "merged_series": merged_rows,
            "metadata": {
                "forecast_horizon_months": forecast_horizon_months,
                "data_sources": ["DAB", "DOSM", "World Bank", "Yahoo Finance"],
                "internal_sources": ["DAB:V_EMP" if tenant_id != "LOCALDEV" else "DAB:Employees"],
                "external_sources": ["OpenDOSM", "World Bank", "Yahoo Finance"],
                "generated_at": datetime.now().isoformat(),
                "tenant_id": tenant_id,
            },
        }
        logger.info(
            "Enriched dataset built: %d HR rows, %d market rows, %d merged rows",
            len(hr_series), len(market_series), len(merged_rows),
        )
        return enriched

    def save_input_json(self, enriched: Dict[str, Any]) -> str:
        """Write enriched dataset to a temp JSON file for the sandbox."""
        import tempfile
        base_dir = _FORECAST_TEMP_DIR or tempfile.gettempdir()
        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"forecast_input_{int(time.time())}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, default=str, indent=2)
        logger.info("Saved forecast input to %s", path)
        return path

    def invalidate_cache(self, source: Optional[str] = None) -> None:
        """Invalidate cached entries.

        Args:
            source: If provided, invalidate only entries whose key starts with
                ``"<source>:"`` (e.g. "dosm", "worldbank"). If None, clear all.
        """
        if source is None:
            self._cache.clear()
            return
        prefix = f"{source}:"
        # Build a new store instead of mutating during iteration.
        self._cache._store = {
            k: v for k, v in self._cache._store.items() if not k.startswith(prefix)
        }
