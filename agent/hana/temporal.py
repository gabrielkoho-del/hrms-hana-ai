"""agent/hana/temporal.py

Temporal reasoning for HANA financial queries.

Resolves implicit year context for financial metric queries by:
  - Detecting explicit year mentions in the query
  - Querying HANA financial tables for available fiscal years
  - Defaulting to the most recent complete year when no year is specified
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.integrations.hana_client import hana_manager, normalize_hana_result, is_hana_available
from agent.core.metric_registry import financial_keywords

logger = logging.getLogger("hr_agent")

# Financial-domain keyword set is sourced from the shared metric registry
# (config/metrics.yaml) so finance + HR live in a single source of truth.
_FINANCIAL_METRIC_KEYWORDS = frozenset(kw.lower() for kw in financial_keywords())

_YEAR_PATTERN = __import__("re").compile(r"\b(20\d{2}|19\d{2})\b")


def is_financial_metric_query(user_query: str) -> bool:
    """Heuristic check whether the query is about financial metrics."""
    q = user_query.lower()
    return any(kw in q for kw in _FINANCIAL_METRIC_KEYWORDS)


def extract_year_from_query(user_query: str) -> Optional[int]:
    """Extract explicit year mention from the query, if any."""
    matches = _YEAR_PATTERN.findall(user_query)
    if not matches:
        return None
    # Prefer the last mentioned year (usually the relevant one)
    return int(matches[-1])


async def get_available_fiscal_years(tenant_id: str = "default") -> List[int]:
    """Return available fiscal years from HANA financial tables, sorted descending."""
    # If HANA is known to be down, don't attempt connections -- just return no
    # years so callers fall back to the current/most-recent year.
    if not is_hana_available():
        return []
    try:
        client = hana_manager.get_client(tenant_id)
        years = set()
        for query in [
            "SELECT DISTINCT GJAHR FROM DBADMIN.BKPF ORDER BY GJAHR DESC",
            "SELECT DISTINCT RYEAR FROM DBADMIN.FAGLFLEXT ORDER BY RYEAR DESC",
            "SELECT DISTINCT RYEAR FROM DBADMIN.GLT0 ORDER BY RYEAR DESC",
        ]:
            try:
                result = client.call_tool("hana_execute_query", {"query": query, "maxRows": 50})
                normalized = normalize_hana_result(result, "hana_execute_query")
                if isinstance(normalized, dict) and "result" in normalized:
                    for row in normalized["result"]:
                        for value in row.values():
                            if isinstance(value, (int, str)):
                                try:
                                    years.add(int(str(value).strip()))
                                except (TypeError, ValueError):
                                    pass
            except Exception:
                continue
        return sorted(years, reverse=True)
    except Exception:
        return []


def resolve_temporal_context(user_query: str, available_years: List[int]) -> Dict[str, Any]:
    """Resolve temporal context for financial queries.

    Returns a dict with:
      - resolved_year: int or None
      - is_inferred: bool
      - note: str
    """
    explicit_year = extract_year_from_query(user_query)
    if explicit_year is not None:
        return {
            "resolved_year": explicit_year,
            "is_inferred": False,
            "note": f"Using explicitly requested year {explicit_year}.",
        }

    current_year = datetime.now().year
    if not available_years:
        fallback = current_year - 1 if current_year > 2000 else current_year
        return {
            "resolved_year": fallback,
            "is_inferred": True,
            "note": f"No available fiscal years detected; defaulting to most recent complete year {fallback}.",
        }

    if current_year in available_years:
        complete_years = [y for y in available_years if y < current_year]
        if complete_years:
            resolved = complete_years[0]
            return {
                "resolved_year": resolved,
                "is_inferred": True,
                "note": f"Current year {current_year} data may be incomplete; defaulting to most recent complete year {resolved}.",
            }
        resolved = current_year
        return {
            "resolved_year": resolved,
            "is_inferred": True,
            "note": f"Only current year {current_year} is available; using it.",
        }

    resolved = available_years[0]
    return {
        "resolved_year": resolved,
        "is_inferred": True,
        "note": f"No data for current year {current_year}; defaulting to most recent available year {resolved}.",
    }
