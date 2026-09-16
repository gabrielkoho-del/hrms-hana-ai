"""Multi-chart dashboard query builder.

Builds structured dashboard queries from multi-chart intent and provides
pre-defined chart configs for named dashboards.
"""
import logging
import re
from typing import Any, Dict, List, Optional

from agent.core.config_loader import (
    get_chart_type_patterns_for_tenant as get_chart_type_patterns,
    get_multi_chart_patterns_for_tenant as get_multi_chart_patterns,
    get_named_dashboards_for_tenant as get_named_dashboards,
    get_dimension_mappings_for_tenant as get_dimension_mappings,
)

logger = logging.getLogger("hr_agent")


def _match_named_dashboard(query: str, tenant_id: str = "default") -> Optional[str]:
    """Match query to a named dashboard pattern."""
    dashboards = get_named_dashboards(tenant_id)
    for name in dashboards.keys():
        # Build a regex from the dashboard name to match in the query
        pattern = rf"\b{name.replace('_', '\\s+')}\s+(overview|summary|dashboard)\b"
        if re.search(pattern, query, re.I):
            return name
        # Also match just the dashboard name
        if re.search(rf"\b{name.replace('_', '\\s+')}\b", query, re.I):
            return name
    return None


def _get_named_dashboard_queries(dashboard_name: str, tenant_id: str = "default") -> List[Dict]:
    """Return pre-defined chart configs for named dashboards."""
    dashboards = get_named_dashboards(tenant_id)
    return dashboards.get(dashboard_name, [{
        "dataset": "V_EMP",
        "dimensions": [],
        "metrics": [{"field": "EMPLOYEE_NO", "agg": "count"}],
        "chart_type": "bar",
        "sort": "desc",
        "title": dashboard_name.replace("_", " ").title(),
    }])


def build_dashboard_queries(user_query: str, steps: List[Dict], tenant_id: str = "default") -> List[Dict]:
    """Build structured dashboard queries from multi-chart intent.

    Parses the user's natural-language multi-chart request and converts it
    into an array of structured configs compatible with the dynamic dashboard
    ?queries= parameter.
    """
    q = user_query.lower()
    queries: List[Dict] = []

    # Check for named dashboard patterns first
    named_dashboard = _match_named_dashboard(q, tenant_id)
    if named_dashboard:
        return _get_named_dashboard_queries(named_dashboard, tenant_id)

    # Split on "and", ",", "vs", "versus" to extract individual chart requests
    # Be careful not to split inside chart type names like "pie chart"
    parts = re.split(r'\b(?:and|,|vs\.?|versus)\b', q)
    parts = [p.strip() for p in parts if p.strip()]

    # If splitting didn't produce multiple parts, treat the whole query as one chart
    if len(parts) <= 1:
        parts = [q]

    chart_type_patterns = get_chart_type_patterns(tenant_id)
    for part in parts:
        chart_type = "bar"
        if "horizontal" in part or "hbar" in part:
            chart_type = "hbar"
        elif "line" in part or "trend" in part or "over time" in part:
            chart_type = "line"
        elif "area" in part:
            chart_type = "area"
        elif "pie" in part:
            chart_type = "pie"
        elif "doughnut" in part or "donut" in part:
            chart_type = "doughnut"

        # Detect dimension from part
        dimension = None
        for pattern_info in chart_type_patterns:
            if re.search(pattern_info["pattern"], part, re.I):
                chart_type = pattern_info["type"]

        # Detect dimension from part using keyword matching
        dimension_map = get_dimension_mappings(tenant_id)
        for kw, candidates in dimension_map.items():
            if kw in part:
                for cand in candidates:
                    if cand.lower() in part:
                        dimension = cand
                        break
                if dimension:
                    break

        # Detect metric intent from part
        metric_field = "EMPLOYEE_NO"
        metric_agg = "count"
        if any(k in part for k in ("salary", "payroll", "compensation", "wage", "hourly", "rate")):
            metric_field = "HOURLY_RATE"
            metric_agg = "avg"
        elif any(k in part for k in ("performance", "rating", "score")):
            metric_field = "PERFORMANCE_RATING"
            metric_agg = "avg"
        elif any(k in part for k in ("headcount", "count", "number of", "how many", "employees")):
            metric_field = "EMPLOYEE_NO"
            metric_agg = "count"

        query_config: Dict[str, Any] = {
            "dataset": "V_EMP",
            "dimensions": [dimension] if dimension else [],
            "metrics": [{"field": metric_field, "agg": metric_agg}],
            "chart_type": chart_type,
            "sort": "desc",
            "title": part[:60],
        }
        queries.append(query_config)

    return queries if queries else [{
        "dataset": "V_EMP",
        "dimensions": [],
        "metrics": [{"field": "EMPLOYEE_NO", "agg": "count"}],
        "chart_type": "bar",
        "sort": "desc",
        "title": user_query[:60],
    }]
