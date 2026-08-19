# agent/tool_planner.py
"""Tool planning module for DAB (Data API Builder) -- native tool calling.

Production-grade architecture:
  Native OpenAI tool calling -- model emits structured tool_calls.

All previous fixes preserved:
  * asyncio.to_thread for non-blocking LLM calls
  * Schema-driven entity normalization
  * Aligned client-side binning example
"""
import asyncio
import json
import re
import logging
from typing import List, Dict, Optional, Any

import jsonschema

from agent.integrations.llm_client import call_llm
from agent.integrations.hana_client import normalize_hana_result, get_cached_hana_tool_schemas
from agent.integrations.hana_sql import (
    _load_hana_semantic_schema,
    _match_semantic_pattern,
    _repair_sql_with_semantic_template,
    _validate_hana_sql,
    _extract_year_from_query,
)
from agent.config import DEFAULT_MAX_ROWS, LARGE_RESULT_THRESHOLD, UNLIMITED_ROWS, PLANNER_ESTIMATED_TOKENS, CONVERSATION_HISTORY_TOKEN_BUDGET
from agent.core.prompts import (
    build_tone_aware_guidance,
    build_dab_tool_schemas,
    build_hana_tool_schemas,
    build_all_tool_schemas,
    build_tool_calling_system_prompt,
    build_user_context_rules,
    format_schema_for_prompt,
    format_schema_for_prompt_cached,
    TOKEN_BUDGET,
)
from agent.core.utils import count_tokens, TTLCache, schema_cache
from agent.hana.temporal import (
    is_financial_metric_query,
    get_available_fiscal_years,
    resolve_temporal_context,
)
from agent.hana.semantic import format_hana_semantics_for_prompt
from agent.output.chart_metadata import extract_metrics_hybrid
from agent.output.binning import (
    BINNING_MAP,
    _resolve_binning_column,
    _derive_y_label,
)

logger = logging.getLogger("hr_agent")

_PLANNER_TIMEOUT_SECONDS = 30


# =============================================================================
# DAB FILTER VALIDATION
# =============================================================================
def validate_dab_filter_permissions(filter_str: str, entity: str, auth_context: Any) -> tuple[bool, str]:
    if not auth_context or not auth_context.authenticated:
        return True, ""

    if "read:all_employees" in auth_context.permissions:
        return True, ""

    if not filter_str:
        if entity and entity.lower() in ("employee", "leave_entitlement", "employee_leave", "leave_balance_view", "v_emp"):
            return False, (
                "Access denied: Your role requires a self-filter. "
                "Please specify your emp_id (or EMPLOYEE_NO) or email in the filter."
            )
        return True, ""

    if "read:self" in auth_context.permissions:
        filt_lower = filter_str.lower()
        has_self = False

        if auth_context.email:
            email_lower = auth_context.email.lower()
            if email_lower in filt_lower:
                has_self = True

        if auth_context.emp_id:
            emp_id_str = str(auth_context.emp_id)
            # Check for either EMPLOYEE_NO (RDEMOROCKFORT) or emp_id (LOCALDEV) in filter
            if emp_id_str in filter_str:
                has_self = True

        if not has_self:
            return False, (
                f"Access denied: Your role only allows querying your own profile. "
                f"Filters must include your email ({auth_context.email}) or emp_id/EMPLOYEE_NO ({auth_context.emp_id})."
            )

    return True, ""


# =============================================================================
# NATIVE TOOL CALLING -- OpenAI-compatible schemas for DAB tools
# =============================================================================

def build_dab_tool_schemas(cached_schema: Dict) -> List[Dict]:
    """Build OpenAI function-calling schemas for DAB MCP tools.

    Schemas are static (tool interface is fixed) but enriched with
    entity names from the cached schema for better model context.
    """
    entity_examples = ", ".join(list(cached_schema.keys())[:5]) if cached_schema else "employee"

    read_records_schema = {
        "type": "function",
        "function": {
            "name": "read_records",
            "description": (
                "Query records from a DAB entity using OData filters. "
                "Use eq, ne, gt, ge, lt, le, and, or, not. "
                "Text filters do NOT support contains/LIKE -- use exact eq or fetch broader. "
                "For dates, use pre-computed fields like hire_year/hire_month -- NEVER year(hire_date). "
                f"Available entities: {entity_examples}..."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to query"
                    },
                    "select": {
                        "type": "string",
                        "description": "Comma-separated fields to select (e.g., 'emp_id,name,department')"
                    },
                    "filter": {
                        "type": "string",
                        "description": "OData filter expression (e.g., 'department eq \"Sales\"')"
                    },
                    "orderby": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sort fields like ['salary desc']"
                    },
                    "first": {
                        "type": "string",
                        "description": f"Max rows to return as string (default {DEFAULT_MAX_ROWS}, use {UNLIMITED_ROWS} for 'all')"
                    }
                },
                "required": ["entity"]
            }
        }
    }

    aggregate_records_schema = {
        "type": "function",
        "function": {
            "name": "aggregate_records",
            "description": (
                "Aggregate data from a DAB entity. Use for counts, sums, averages, grouped results. "
                "Use * for count. groupby for grouped results. "
                f"Available entities: {entity_examples}..."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to aggregate"
                    },
                    "function": {
                        "type": "string",
                        "enum": ["count", "sum", "avg", "min", "max"],
                        "description": "Aggregation function"
                    },
                    "field": {
                        "type": "string",
                        "description": "Field to aggregate (use * for count)"
                    },
                    "groupby": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to group by"
                    },
                    "orderby": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sort fields"
                    },
                    "filter": {
                        "type": "string",
                        "description": "OData filter expression"
                    },
                    "having": {
                        "type": "string",
                        "description": "Having clause for group filtering"
                    },
                    "first": {
                        "type": "string",
                        "description": "Max groups to return as string (default 20, use -1 for all)"
                    }
                },
                "required": ["entity", "function"]
            }
        }
    }

    describe_entities_schema = {
        "type": "function",
        "function": {
            "name": "describe_entities",
            "description": "Discover entity schema and fields. Use only when uncertain about available columns or entity names.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

    return [read_records_schema, aggregate_records_schema, describe_entities_schema]


def build_hana_tool_schemas(tenant_id: Optional[str] = None) -> List[Dict]:
    """Build OpenAI function-calling schemas for SAP HANA MCP tools.

    Uses dynamically discovered tool schemas when available, otherwise
    falls back to the static schema list.
    """
    return get_cached_hana_tool_schemas(tenant_id=tenant_id)


def build_superset_tool_schemas() -> List[Dict]:
    """Build OpenAI function-calling schemas for Superset MCP dashboard tools."""
    list_datasets_schema = {
        "type": "function",
        "function": {
            "name": "superset_list_datasets",
            "description": "List available Superset datasets. Use to discover what data sources are available for dashboard building.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional filters to narrow dataset list (e.g., by table_name containing 'hr')"
                    }
                },
                "required": []
            }
        }
    }

    get_dataset_info_schema = {
        "type": "function",
        "function": {
            "name": "superset_get_dataset_info",
            "description": "Get schema, columns, and metadata for a specific Superset dataset. Use after selecting a dataset to understand available metrics and dimensions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {
                        "type": "integer",
                        "description": "ID of the dataset to inspect"
                    }
                },
                "required": ["dataset_id"]
            }
        }
    }

    generate_chart_schema = {
        "type": "function",
        "function": {
            "name": "superset_generate_chart",
            "description": "Generate a chart preview or save it to Superset. Use save_chart=False for preview, save_chart=True to persist. Returns explore_url for preview and chart_id when saved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "integer", "description": "Dataset to query"},
                    "viz_type": {"type": "string", "description": "Chart type: bar, line, pie, big_number, table, etc."},
                    "metrics": {"type": "array", "items": {"type": "string"}, "description": "Metrics to plot"},
                    "groupby": {"type": "array", "items": {"type": "string"}, "description": "Dimensions to group by"},
                    "filters": {"type": "array", "items": {"type": "object"}, "description": "Superset filter config"},
                    "save_chart": {"type": "boolean", "description": "True to persist chart, False for preview only"},
                    "chart_name": {"type": "string", "description": "Name for saved chart"}
                },
                "required": ["dataset_id", "viz_type"]
            }
        }
    }

    generate_dashboard_schema = {
        "type": "function",
        "function": {
            "name": "superset_generate_dashboard",
            "description": "Build a Superset dashboard from saved chart IDs with auto-layout. Use only after user confirms previewed charts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboard_title": {"type": "string", "description": "Title for the new dashboard"},
                    "charts": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of chart specs or saved chart IDs"
                    },
                    "auto_layout": {"type": "boolean", "description": "Use automatic grid layout"},
                    "layout_mode": {"type": "string", "description": "Layout mode: grid or free_form"},
                    "description": {"type": "string", "description": "Dashboard description"}
                },
                "required": ["dashboard_title"]
            }
        }
    }

    execute_sql_schema = {
        "type": "function",
        "function": {
            "name": "superset_execute_sql",
            "description": "Execute SQL against a Superset dataset or database. Use for ad-hoc queries when pre-built datasets are insufficient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "dataset_id": {"type": "integer", "description": "Optional dataset context"},
                    "limit": {"type": "integer", "description": "Max rows to return"}
                },
                "required": ["query"]
            }
        }
    }

    create_virtual_dataset_schema = {
        "type": "function",
        "function": {
            "name": "superset_create_virtual_dataset",
            "description": "Create a virtual dataset in Superset from a SQL query. Use when existing datasets don't expose the needed metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset name"},
                    "sql": {"type": "string", "description": "SQL query defining the dataset"},
                    "schema": {"type": "string", "description": "Database schema"},
                    "database_id": {"type": "integer", "description": "Superset database ID"}
                },
                "required": ["name", "sql"]
            }
        }
    }

    add_chart_to_existing_dashboard_schema = {
        "type": "function",
        "function": {
            "name": "superset_add_chart_to_existing_dashboard",
            "description": "Add an already-saved chart to an existing dashboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dashboard_id": {"type": "integer", "description": "Target dashboard ID"},
                    "chart_id": {"type": "integer", "description": "Chart to add"},
                    "position": {"type": "object", "description": "Optional position config"}
                },
                "required": ["dashboard_id", "chart_id"]
            }
        }
    }

    return [
        list_datasets_schema,
        get_dataset_info_schema,
        generate_chart_schema,
        generate_dashboard_schema,
        execute_sql_schema,
        create_virtual_dataset_schema,
        add_chart_to_existing_dashboard_schema,
    ]


# =============================================================================
# TOOL SCHEMA VALIDATION -- Fail fast on invalid tool arguments
# =============================================================================

# Cached tool schemas for validation: {tool_name: {properties, required}}
_TOOL_SCHEMA_MAP: Dict[str, Dict] = {}


def _build_tool_schema_map(cached_schema: Dict, include_hana: bool = True) -> Dict[str, Dict]:
    """Build a lookup map of tool_name -> {properties, required} for validation."""
    global _TOOL_SCHEMA_MAP
    schemas = build_all_tool_schemas(cached_schema, include_hana=include_hana)
    _TOOL_SCHEMA_MAP = {}
    for schema in schemas:
        func = schema.get("function", {})
        name = func.get("name")
        if name:
            params = func.get("parameters", {})
            _TOOL_SCHEMA_MAP[name] = {
                "properties": params.get("properties", {}),
                "required": params.get("required", []),
            }
    return _TOOL_SCHEMA_MAP


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """Validate tool arguments against the registered JSON schema.

    Returns error message string if validation fails, None if valid.
    """
    schema = _TOOL_SCHEMA_MAP.get(tool_name)
    if not schema:
        return None  # No schema registered; allow through

    try:
        jsonschema.validate(instance=args, schema=schema)
        return None
    except jsonschema.ValidationError as e:
        return f"Invalid arguments for {tool_name}: {e.message}"


def build_all_tool_schemas(cached_schema: Dict, include_hana: bool = True, tenant_id: Optional[str] = None, include_superset: bool = True) -> List[Dict]:
    """Merge DAB, HANA, and Superset tool schemas for the planner prompt.

    Args:
        cached_schema: DAB entity schema map.
        include_hana: If True, append HANA schemas (requires HANA server reachable).
        tenant_id: Tenant identifier for HANA tool cache lookup.
        include_superset: If True, append Superset dashboard schemas.
    """
    schemas = list(build_dab_tool_schemas(cached_schema))
    if include_hana:
        schemas.extend(build_hana_tool_schemas(tenant_id=tenant_id))
    if include_superset:
        schemas.extend(build_superset_tool_schemas())
    return schemas


_ALL_DAB_TOOLS = {"read_records", "aggregate_records", "describe_entities"}
_ALL_SUPERSET_TOOLS = {
    "superset_list_datasets",
    "superset_get_dataset_info",
    "superset_generate_chart",
    "superset_generate_dashboard",
    "superset_execute_sql",
    "superset_create_virtual_dataset",
    "superset_add_chart_to_existing_dashboard",
}


def extract_steps_from_tool_calls(
    tool_calls: List[Dict],
    allowed_tools: Optional[set] = None,
    tool_schema_map: Optional[Dict[str, Dict]] = None,
    tenant_id: Optional[str] = None,
) -> List[Dict]:
    """Convert LLM tool_calls to plan steps.

    Args:
        tool_calls: Raw tool_calls from LLM response.
        allowed_tools: Set of tool names to permit. If None, allow DAB + dynamically discovered HANA tools.
        tool_schema_map: Optional map of tool_name -> schema for argument validation.
        tenant_id: Tenant identifier for HANA tool cache lookup.
    """
    if allowed_tools is None:
        hana_schemas = get_cached_hana_tool_schemas(tenant_id=tenant_id)
        hana_tools = {
            s["function"]["name"]
            for s in hana_schemas
            if isinstance(s, dict) and s.get("function", {}).get("name")
        }
        allowed_tools = _ALL_DAB_TOOLS | hana_tools | _ALL_SUPERSET_TOOLS

    steps = []
    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        if tool_name not in allowed_tools:
            logger.warning("LLM tried to call unauthorized tool '%s' -- skipping", tool_name)
            continue
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tool arguments for %s: %s", tool_name, e)
            continue

        # Validate args against tool schema if available
        if tool_schema_map and tool_name in tool_schema_map:
            validation_error = validate_tool_args(tool_name, args)
            if validation_error:
                logger.warning("LLM tool arg validation failed for %s: %s -- skipping", tool_name, validation_error)
                continue

        steps.append({"tool": tool_name, "args": args})
    return steps


def _normalize_entity_names(steps: List[Dict], cached_schema: Dict) -> List[Dict]:
    """
    Post-processor: normalize entity names in DAB tool-call steps to match schema exactly.
    HANA tools use schema_name/table_name or raw SQL; they are skipped here.
    """
    if not steps or not cached_schema:
        return steps

    schema_names = list(cached_schema.keys()) if isinstance(cached_schema, dict) else []
    lower_to_exact = {k.lower(): k for k in schema_names}

    for step in steps:
        tool = step.get("tool", "")
        if tool not in _ALL_DAB_TOOLS:
            continue

        entity = step.get("args", {}).get("entity", "")
        if not entity:
            continue

        if entity in schema_names:
            continue

        matched = lower_to_exact.get(entity.lower())
        if matched:
            logger.warning("Planner entity case-corrected: %s -> %s", entity, matched)
            step["args"]["entity"] = matched
            continue

        for name in schema_names:
            if name.lower().startswith(entity.lower()) or name.lower().endswith(entity.lower()):
                logger.warning("Planner entity fuzzy-corrected: %s -> %s", entity, name)
                step["args"]["entity"] = name
                break

    return steps


# =============================================================================
# METADATA INFERENCE -- Heuristic chart/rag/binning from query + steps
# =============================================================================

# Chart intent patterns using word boundaries to avoid false positives
# (e.g., "chartroom" should not match "chart").
_CHART_TYPE_PATTERNS = [
    (re.compile(r"\bbar\s+(chart|graph|plot)\b", re.I), "bar"),
    (re.compile(r"\bcolumn\s+(chart|graph|plot)\b", re.I), "bar"),
    (re.compile(r"\bpie\s+(chart|graph|plot|donut)\b", re.I), "pie"),
    (re.compile(r"\bline\s+(chart|graph|plot|trend|time\s*series)\b", re.I), "line"),
    (re.compile(r"\bhistogram\b|\bhist\s+chart\b|\bdistribution\s+chart\b", re.I), "hist"),
    (re.compile(r"\bbox\s*plot\b|\bbox\s*chart\b|\bboxplot\b", re.I), "box"),
    (re.compile(r"\bgauge\s+chart\b|\bkpi\s+gauge\b|\bgauge\b", re.I), "gauge"),
]

# General chart intent: user wants a visual representation, but didn't specify type.
# Uses word boundaries to avoid matching "chartroom", "epicchart", etc.
_GENERAL_CHART_PATTERN = re.compile(
    r"\b(chart|graph|visualize|visualization|plot|dashboard|kpi)\b",
    re.I,
)


def _detect_chart_intent(user_query: str) -> Dict[str, Any]:
    """Detect chart intent from user query using regex with word boundaries.

    Returns dict with:
      - explicit_chart_type: str or None
      - general_chart_intent: bool
    """
    query_lower = user_query.lower()
    explicit_chart_type = None
    for pattern, chart_type in _CHART_TYPE_PATTERNS:
        if pattern.search(query_lower):
            explicit_chart_type = chart_type
            break

    general_chart_intent = bool(_GENERAL_CHART_PATTERN.search(query_lower))
    return {
        "explicit_chart_type": explicit_chart_type,
        "general_chart_intent": general_chart_intent,
    }


async def infer_metadata(user_query: str, steps: List[Dict], tone_context: Optional[Dict]) -> Dict:
    """Infer chart, RAG, client-side binning, and other metadata from query and steps.

    Uses hybrid metric extraction: fast regex gate + LLM fallback with TTL cache.
    """
    metadata = {
        "chart": None,
        "rag": False,
        "rag_query": "",
        "needs_export": False,
        "action_context": "",
        "client_side_binning": None,
        "reasoning": "",
    }

    query_lower = user_query.lower()
    tone_context = tone_context or {}
    category = tone_context.get("intent_category", "policy_info")

    has_aggregate = any(s["tool"] == "aggregate_records" for s in steps)
    has_read = any(s["tool"] == "read_records" for s in steps)
    has_hana = any(s["tool"].startswith("hana_") for s in steps)

    # Detect explicit chart type request from user query (must happen before HANA early return)
    chart_intent = _detect_chart_intent(user_query)
    explicit_chart_type = chart_intent["explicit_chart_type"]
    general_chart_intent = chart_intent["general_chart_intent"]
    if explicit_chart_type:
        logger.info("Planner: explicit chart type detected from query: %s", explicit_chart_type)

    # Also treat tone_context chart_eligible=True as implicit chart intent.
    tone_context = tone_context or {}
    chart_eligible = tone_context.get("chart_eligible", False)
    if not general_chart_intent and chart_eligible and has_hana:
        general_chart_intent = True

    # Detect multi-metric financial report intent using hybrid extraction.
    unique_metrics = await extract_metrics_hybrid(user_query)
    is_multi_metric = len(unique_metrics) >= 2
    if is_multi_metric:
        logger.info("Planner: multi-metric report detected: %s", unique_metrics)

    # HANA queries use raw SQL; skip DAB chart/binning metadata inference
    # BUT preserve explicit chart requests so executor can still generate the chart
    if has_hana and not has_aggregate and not has_read:
        metadata["reasoning"] = f"Selected {len(steps)} HANA tool(s) for direct SQL query."
        if explicit_chart_type or general_chart_intent:
            chart_type = explicit_chart_type or "bar"
            metadata["chart"] = {
                "type": chart_type,
                "x_column": "",
                "y_column": "",
                "y_label": "",
                "title": "",
                "gauge_min": 0 if chart_type == "gauge" else None,
                "gauge_max": 100 if chart_type == "gauge" else None,
                "gauge_threshold": 70 if chart_type == "gauge" else None,
                "trend_line": chart_type == "line" and any(kw in query_lower for kw in ("trend line", "trend", "with trend")),
                "multi_series": is_multi_metric,
                "metrics": unique_metrics if is_multi_metric else [],
            }
        return metadata

    # -- Chart inference (intent category + data shape, NOT keywords) --
    chart_eligible = tone_context.get("chart_eligible", False) if tone_context else False

    # Must also have aggregate step with groupby OR read_records with 2-column select
    has_aggregate_with_groupby = any(
        s["tool"] == "aggregate_records" and s.get("args", {}).get("groupby")
        for s in steps
    )
    has_read_with_2col = any(
        s["tool"] == "read_records" and
        len([f for f in (s.get("args", {}).get("select", "") or "").split(",") if f.strip()]) == 2
        for s in steps
    )
    has_groupby_like = has_aggregate_with_groupby or has_read_with_2col

    # Row count guard: 1-5 rows = no chart (too small), 6-50 = chart, 50+ = chart + export
    # We don't know row count yet, so we plan the chart and let executor decide later
    wants_chart = chart_eligible and has_groupby_like

    if wants_chart:
        chart_type = "bar"
        x_col = ""
        y_col = ""
        title = "Distribution"
        y_label = ""

        for step in steps:
            if step["tool"] == "aggregate_records":
                args = step.get("args", {})
                groupby = args.get("groupby", [])
                if groupby:
                    x_col = groupby[-1]  # Most granular

                func = args.get("function", "count")
                field = args.get("field", "")
                y_col = "count" if func == "count" else (field or "value")
                y_label = _derive_y_label(func, args.get("entity", ""), field)

                groupby_count = len(groupby)
                if explicit_chart_type:
                    chart_type = explicit_chart_type
                elif groupby_count == 0:
                    chart_type = "hist" if func == "count" else "bar"
                elif groupby_count == 1:
                    cat_col = groupby[0]
                    if func == "count" and any(suffix in cat_col for suffix in ["_group", "_range", "_band"]):
                        chart_type = "bar"
                    elif any(dim in cat_col for dim in ["year", "month", "quarter"]):
                        chart_type = "line"
                    elif func == "count":
                        # Small categorical -> pie; let executor decide based on row count
                        chart_type = "pie"
                    else:
                        chart_type = "bar"
                elif groupby_count >= 2:
                    chart_type = "bar"  # Multi-series grouped bar (matplotlib only)

                # Add filter context to title
                having = args.get("having", "")
                filter_str = args.get("filter", "")
                context_suffix = ""
                if having:
                    context_suffix = f" (filtered: {having})"
                elif filter_str:
                    context_suffix = " (filtered)"
                title = f"{x_col.replace('_', ' ').title()} Distribution{context_suffix}" if x_col else "Distribution"
                break

        metadata["chart"] = {
            "type": chart_type,
            "x_column": x_col,
            "y_column": y_col,
            "y_label": y_label,
            "title": title,
            "gauge_min": 0 if chart_type == "gauge" else None,
            "gauge_max": 100 if chart_type == "gauge" else None,
            "gauge_threshold": 70 if chart_type == "gauge" else None,
            "trend_line": chart_type == "line" and any(kw in query_lower for kw in ("trend line", "trend", "with trend")),
        }

    # -- Dashboard inference (Superset dashboard building) --
    has_superset_dashboard = any(s["tool"] == "superset_generate_dashboard" for s in steps)
    has_superset_chart = any(s["tool"] == "superset_generate_chart" for s in steps)
    has_superset_dataset = any(s["tool"] in ("superset_list_datasets", "superset_get_dataset_info") for s in steps)

    if category == "dashboard_building" or has_superset_dashboard or has_superset_chart:
        metadata["dashboard_build"] = {
            "has_superset_dashboard": has_superset_dashboard,
            "has_superset_chart": has_superset_chart,
            "has_superset_dataset": has_superset_dataset,
            "pending_preview_charts": [],
            "confirmed_chart_ids": [],
            "dashboard_title": user_query[:80],
        }
        logger.info("Planner: dashboard_build metadata inferred (category=%s, dashboard=%s, chart=%s, dataset=%s)",
                     category, has_superset_dashboard, has_superset_chart, has_superset_dataset)

    # -- RAG inference --
    rag_keywords = ["policy", "rule", "handbook", "procedure", "guideline", "entitled", "eligible", "how do i", "how to"]
    if any(kw in query_lower for kw in rag_keywords) or category == "policy_info":
        metadata["rag"] = True
        metadata["rag_query"] = user_query

    # -- Export intent inference --
    export_keywords = ("export", "download", "save", "excel", "spreadsheet",
                         "xlsx", "workbook", "file", "send me", "give me the data")
    wants_export = (
        any(kw in query_lower for kw in export_keywords)
        and category == "aggregate_data"
        and (has_aggregate or has_read)
    )
    if wants_export:
        metadata["needs_export"] = True

    # -- Client-side binning inference --
    if has_aggregate:
        for step in steps:
            if step["tool"] == "aggregate_records":
                args = step.get("args", {})
                groupby = args.get("groupby", [])
                if len(groupby) == 1:
                    raw_col = groupby[0]
                    canonical = _resolve_binning_column(raw_col)
                    if canonical:
                        bin_config = dict(BINNING_MAP[canonical]["config"])
                        bin_config["column"] = raw_col
                        metadata["client_side_binning"] = bin_config
                        logger.info("infer_metadata: resolved binning for '%s' -> canonical '%s'", raw_col, canonical)

    # -- Action context --
    metadata["action_context"] = tone_context.get("action_context", "")

    # -- Reasoning --
    metadata["reasoning"] = (
        f"Selected {len(steps)} tool(s) based on query intent ({category}). "
        f"Chart={'yes' if metadata['chart'] else 'no'}, RAG={'yes' if metadata['rag'] else 'no'}, "
        f"Export={'yes' if metadata['needs_export'] else 'no'}."
    )

    return metadata


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

async def build_tool_plan(
    user_query: str,
    conversation_history: str,
    auth_context: Any,
    cached_tools: List[Dict],
    cached_schema: Dict,
    large_result_threshold: int = LARGE_RESULT_THRESHOLD,
    default_max_rows: int = DEFAULT_MAX_ROWS,
    rag_available: bool = False,
    tone_context: Optional[Dict] = None,
    hana_schema_registry: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Build an agentic, permission-aware tool plan for DAB using native tool calling."""
    if not cached_tools:
        return _empty_plan()

    tenant_id = getattr(auth_context, "tenant_id", "default") if auth_context else "default"

    # =======================================================================
    # FORECASTING PIPELINE -- bypass native tool calling for forecasting queries
    # =======================================================================
    intent = (tone_context or {}).get("intent", "")
    if intent == "forecasting_query":
        logger.info("Planner: forecasting_query detected -- returning forecasting pipeline plan")
        plan = {
            "steps": [
                {"tool": "fetch_external_data", "args": {"tenant_id": tenant_id}},
                {"tool": "generate_code", "args": {"model_type": "statsforecast"}},
                {"tool": "execute_sandbox", "args": {}},
                {"tool": "summarize_forecast", "args": {}},
            ],
            "direct_answer": "",
            "chart": None,
            "rag": False,
            "rag_query": "",
            "needs_export": False,
            "reasoning": "Forecasting pipeline: external data -> codegen -> sandbox -> summarize",
            "action_context": "",
            "client_side_binning": None,
        }
        return plan
    schema_block = await format_schema_for_prompt_cached(
        cached_schema, tenant_id=tenant_id, user_query=user_query
    )
    schema_tokens = count_tokens(schema_block)
    if schema_tokens > TOKEN_BUDGET:
        logger.warning(
            "Schema block exceeds token budget (%d > %d). Progressive pruning applied.",
            schema_tokens, TOKEN_BUDGET
        )

    available_entities = list(cached_schema.keys()) if isinstance(cached_schema, dict) else []
    entity_list = ", ".join(available_entities) if available_entities else "employee"
    user_context = build_user_context_rules(auth_context)
    rag_status = "AVAILABLE" if rag_available else "NOT AVAILABLE"
    tone_guidance = build_tone_aware_guidance(tone_context)

    full_query = user_query
    if conversation_history:
        history_tokens = count_tokens(conversation_history)
        if history_tokens > CONVERSATION_HISTORY_TOKEN_BUDGET:
            logger.warning(
                "Conversation history exceeds token budget (%d > %d). Truncating.",
                history_tokens, CONVERSATION_HISTORY_TOKEN_BUDGET
            )
            # Truncate from the start to preserve most recent context
            # Rough truncation by character budget: keep the tail that fits
            budget_chars = CONVERSATION_HISTORY_TOKEN_BUDGET * 3  # conservative chars-per-token
            truncated = conversation_history[-budget_chars:]
            # Try to start at a newline to avoid mid-message cuts
            newline_idx = truncated.find('\n')
            if newline_idx > 0:
                truncated = truncated[newline_idx + 1:]
            conversation_history = truncated
            logger.info(
                "Conversation history truncated to ~%d tokens",
                count_tokens(conversation_history)
            )
        full_query = "Previous conversation:\n" + conversation_history + "\n\nCurrent question: " + user_query

    # =======================================================================
    # Native tool calling
    # =======================================================================
    all_tools = build_all_tool_schemas(cached_schema, include_hana=True, tenant_id=tenant_id)
    tool_schema_map = _build_tool_schema_map(cached_schema, include_hana=True)

    hana_semantic_schema = _load_hana_semantic_schema()
    hana_semantic_block = format_hana_semantics_for_prompt(
        hana_semantic_schema,
        user_query=user_query,
        registry=hana_schema_registry,
    )

    system_tc = build_tool_calling_system_prompt(
        schema_block, entity_list, user_context, rag_status, tone_guidance,
        hana_schema_registry=hana_schema_registry,
        hana_semantic_block=hana_semantic_block,
    )

    logger.info("Planner: native tool calling (system=%d tokens, tools=%d)", count_tokens(system_tc), len(all_tools))
    # Log tool names for debugging
    tool_names = []
    for t in all_tools:
        if isinstance(t, dict):
            func = t.get("function", {})
            if func:
                name = func.get("name")
                if name:
                    tool_names.append(name)
    logger.info("Planner: available tools: %s", ", ".join(tool_names))

    estimated = PLANNER_ESTIMATED_TOKENS
    logger.info("Planner: about to call LLM with %d tools", len(all_tools))
    choice = await asyncio.wait_for(
        asyncio.to_thread(
            call_llm, system_tc, full_query,
            tools=all_tools, temperature=0.1, json_mode=False,
            tier="planner", estimated_tokens=estimated
        ),
        timeout=_PLANNER_TIMEOUT_SECONDS,
    )
    logger.info("Planner: LLM call completed. choice is %s", "None" if choice is None else "present")
    if choice and choice.get("message", {}).get("tool_calls"):
        tc_list = choice["message"]["tool_calls"]
        logger.info("Planner: LLM returned %d tool calls: %s", len(tc_list), 
                    [t.get("function",{}).get("name", "?") for t in tc_list])
    else:
        logger.info("Planner: LLM returned no tool calls")

    steps = []
    if choice and choice.get("message", {}).get("tool_calls"):
        steps = extract_steps_from_tool_calls(choice["message"]["tool_calls"], tool_schema_map=tool_schema_map, tenant_id=tenant_id)
        steps = _normalize_entity_names(steps, cached_schema)
        logger.info("Planner: Tool calling produced %d steps: %s", len(steps), [s["tool"] for s in steps])

        # -- SQL validation and semantic-template repair ---------------------
        validated_steps = []
        for step in steps:
            if step.get("tool") == "hana_execute_query":
                sql = step.get("args", {}).get("query", "")
                is_valid, error = _validate_hana_sql(sql)
                if not is_valid:
                    logger.warning("HANA SQL validation failed: %s. Query: %.200s", error, sql)
                    repaired = _repair_sql_with_semantic_template(sql, user_query, hana_semantic_schema, tenant_id)
                    if repaired:
                        step = dict(step)
                        step["args"] = dict(step.get("args", {}))
                        step["args"]["query"] = repaired
                        logger.info("HANA SQL repaired with semantic template")
                    else:
                        logger.error("HANA SQL rejected, dropping step: %s", error)
                        continue
            validated_steps.append(step)
        steps = validated_steps

    if steps:
        # Infer metadata from query + steps (deterministic, no extra API call)
        metadata = await infer_metadata(user_query, steps, tone_context)

        # Native tool calling provides structured tool_calls; metadata comes from infer_metadata

        plan = {
            "steps": steps,
            "direct_answer": "",
            "chart": metadata.get("chart"),
            "rag": metadata.get("rag", False),
            "rag_query": metadata.get("rag_query", ""),
            "needs_export": metadata.get("needs_export", False),
            "reasoning": metadata.get("reasoning", "Native tool calling from LLM"),
            "action_context": metadata.get("action_context", ""),
            "client_side_binning": metadata.get("client_side_binning"),
        }

        # =======================================================================
        # TEMPORAL REASONING -- resolve year for financial queries
        # =======================================================================
        if is_financial_metric_query(user_query):
            available_years = await get_available_fiscal_years(tenant_id)
            temporal = resolve_temporal_context(user_query, available_years)
            if temporal.get("resolved_year") is not None:
                plan["temporal_context"] = temporal
                plan["reasoning"] += " " + temporal.get("note", "")

        logger.info("Planner: SUCCESS -- %d steps, chart=%s, rag=%s, binning=%s",
                   len(plan["steps"]),
                   "yes" if plan["chart"] else "no",
                   "yes" if plan["rag"] else "no",
                   "yes" if plan["client_side_binning"] else "no")
        return plan

    # No tool steps produced by the model. Treat as empty plan rather than silently degraded JSON parsing.
    logger.warning("Planner: native tool calling produced no steps. Returning empty plan.")
    return _empty_plan()