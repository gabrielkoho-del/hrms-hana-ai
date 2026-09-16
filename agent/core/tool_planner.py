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
from agent.actions.strategies.leave_entitlement_guard import ensure_leave_entitlement_entity
from agent.output.chart_planner import infer_metadata, classify_chart_intent
from agent.actions import inject_all_actions

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
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort direction for grouped results by aggregated value (requires groupby; default desc)"
                    },
                    "filter": {
                        "type": "string",
                        "description": "OData filter expression"
                    },
                    "having": {
                        "type": "object",
                        "description": "Filter groups by aggregated value. Operators: eq, neq, gt, gte, lt, lte, in. Requires groupby.",
                        "properties": {
                            "eq": {"type": "number"},
                            "neq": {"type": "number"},
                            "gt": {"type": "number"},
                            "gte": {"type": "number"},
                            "lt": {"type": "number"},
                            "lte": {"type": "number"},
                            "in": {"type": "array", "items": {"type": "number"}}
                        },
                        "additionalProperties": False,
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


def build_all_tool_schemas(cached_schema: Dict, include_hana: bool = True, tenant_id: Optional[str] = None) -> List[Dict]:
    """Merge DAB and HANA tool schemas for the planner prompt.

    Args:
        cached_schema: DAB entity schema map.
        include_hana: If True, append HANA schemas (requires HANA server reachable).
        tenant_id: Tenant identifier for HANA tool cache lookup.
    """
    schemas = list(build_dab_tool_schemas(cached_schema))
    if include_hana:
        schemas.extend(build_hana_tool_schemas(tenant_id=tenant_id))
    return schemas


_ALL_DAB_TOOLS = {"read_records", "aggregate_records", "describe_entities", "create_record", "update_record"}


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
        allowed_tools = _ALL_DAB_TOOLS | hana_tools

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
# METADATA INFERENCE -- delegated to agent.output.chart_planner
# =============================================================================


def _build_dashboard_steps_from_queries(
    dashboard_queries: List[Dict],
    hana_schema_registry: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    """Convert named-dashboard chart configs into execution steps.

    For V_EMP (DAB): generates aggregate_records steps.
    For HANA tables: generates hana_execute_query steps.
    Replaces the LLM's generic schema-discovery steps with actual data retrieval.
    """
    if not dashboard_queries:
        return []

    registry = hana_schema_registry or {}
    steps: List[Dict] = []

    for q in dashboard_queries:
        dataset = q.get("dataset", "")
        dimensions = q.get("dimensions", [])
        metrics = q.get("metrics", [])
        chart_filter = q.get("filter") or ""
        sort = q.get("sort", "desc")

        if not metrics:
            continue

        metric = metrics[0]
        metric_field = metric.get("field", "")
        metric_agg = metric.get("agg", "count")

        # Determine if this is a HANA dataset (check against registered HANA tables)
        is_hana = False
        if registry:
            is_hana = any(dataset.upper() == t.upper() for tables in registry.values() for t in tables)

        if is_hana:
            # HANA: build SQL query
            agg_expr = f"{metric_agg.upper()}({metric_field})" if metric_agg != "count" else "COUNT(*)"
            dim_cols = ", ".join(dimensions) if dimensions else "*"
            groupby_clause = f" GROUP BY {dim_cols}" if dimensions else ""
            order_clause = f" ORDER BY 1 DESC" if sort == "desc" and groupby_clause else ""
            filter_clause = f" WHERE {chart_filter}" if chart_filter else ""
            sql = f"SELECT {dim_cols}, {agg_expr} AS m FROM {dataset}{filter_clause}{groupby_clause}{order_clause} LIMIT 50"
            steps.append({"tool": "hana_execute_query", "args": {"query": sql}})
        else:
            # DAB: build aggregate_records step
            agg_func = metric_agg if metric_agg in ("count", "sum", "avg", "min", "max") else "count"
            agg_field = metric_field if agg_func != "count" else "*"
            # DAB aggregate_records 'orderby' is a direction string ("asc"/"desc")
            # that defaults to "desc" when omitted (verified against shipped DAB
            # source AggregateRecordsTool.cs; the docs' array format is a
            # read_records-only convention). Dashboard configs want descending
            # sort, which matches the server default, so orderby is omitted for
            # "desc" and only sent explicitly for non-default sorts.
            orderby: Optional[str] = "asc" if sort == "asc" else None
            step_args: Dict[str, Any] = {
                "entity": dataset,
                "function": agg_func,
                "field": agg_field,
                "groupby": dimensions,
                "first": "50",
            }
            if orderby:
                step_args["orderby"] = orderby
            if chart_filter:
                step_args["filter"] = chart_filter
            steps.append({"tool": "aggregate_records", "args": step_args})

    return steps


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

def _empty_plan(pending_slots: Optional[List[Dict]] = None) -> Dict:
    """Empty plan returned when no cached tools or no steps were produced.

    Mirrors the full plan shape (see the plan dict built at the end of
    build_tool_plan) so downstream consumers never need special-casing.

    Args:
        pending_slots: Optional list of pending slot dicts from action injection.
            Preserved so the executor can surface clarification questions even
            when the LLM produced no tool-call steps.
    """
    return {
        "steps": [],
        "direct_answer": "",
        "chart": None,
        "rag": False,
        "rag_query": "",
        "needs_export": False,
        "reasoning": "No tool steps produced",
        "action_context": "",
        "client_side_binning": None,
        "chart_intent_clarification_needed": False,
        "chart_intent_clarification_reason": "",
        "kpi_only": False,
        "multi_chart": False,
        "dashboard_queries": [],
        "pending_slots": pending_slots or [],
    }


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
    # =======================================================================
    # CHART INTENT CLASSIFICATION -- structured LLM-based multi-chart detection
    # Runs early so tone_context["chart_intent"] is available to infer_metadata.
    # =======================================================================
    if tone_context is None:
        tone_context = {}
    chart_intent_result = await classify_chart_intent(
        user_query,
        conversation_history=conversation_history,
        tone_context=tone_context,
        tenant_id=tenant_id,
    )
    tone_context = dict(tone_context)  # Don't mutate caller's dict
    tone_context["chart_intent"] = chart_intent_result
    if chart_intent_result.needs_clarification:
        logger.info(
            "Chart intent: %s (confidence=%.2f) -- clarification recommended: %s",
            chart_intent_result.intent, chart_intent_result.confidence,
            chart_intent_result.clarification_reason,
        )
    else:
        logger.info(
            "Chart intent: %s (confidence=%.2f)",
            chart_intent_result.intent, chart_intent_result.confidence,
        )

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
    pending_slots: List[Dict] = []
    if choice and choice.get("message", {}).get("tool_calls"):
        steps = extract_steps_from_tool_calls(choice["message"]["tool_calls"], tool_schema_map=tool_schema_map, tenant_id=tenant_id)
        steps = _normalize_entity_names(steps, cached_schema)
        logger.info("Planner: Tool calling produced %d steps: %s", len(steps), [s.get("tool", "unknown") for s in steps])

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

        # -- Leave action injection is handled by inject_all_actions below --

        # -- Leave entitlement query guard -- ensures entitlement/balance questions
        # actually query a leave-entitlement entity (not generic V_EMP profile).
        # Skips for action requests (which use create/update) unless it's leave_request.
        steps = ensure_leave_entitlement_entity(
            steps,
            user_query,
            auth_context,
            cached_schema,
            cached_tools=cached_tools,
            tone_context=tone_context,
            tenant_id=tenant_id,
        )

        # -- Unified Action Framework -- handles entity-specific action injection
        # for entities registered in config/entity_actions.yaml.
        # Currently covers: employee_general (simple_update)
        # Delegates to strategy-specific injectors (simple_update_strategy, leave_strategy, etc.)
        #
        # Returns (steps, pending_slots) where pending_slots surfaces fields that need
        # clarification. The LLM/planner decides whether to ask or skip — not the injector.
        #
        # Warm the CodeResolver cache before action injection so leave-type
        # description matching (e.g. "Annual Leave" -> "ANL") resolves against
        # fresh codesetup rather than a stale/hardcoded fallback.
        try:
            from agent.dab.code_resolver import CodeResolver
            await CodeResolver(tenant_id).get_reverse_index()
        except Exception as _warm_exc:
            logger.debug("build_tool_plan: CodeResolver warm failed (non-fatal): %s", _warm_exc)

        steps, pending_slots = inject_all_actions(
            steps,
            user_query,
            auth_context,
            cached_schema,
            cached_tools=cached_tools,
            tone_context=tone_context,
            tenant_id=tenant_id,
            conversation_history=conversation_history,
        )

    # -- Action-domain clarification gate --
    # When inject_all_actions returned pending_slots but produced no write steps
    # (create/update), the LLM likely misclassified the query as kpi_only or
    # read-only. The planner must ask for clarification rather than silently
    # proceeding with the wrong intent.
    if pending_slots:
        write_tools = {"create_record", "update_record", "delete_record"}
        has_write_step = any(s.get("tool") in write_tools for s in steps)
        if not has_write_step:
            # Build a human-readable reason from the pending slot details
            slot_summaries = [
                f"{p.get('entity', '?')}.{p.get('field', '?')}"
                for p in pending_slots
                if isinstance(p, dict)
            ]
            clarification_reason = (
                f"Action intent detected but required fields missing: {', '.join(slot_summaries)}. "
                f"The query may have been misclassified — clarification needed."
            )
            tone_context["chart_intent_clarification_needed"] = True
            tone_context["chart_intent_clarification_reason"] = clarification_reason
            logger.info(
                "Action clarification: pending_slots=%s (no write steps produced)",
                slot_summaries,
            )

    # Propagate chart intent clarification need into tone_context so the summarizer
    # can generate an appropriate clarification question before any steps execute.
    if chart_intent_result.needs_clarification:
        tone_context["chart_intent_clarification_needed"] = True
        tone_context["chart_intent_clarification_reason"] = chart_intent_result.clarification_reason
        logger.info(
            "Chart intent clarification: %s (confidence=%.2f)",
            chart_intent_result.clarification_reason, chart_intent_result.confidence,
        )

    # Surface pending_slots to tone_context so the LLM sees them in the next turn.
    # pending_slots contain {entity, field, code_type, user_question} for fields
    # that were mentioned but not yet provided. The planner/LLM decides to ask or skip.
    if pending_slots:
        tone_context["pending_slots"] = pending_slots
        logger.info(
            "Planner: pending_slots from action injection: %s",
            [{"entity": p["entity"], "field": p["field"]} for p in pending_slots]
        )

    if steps:
        # Infer metadata from query + steps (deterministic, no extra API call)
        metadata = await infer_metadata(user_query, steps, tone_context, tenant_id)

        # -- Named dashboard injection -- when a named dashboard is matched,
        # dashboard_queries replaces the LLM's generic steps (e.g. hana_list_schemas)
        # with structured data-retrieval steps for each chart.
        dashboard_queries = metadata.get("dashboard_queries")
        if dashboard_queries:
            dashboard_steps = _build_dashboard_steps_from_queries(
                dashboard_queries,
                hana_schema_registry=hana_schema_registry,
            )
            if dashboard_steps:
                logger.info(
                    "Planner: named dashboard matched -- replaced %d LLM steps with %d dashboard query steps",
                    len(steps),
                    len(dashboard_steps),
                )
                steps = dashboard_steps

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
            "chart_intent_clarification_needed": metadata.get("chart_intent_clarification_needed", False) or tone_context.get("chart_intent_clarification_needed", False),
            "chart_intent_clarification_reason": metadata.get("chart_intent_clarification_reason", "") or tone_context.get("chart_intent_clarification_reason", ""),
            "kpi_only": metadata.get("kpi_only", False),
            "multi_chart": metadata.get("multi_chart", False),
            "dashboard_queries": metadata.get("dashboard_queries", []),
            "pending_slots": pending_slots,
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
    return _empty_plan(pending_slots=pending_slots)