"""
agentic_executor.py
Reflexive agent loop — delegates planning to tool_planner.py, 
presentation to summarizer/response_summarizer.py (proven components).
Uses DAB (Data API Builder) 
Implements Inform -> Assist -> Offer feedback via intent_category pipeline.

PATCH 2026-06-19: Chart artifacts are now injected into LLM context as raw facts
(industry standard: generate -> feed into context -> LLM embeds naturally).
Post-hoc appending removed to eliminate [Insert chart here] hallucinations.
Added extensive chart debug logging to trace extraction failures.

PATCH 2026-06-22: Added client-side dynamic binning for multi-tenant schemas.
Supports runtime age group binning, tenure bands, and salary ranges via pandas.

PATCH 2026-06-26: Fixed aggregate column detection in client-side binning
(post_aggregate now uses _is_aggregate_value_col instead of exact "count" match).
Fixed aggregate_records with groupby pagination — strips 'first' to ensure
complete group data for accurate charting.
"""
import json
import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

from agent.llm_client import call_llm
from agent.config import LARGE_RESULT_THRESHOLD
from agent.rag_retriever import retrieve_policy_context
from agent.chart_generator import generate_chart, extract_chartable_data, _is_aggregate_value_col
from agent.tool_planner import build_tool_plan
from agent.summarizer.response_summarizer import summarize_results
from agent.intent_classifier import classify_intent, IntentResult, _is_short_affirmative

from agent.excel_exporter import export_to_excel, export_to_excel_with_chart

# DAB client integration
from agent.dab_client import dab_manager, invoke_dab_tool_with_retry
from agent.dab.dab_response import extract_items, extract_payload, is_dab_error
from agent.dab.validation import validate_dab_args
from agent.dab.metrics import (
    record_dab_tool_call, record_dab_error,
    record_chart_generated
)
from agent.dab.odata_normalizer import normalize_odata_args

# HANA client integration
from agent.hana_client import hana_manager, normalize_hana_result, extract_hana_items

logger = logging.getLogger("hr_agent")

# Lightweight session state for follow-up actions (DST-lite)
_SESSION_STATE: Dict[str, Dict] = {}

# ─── Tool execution registry ────────────────────────────────────────────────
# Maps tool names to executor functions.
# Idiomatic replacement for string-matching in the main loop.
_TOOL_REGISTRY: Dict[str, Any] = {}


def _register_tool(tool_name: str, executor: Any):
    _TOOL_REGISTRY[tool_name] = executor


def _get_executor(tool_name: str):
    if tool_name in _TOOL_REGISTRY:
        return _TOOL_REGISTRY[tool_name]
    if tool_name in ("read_records", "aggregate_records", "describe_entities"):
        return _execute_dab_tool_call
    if tool_name.startswith("hana_"):
        return _execute_hana_tool_call
    return None


def _get_session_key(auth_context: Any) -> str:
    """Derive a session key from auth context."""
    if auth_context and auth_context.email:
        return auth_context.email.lower()
    if auth_context and auth_context.emp_id:
        return str(auth_context.emp_id)
    return "anonymous"


def _load_session_state(auth_context: Any) -> Dict:
    """Load previous turn state for this session."""
    key = _get_session_key(auth_context)
    return _SESSION_STATE.get(key, {})


def _save_session_state(auth_context: Any, state: Dict):
    """Save current turn state for follow-up retrieval."""
    key = _get_session_key(auth_context)
    _SESSION_STATE[key] = state


def _is_data_about_user(items: List[Dict], auth_context: Any) -> bool:
    """Check if ALL returned data items belong to the requesting user."""
    if not auth_context or (not auth_context.email and not auth_context.emp_id):
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        is_self = False
        if auth_context.email:
            item_email = str(item.get("EMAIL", item.get("email", ""))).lower()
            if item_email == auth_context.email.lower():
                is_self = True
        if auth_context.emp_id:
            item_emp_id = str(item.get("EMPLOYEE_NO", item.get("emp_id", "")))
            if item_emp_id == str(auth_context.emp_id):
                is_self = True
        if not is_self:
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# L1: LIGHTWEIGHT INTENT GUARD (industry standard fast path for greetings)
# ═════════════════════════════════════════════════════════════════════════════

_GREETING_PATTERNS = [
    # Pure greeting words + optional "there" or trailing punctuation
    re.compile(r'^\s*(hi|hello|hey|greetings|howdy|hola|bonjour|sup|yo)\b(\s+there)?[\s,!?.…]*$', re.I),
    # Time-of-day greetings + optional addressee
    re.compile(r'^\s*good\s+(morning|afternoon|evening|day)\b(\s+(everyone|all|there|team|folks))?[\s,!?.…]*$', re.I),
    # Thanks + optional extension
    re.compile(r'^\s*(thanks|thank\s+you|thx|ty)\b(\s+(very\s+much|a\s+lot|so\s+much))?[\s,!?.…]*$', re.I),
    # Goodbye
    re.compile(r'^\s*(bye|goodbye|see\s+ya|later)\b[\s,!?.…]*$', re.I),
    # "what's up" and variants
    re.compile(r"^\s*what\'s\s+up\b[\s,!?.…]*$", re.I),
    re.compile(r"^\s*what\s+is\s+up\b[\s,!?.…]*$", re.I),
    # Punctuation only
    re.compile(r'^\s*[!?.…]{1,3}\s*$'),
]

_SMALLTALK_PATTERNS = [
    re.compile(r'^\s*(who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name|tell\s+me\s+about\s+yourself)\b[\s,!?.…]*$', re.I),
    re.compile(r'^\s*(are\s+you\s+(an?\s+)?(ai|bot|assistant|human))\b[\s,!?.…]*$', re.I),
]


def _is_greeting_or_smalltalk(query: str) -> tuple[bool, str]:
    q = query.strip()
    if len(q) <= 3:
        return True, "greeting"
    for p in _GREETING_PATTERNS:
        if p.match(q):
            return True, "greeting"
    for p in _SMALLTALK_PATTERNS:
        if p.match(q):
            return True, "smalltalk"
    return False, "task_or_followup"


def _greeting_response(label: str) -> str:
    if label == "greeting":
        return "Hello! How can I help you with HR-related questions today?"
    return "I'm your HR AI Agent. I can help you look up employee information, leave balances, org hierarchy, and HR policies. What would you like to know?"


# ═════════════════════════════════════════════════════════════════════════════
# CLIENT-SIDE DYNAMIC BINNING (multi-tenant schema support)
# ═════════════════════════════════════════════════════════════════════════════


    field_val = args.get("field")
    if field_val and isinstance(field_val, str) and field_val != "*":
        args["field"] = _normalize_field_name(field_val)

    having_val = args.get("having")
    if having_val and isinstance(having_val, str):
        args["having"] = _normalize_filter_expr(having_val)


def apply_client_side_binning(items: List[Dict], bin_config: Dict) -> List[Dict]:
    """
    Apply dynamic binning to raw data after fetching from DAB.
    Supports: post_aggregate (bin then sum counts) and pre_aggregate (raw records -> bin -> count).
    """
    if not items or not isinstance(items, list):
        logger.warning("CLIENT_SIDE_BINNING: no items to bin")
        return []

    method = bin_config.get("method", "post_aggregate")
    column = bin_config.get("column")
    bins = bin_config.get("bins", [0, 25, 35, 45, 55, 100])
    labels = bin_config.get("labels", ["18-25", "26-35", "36-45", "46-55", "56+"])
    output_column = bin_config.get("output_column", f"{column}_group" if column else "binned_group")

    logger.info("CLIENT_SIDE_BINNING: method=%s column=%s output_column=%s bins=%s labels=%s items=%d",
                method, column, output_column, bins, labels, len(items))

    df = pd.DataFrame(items)
    if df.empty:
        logger.warning("CLIENT_SIDE_BINNING: empty dataframe")
        return []

    if method == "post_aggregate":
        # Data already grouped by raw value: [{"age": 25, "count": 5}, ...]
        if column not in df.columns:
            logger.error("CLIENT_SIDE_BINNING: column '%s' not found in %s", column, list(df.columns))
            return items

        df[column] = pd.to_numeric(df[column], errors="coerce")
        initial_rows = len(df)
        df = df.dropna(subset=[column])
        if len(df) < initial_rows:
            logger.info("CLIENT_SIDE_BINNING: dropped %d non-numeric rows", initial_rows - len(df))

        # Apply bins
        df[output_column] = pd.cut(df[column], bins=bins, labels=labels, right=True, include_lowest=True)

        # PATCH 2026-06-26: Find aggregate column using _is_aggregate_value_col instead of exact "count" match.
        # DAB returns aggregate aliases like "count_age", "total_salary", etc.
        count_cols = [c for c in df.columns if _is_aggregate_value_col(c)]
        count_col = count_cols[0] if count_cols else None
        if count_col:
            result = df.groupby(output_column, observed=False)[count_col].sum().reset_index()
            logger.info("CLIENT_SIDE_BINNING: using aggregate column '%s' for summing", count_col)
        else:
            result = df.groupby(output_column, observed=False).size().reset_index(name="count")
            logger.warning("CLIENT_SIDE_BINNING: no aggregate column found; using .size() (counts rows, not employees)")

        result[output_column] = result[output_column].astype(str)
        binned = result.to_dict("records")
        total_count = sum(r.get(count_col, r.get("count", 0)) for r in binned) if count_col else sum(r.get("count", 0) for r in binned)
        logger.info("CLIENT_SIDE_BINNING: post_aggregate binned %d raw groups into %d bins, total_count=%s",
                    len(items), len(binned), total_count)
        return binned

    elif method == "pre_aggregate":
        # Raw records: [{"emp_id": 1, "date_of_birth": "1990-05-15"}, ...]
        if bin_config.get("calculate_age"):
            dob_col = column
            if dob_col not in df.columns:
                logger.error("CLIENT_SIDE_BINNING: DOB column '%s' not found in %s", dob_col, list(df.columns))
                return items

            today = datetime.today()
            df[dob_col] = pd.to_datetime(df[dob_col], errors="coerce")
            initial_rows = len(df)
            df = df.dropna(subset=[dob_col])
            if len(df) < initial_rows:
                logger.info("CLIENT_SIDE_BINNING: dropped %d invalid DOB rows", initial_rows - len(df))

            df["age"] = df[dob_col].apply(
                lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day))
                if pd.notna(x) else None
            )
            column = "age"

        if column not in df.columns:
            logger.error("CLIENT_SIDE_BINNING: column '%s' not found after preprocessing", column)
            return items

        df[column] = pd.to_numeric(df[column], errors="coerce")
        initial_rows = len(df)
        df = df.dropna(subset=[column])
        if len(df) < initial_rows:
            logger.info("CLIENT_SIDE_BINNING: dropped %d non-numeric rows", initial_rows - len(df))

        df[output_column] = pd.cut(df[column], bins=bins, labels=labels, right=True, include_lowest=True)
        result = df.groupby(output_column, observed=False).size().reset_index(name="count")
        result[output_column] = result[output_column].astype(str)
        binned = result.to_dict("records")
        logger.info("CLIENT_SIDE_BINNING: pre_aggregate binned %d raw records into %d bins",
                    len(items), len(binned))
        return binned

    logger.warning("CLIENT_SIDE_BINNING: unknown method '%s'", method)
    return items


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIC FILENAME DERIVATION
# ═════════════════════════════════════════════════════════════════════════════

def _derive_export_prefix(chart_config: Optional[dict], entity: str = "") -> str:
    """Generate human-readable filename from data context, not user query.

    Derives semantic name from:
    1. chart_config.x_column (e.g., "job_level" -> "job_level")
    2. chart_config.type (e.g., "pie" -> "pie_chart")
    3. Entity name fallback (e.g., "Employees" -> "employee")

    Returns snake_case string suitable for filenames.
    """
    parts = []

    if chart_config and isinstance(chart_config, dict):
        x_col = chart_config.get("x_column", "")
        y_col = chart_config.get("y_column", "")
        chart_type = chart_config.get("type", "")

        # Primary: use x_column (the dimension being analyzed)
        if x_col:
            parts.append(x_col.lower().replace(" ", "_"))
        # Secondary: use entity name
        elif entity:
            parts.append(entity.lower().rstrip("s"))

        # Tertiary: chart type descriptor
        if chart_type and chart_type not in ("bar", "table"):
            parts.append(chart_type.lower())
        elif chart_type == "bar":
            parts.append("distribution")

    elif entity:
        parts.append(entity.lower().rstrip("s"))

    if not parts:
        parts.append("export")

    return "_".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN AGENT ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

async def run_reflexive_agent(
    user_query: str,
    conversation_history: str,
    auth_context: Any,
    cached_tools: List[Dict],
    cached_schema: Dict,
) -> str:
    is_gs, gs_label = _is_greeting_or_smalltalk(user_query)
    if is_gs:
        logger.info("L1 guard triggered: %s", gs_label)
        return _greeting_response(gs_label)

    session_state = _load_session_state(auth_context)

    intent_result = classify_intent(user_query, conversation_history)
    logger.info(
        "Intent classified: intent=%s category=%s urgency=%s emotional=%s empathy=%s confidence=%.2f action_oriented=%s",
        intent_result.intent, intent_result.intent_category, intent_result.urgency_level,
        intent_result.emotional_state, intent_result.needs_empathy, intent_result.confidence,
        intent_result.action_oriented
    )

    # PATCH 2026-07-02: Force wants_export=True for short affirmatives when session has cached export data.
    # This prevents the planner from missing the export intent and the LLM from hallucinating a second link.
    if not intent_result.wants_export and _is_short_affirmative(user_query):
        if session_state.get("last_export_url") or session_state.get("last_chart_data"):
            intent_result = IntentResult(
                intent=intent_result.intent,
                intent_category=intent_result.intent_category,
                data_scope=intent_result.data_scope,
                chart_eligible=intent_result.chart_eligible,
                urgency_level=intent_result.urgency_level,
                emotional_state=intent_result.emotional_state,
                topic_sensitivity=intent_result.topic_sensitivity,
                needs_empathy=intent_result.needs_empathy,
                confidence=intent_result.confidence,
                action_oriented=intent_result.action_oriented,
                wants_export=True,
            )
            logger.info(
                "FOLLOW_UP_EXPORT_FORCED: short affirmative '%s' + session has cached export -> wants_export=True",
                user_query.strip()
            )

    tone_context = {
        "needs_empathy": intent_result.needs_empathy,
        "urgency_level": intent_result.urgency_level,
        "emotional_state": intent_result.emotional_state,
        "topic_sensitivity": intent_result.topic_sensitivity,
        "intent": intent_result.intent,
        "intent_category": intent_result.intent_category,
        "confidence": intent_result.confidence,
        "action_oriented": intent_result.action_oriented,
        "chart_eligible": intent_result.chart_eligible,
        "data_scope": intent_result.data_scope,
        "wants_export": intent_result.wants_export,
    }

    logger.info("Planning with tool_planner for: %s", user_query)
    plan = await build_tool_plan(
        user_query=user_query, conversation_history=conversation_history,
        auth_context=auth_context, cached_tools=cached_tools,
        cached_schema=cached_schema, rag_available=True, tone_context=tone_context,
    )

    if plan.get("direct_answer"):
        # If user wants export and we have cached data,
        # don't return the planner's hallucinated direct_answer. Instead,
        # initialize state from session and proceed to export generation.
        if intent_result.wants_export and not (intent_result.wants_export and session_state.get("last_chart_data")):
            logger.info("Direct answer from planner")
            return plan["direct_answer"]

    if not plan.get("steps"):
        # FOLLOW-UP: if user accepts a prior export offer, don't bail out early
        if intent_result.wants_export and session_state.get("last_chart_data"):
            logger.info("FOLLOW_UP_EXPORT: empty plan but session has cached data; proceeding to export")
            # Initialize minimal state so downstream paths can reuse cached data
            state = {
                "tool_results": session_state.get("last_tool_results", {}),
                "tool_calls_made": [],
                "rag_context": "",
                "export_url": session_state.get("last_export_url", ""),
            }
            chartable_data = session_state.get("last_chart_data", [])
            chart_config = session_state.get("last_chart_config", {})
            # Skip to chart/export logic below — do NOT return early
        else:
            logger.info("No steps — clarification needed")
            return plan.get("reasoning", "Could you clarify what you're looking for?")

    # STAGE 2: Execute planned steps via unified tool registry
    if plan.get("steps"):
        state = {
            "tool_results": {},
            "tool_calls_made": [],
            "rag_context": "",
        }

        tenant_id = auth_context.tenant_id if auth_context and auth_context.tenant_id else "LOCALDEV"

        for step in plan["steps"]:
            tool = step.get("tool", "")
            executor = _get_executor(tool)
            if executor is None:
                logger.warning("No executor registered for tool '%s' — skipping", tool)
                state["tool_calls_made"].append({"tool": tool, "args": step.get("args", {}), "status": "NO_EXECUTOR"})
                continue

            await executor(
                step=step,
                state=state,
                auth_context=auth_context,
                tenant_id=tenant_id,
                cached_schema=cached_schema,
                cached_tools=cached_tools,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # CLIENT-SIDE BINNING — applied after DAB execution, before chart/summary
    # ═══════════════════════════════════════════════════════════════════════
    binned_output_column = None  # Initialize regardless of whether binning is needed
    bin_config = plan.get("client_side_binning")
    if bin_config and isinstance(bin_config, dict):
        logger.info("CLIENT_SIDE_BINNING: plan includes binning config: %s", bin_config)
        # Find the first successful tool result with data to bin
        binned_any = False
        for call_id, result in state["tool_results"].items():
            logger.info("CLIENT_SIDE_BINNING: checking call_id=%s result_type=%s", call_id, type(result).__name__)
            items = extract_items(result)
            logger.info("CLIENT_SIDE_BINNING: extracted %d items from %s", len(items), call_id)
            if not items:
                continue

            # Log first few items for debugging
            logger.info("CLIENT_SIDE_BINNING: sample item keys=%s", list(items[0].keys()) if items else [])

            # DYNAMIC METHOD DETECTION:
            # If data has only the raw column (e.g., [{"age": 25}, ...]) -> use pre_aggregate
            # If data has raw column + count (e.g., [{"age": 25, "count": 5}, ...]) -> use post_aggregate
            column = bin_config.get("column", "")
            has_count = any("count" in k.lower() or "total" in k.lower() or "sum" in k.lower() 
                           for k in (items[0].keys() if items else []))
            has_raw_only = len(items[0].keys() if items else []) == 1 and column in (items[0] if items else {})

            dynamic_config = dict(bin_config)  # copy
            if has_raw_only and not has_count:
                dynamic_config["method"] = "pre_aggregate"
                logger.info("CLIENT_SIDE_BINNING: auto-detected pre_aggregate (raw records without count)")
            else:
                dynamic_config["method"] = "post_aggregate"
                logger.info("CLIENT_SIDE_BINNING: auto-detected post_aggregate (grouped data with count)")

            binned = apply_client_side_binning(items, dynamic_config)
            if binned and len(binned) > 0:
                    # Replace the result with binned data, preserving wrapper if possible
                    if isinstance(result, dict) and "result" in result:
                        result["result"] = binned
                        result["message"] = f"Dynamically binned into {len(binned)} groups"
                    elif isinstance(result, dict) and "value" in result:
                        result["value"] = binned
                        result["message"] = f"Dynamically binned into {len(binned)} groups"
                    else:
                        state["tool_results"][call_id] = {
                            "result": binned,
                            "message": f"Dynamically binned into {len(binned)} groups"
                        }
                    logger.info("CLIENT_SIDE_BINNING: replaced result for %s with %d binned rows", call_id, len(binned))
                    binned_output_column = dynamic_config.get("output_column", f"{dynamic_config['column']}_group" if dynamic_config.get('column') else "binned_group")
                    binned_any = True
                    break
        if not binned_any:
            logger.warning("CLIENT_SIDE_BINNING: no suitable tool result found for binning")

    if plan.get("rag"):
        state["rag_context"] = retrieve_policy_context(plan.get("rag_query", user_query))

    if not state["tool_results"] and not (intent_result.wants_export and session_state.get("last_chart_data")):
        return "I wasn't able to retrieve any data for that request."

    # ═══════════════════════════════════════════════════════════════════════
    # CODE RESOLUTION — scan results for codes and build LLM context
    # This runs BEFORE chart generation so LLM sees the mappings when
    # interpreting raw data values for natural language output
    # ═══════════════════════════════════════════════════════════════════════
    code_context_md = ""
    tenant_id_for_codes = auth_context.tenant_id if auth_context and auth_context.tenant_id else "default"
    try:
        from agent.code_resolver import scan_for_codes
        code_context_md = await scan_for_codes(state["tool_results"], tenant_id_for_codes)
        if code_context_md:
            logger.info("CODE_RESOLVER: injected code context (%d chars) into LLM", len(code_context_md))
    except Exception as e:
        logger.warning("CODE_RESOLVER: failed to scan results: %s", e)

    # ═══════════════════════════════════════════════════════════════════════
    # CHART GENERATION — moved BEFORE summarizer, injected as raw artifact
    # ═══════════════════════════════════════════════════════════════════════
    chart_markdown = ""
    if 'chartable_data' not in locals() or not chartable_data:
        chartable_data = []

    # ═══════════════════════════════════════════════════════════════════════
    # FOLLOW-UP EXPORT: Prefer cached chart_config over planner's default
    # When user says "export" after seeing a chart, preserve the original chart
    # type (bar, pie, etc.) from the previous turn. The planner may generate a
    # different default type on follow-up queries without chart context.
    # ═══════════════════════════════════════════════════════════════════════
    cached_cfg = session_state.get("last_chart_config")
    if intent_result.wants_export and cached_cfg:
        # Follow-up export: always use cached config to preserve chart type
        chart_config = dict(cached_cfg)  # shallow copy to avoid mutating session
        logger.info(
            "FOLLOW_UP_EXPORT: restored chart_config from session (type=%s), overriding planner default",
            chart_config.get("type")
        )
        if not chartable_data and session_state.get("last_chart_data"):
            chartable_data = list(session_state["last_chart_data"])  # copy
            logger.info("FOLLOW_UP_EXPORT: restored %d cached rows", len(chartable_data))
    elif 'chart_config' not in locals() or not chart_config:
        chart_config = plan.get("chart")
    # -----------------------------------------------------------------------

    if chart_config and binned_output_column:
        chart_config["x_column"] = binned_output_column
        chart_config["title"] = f"{binned_output_column.replace('_', ' ').title()} Distribution"
        logger.info("CHART_DEBUG_EXECUTOR: synced chart_config to binned column=%s title=%s", binned_output_column, chart_config["title"])

    if chart_config and state["tool_results"]:
        # Sync y_column to actual aggregate column name using same logic as chart_generator
        for call_id, result in state["tool_results"].items():
            items = extract_items(result)
            if items:
                cols = list(items[0].keys())
                agg_cols = [c for c in cols if _is_aggregate_value_col(c)]
                if agg_cols and chart_config.get("y_column") not in cols:
                    chart_config["y_column"] = agg_cols[0]
                    logger.info("CHART_DEBUG_EXECUTOR: synced y_column to %s", agg_cols[0])
                break

    logger.info("CHART_DEBUG_EXECUTOR: chart_config=%s", chart_config)

    # Default auth_role for both chart and export paths
    auth_role = "employee"

    if chart_config and isinstance(chart_config, dict):
        logger.info("CHART_DEBUG_EXECUTOR: chart_config present, type=%s", chart_config.get("type"))

        for i, tc in enumerate(state["tool_calls_made"]):
            call_id = f"{tc['tool']}_{i}"
            result = state["tool_results"].get(call_id, {})
            logger.info("CHART_DEBUG_EXECUTOR: extracting from call_id=%s result_type=%s", 
                       call_id, type(result).__name__)
            extracted = extract_chartable_data({call_id: {"result": result}})
            logger.info("CHART_DEBUG_EXECUTOR: extracted %d rows from %s", len(extracted), call_id)
            chartable_data.extend(extracted)

        logger.info("CHART_DEBUG_EXECUTOR: total chartable_data=%d rows", len(chartable_data))



        # Determine auth_role once for both chart and export paths
        auth_role = "employee"
        fallback_used = False
        if auth_context:
            # Primary: check permissions list/set
            perms = getattr(auth_context, 'permissions', None)
            if perms:
                if "read:all_employees" in perms:
                    auth_role = "admin"
                elif "read:subordinates" in perms:
                    auth_role = "manager"

            # Fallback 1: check internal_roles (from tenant_mappings.yaml)
            # HRMS_HR -> admin, HRMS_MANAGER -> manager, HRMS_EMPLOYEE -> employee
            roles = getattr(auth_context, 'internal_roles', None)
            if roles and auth_role == "employee":
                fallback_used = True
                role_set = {r.lower() for r in roles} if isinstance(roles, (list, set, tuple)) else {str(roles).lower()}
                if any(r in role_set for r in ('hrms_hr', 'admin', 'hr')):
                    auth_role = "admin"
                elif any(r in role_set for r in ('hrms_manager', 'manager', 'supervisor')):
                    auth_role = "manager"

            # Fallback 2: check legacy 'role' attribute (backward compat)
            if auth_role == "employee" and getattr(auth_context, 'role', None):
                fallback_used = True
                role = auth_context.role.lower()
                if role in ('admin', 'hrms_hr', 'hr'):
                    auth_role = "admin"
                elif role in ('manager', 'supervisor', 'hrms_manager'):
                    auth_role = "manager"

        if fallback_used:
            logger.warning(
                "AUTH_ROLE_FALLBACK: auth_role resolved via fallback — "
                "permissions=%s internal_roles=%s. "
                "Investigate why permissions were not populated by auth layer.",
                sorted(perms) if perms else None,
                sorted(roles) if roles else None
            )

        logger.info("AUTH_ROLE_RESOLVED: auth_role=%s permissions=%s internal_roles=%s",
                    auth_role, 
                    sorted(perms) if perms else None,
                    sorted(roles) if roles else None)

        if chartable_data:
            chart_type = chart_config.get("type", "bar")
            x_col = chart_config.get("x_column")
            y_col = chart_config.get("y_column")
            title = chart_config.get("title", "Chart")

            chart_markdown = generate_chart(
                data=chartable_data, chart_type=chart_type,
                x_column=x_col, y_column=y_col, title=title,
                y_label=chart_config.get("y_label") if chart_config else None,
                auth_role=auth_role, include_table=False
            )
            record_chart_generated(chart_type, "inline")
            logger.info(
                "CHART_DEBUG_EXECUTOR: generate_chart returned len=%d empty=%s starts_mermaid=%s",
                len(chart_markdown), chart_markdown == "", chart_markdown.startswith("```mermaid")
            )
        else:
            logger.warning("CHART_DEBUG_EXECUTOR: no chartable data found in any tool result")
    else:
        logger.info("CHART_DEBUG_EXECUTOR: no chart_config in plan")



    # ═══════════════════════════════════════════════════════════════════════
    # BUILD METADATA for audit & provenance
    # ═══════════════════════════════════════════════════════════════════════
    metadata = {
        "Report": chart_config.get("title", "HR Analytics Report") if chart_config else "HR Analytics Report",
        "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Tenant": getattr(auth_context, "tenant_id", "LOCALDEV") if auth_context else "LOCALDEV",
        "Generated By": getattr(auth_context, "email", "system") if auth_context else "system",
        "User Role": auth_role,
        "Query": user_query,
        "Intent": intent_result.intent if intent_result else "unknown",
        "Data Scope": intent_result.data_scope if intent_result else "unknown",
        "Chart Type": chart_config.get("type", "bar") if chart_config else "none",
        "X Column": chart_config.get("x_column", "") if chart_config else "",
        "Y Column": chart_config.get("y_column", "") if chart_config else "",
        "AI Version": "HRMS Agent 1.2",
        "Source": "DAB 2.0.8 / SQL Server",
    }
    # Add filter info from tool calls if available
    if state.get("tool_calls_made"):
        tc = state["tool_calls_made"][0]
        args = tc.get("args", {})
        if args.get("filter"):
            metadata["Filters Applied"] = str(args["filter"])
        if args.get("entity"):
            metadata["Entity"] = args["entity"]

    # ═══════════════════════════════════════════════════════════════════════
    # EXCEL EXPORT — chart-rich if allowed, data-only as fallback
    # ═══════════════════════════════════════════════════════════════════════
    if (plan.get("needs_export") or intent_result.wants_export) and chart_config and isinstance(chart_config, dict) and chartable_data:
        chart_type = chart_config.get("type", "bar")
        x_col = chart_config.get("x_column")
        y_col = chart_config.get("y_column")
        title = chart_config.get("title")

        tc = state["tool_calls_made"][0] if state["tool_calls_made"] else {}
        prefix = _derive_export_prefix(chart_config, tc.get("args", {}).get("entity", ""))
        export_url = export_to_excel_with_chart(
            data=chartable_data,
            chart_type=chart_type,
            x_column=x_col,
            y_column=y_col,
            title=title,
            prefix=prefix or "export",
            auth_role=auth_role,
            metadata=metadata,
        )
        if export_url:
            state["export_url"] = export_url
            logger.info("EXECUTOR_EXPORT: generated Excel chart export: %s", export_url)
            record_chart_generated(chart_type, "excel")
        else:
            logger.warning(
                "EXECUTOR_EXPORT: chart export blocked or failed (privacy/role/invalid data). "
                "Falling back to data-only export."
            )
            export_url = export_to_excel(chartable_data, prefix=prefix or "export")
            if export_url:
                state["export_url"] = export_url
                logger.info("EXECUTOR_EXPORT: generated data-only Excel export: %s", export_url)

    # ═══════════════════════════════════════════════════════════════════════
    # EXPORT FALLBACK — personal data / full-info requests without charts
    # Industry standard: Excel workbook is canonical deliverable for complete records.
    # Progressive disclosure: show key fields inline; full record in Excel.
    # ═══════════════════════════════════════════════════════════════════════
    if (plan.get("needs_export") or intent_result.wants_export) and not state.get("export_url"):
        is_self_access = (
            auth_context
            and getattr(auth_context, "permissions", None)
            and "read:self" in auth_context.permissions
        )
        for call_id, result in state["tool_results"].items():
            items = extract_items(result)
            if items and isinstance(items, list) and len(items) > 0:
                is_self_access = is_self_access or _is_data_about_user(items, auth_context)
                export_url = export_to_excel(
                    items,
                    prefix="full_profile",
                    strict_privacy=not is_self_access,
                )
                if export_url:
                    state["export_url"] = export_url
                    logger.info(
                        "EXECUTOR_FULLINFO_EXPORT: generated data-only Excel export for %d records: %s",
                        len(items), export_url,
                    )
                break

    # Build summarizer input
    tool_results_for_summarizer = {}
    for i, tc in enumerate(state["tool_calls_made"]):
        call_id = f"{tc['tool']}_{i}"
        result = state["tool_results"].get(call_id, {})
        tool_results_for_summarizer[call_id] = {
            "result": json.dumps(result, default=str),
            "args": tc.get("args", {})
        }

    if state["rag_context"]:
        tool_results_for_summarizer["__rag_context"] = {
            "result": state["rag_context"], "args": {}
        }

    if chart_markdown:
        tool_results_for_summarizer["__chart"] = {
            "result": chart_markdown,
            "args": chart_config or {}
        }
        logger.info("CHART_DEBUG_INJECT: injected __chart len=%d", len(chart_markdown))
    else:
        logger.info("CHART_DEBUG_INJECT: no chart_markdown to inject")

    action_context = plan.get("action_context", "")

    answer = summarize_results(
        user_query=user_query,
        tool_results=tool_results_for_summarizer,
        conversation_history=conversation_history,
        chart_config=chart_config,
        call_llm_fn=call_llm,
        large_result_threshold=LARGE_RESULT_THRESHOLD,
        tone_context=tone_context,
        action_context=action_context,
        export_url=state.get("export_url", ""),
        wants_export=plan.get("needs_export", False) or intent_result.wants_export,
        code_context=code_context_md,
    )

    # Save session state for follow-up turns
    _save_session_state(auth_context, {
        "last_query": user_query,
        "last_intent": intent_result.intent,
        "last_intent_category": intent_result.intent_category,
        "last_data_scope": intent_result.data_scope,
        "last_chart_data": chartable_data if chartable_data else session_state.get("last_chart_data"),
        "last_chart_config": chart_config if chart_config else session_state.get("last_chart_config"),
        "last_tool_results": state.get("tool_results") or session_state.get("last_tool_results", {}),
        "last_export_url": state.get("export_url", "") or session_state.get("last_export_url", ""),
    })

    return answer


# ═════════════════════════════════════════════════════════════════════════════
# DAB TOOL EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

async def _execute_dab_tool_call(step: Dict, state: Dict, auth_context: Any, tenant_id: str = "LOCALDEV", cached_schema: Dict = None, cached_tools: List[Dict] = None):
    from agent.main import enforce_tool_args, filter_tool_results

    tool = step.get("tool")
    args = step.get("args", {})
    call_id = f"{tool}_{len(state['tool_calls_made'])}"
    client = dab_manager.get_client(tenant_id)

    # Normalize entity name, field names, and numeric types using consolidated normalizer
    normalize_odata_args(args, cached_schema)

    logger.debug("Executing DAB tool: %s with args: %s", tool, args)

    args, error = enforce_tool_args(tool, args, auth_context)
    if error:
        state["tool_results"][call_id] = {"error": error}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"BLOCKED: {error}"})
        record_dab_tool_call(tool, "BLOCKED")
        record_dab_error("AuthError", entity=args.get("entity"))
        return

    # Validate tool arguments against cached JSON schemas
    args, schema_error = validate_dab_args(tool, args, cached_tools)
    if schema_error:
        state["tool_results"][call_id] = {"error": schema_error}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"BLOCKED: {schema_error}"})
        record_dab_tool_call(tool, "BLOCKED")
        record_dab_error("ValidationError", entity=args.get("entity"))
        return

    if tool == "read_records":
        select_val = args.get("select")
        if select_val is not None:
            if isinstance(select_val, str):
                parsed = [f.strip() for f in select_val.split(",") if f.strip()]
                if not parsed or parsed == ["*"]:
                    args.pop("select", None)
                else:
                    args["select"] = ",".join(parsed)
            elif isinstance(select_val, list):
                cleaned = [str(f).strip() for f in select_val if str(f).strip()]
                if cleaned:
                    args["select"] = ",".join(cleaned)
                else:
                    args.pop("select", None)
            else:
                args.pop("select", None)

        orderby_val = args.get("orderby")
        if orderby_val is not None:
            if isinstance(orderby_val, str):
                parsed = [o.strip() for o in orderby_val.split(",") if o.strip()]
                if parsed:
                    args["orderby"] = parsed
                else:
                    args.pop("orderby", None)
            elif isinstance(orderby_val, list):
                cleaned = [str(o).strip() for o in orderby_val if str(o).strip()]
                if cleaned:
                    args["orderby"] = cleaned
                else:
                    args.pop("orderby", None)
            else:
                args.pop("orderby", None)

    elif tool == "aggregate_records":
        groupby_val = args.get("groupby")
        if groupby_val is not None:
            if isinstance(groupby_val, str):
                parsed = [g.strip() for g in groupby_val.split(",") if g.strip()]
                if parsed:
                    args["groupby"] = parsed
                else:
                    args.pop("groupby", None)
            elif isinstance(groupby_val, list):
                cleaned = [str(g).strip() for g in groupby_val if str(g).strip()]
                if cleaned:
                    args["groupby"] = cleaned
                else:
                    args.pop("groupby", None)
            else:
                args.pop("groupby", None)

        orderby_val = args.get("orderby")
        if orderby_val is not None:
            if isinstance(orderby_val, str):
                parsed = [o.strip() for o in orderby_val.split(",") if o.strip()]
                if parsed:
                    args["orderby"] = parsed
                else:
                    args.pop("orderby", None)
            elif isinstance(orderby_val, list):
                cleaned = [str(o).strip() for o in orderby_val if str(o).strip()]
                if cleaned:
                    args["orderby"] = cleaned
                else:
                    args.pop("orderby", None)
            else:
                args.pop("orderby", None)

        distinct_val = args.get("distinct")
        if distinct_val is not None:
            if isinstance(distinct_val, bool):
                pass
            elif isinstance(distinct_val, str):
                lowered = distinct_val.strip().lower()
                if lowered in ("true", "1", "yes"):
                    args["distinct"] = True
                elif lowered in ("false", "0", "no", ""):
                    args["distinct"] = False
                else:
                    args.pop("distinct", None)
            else:
                args.pop("distinct", None)

        # For aggregate_records with groupby, remove 'first' to ensure
        # all groups are returned. Charts need complete data; CHART_MAX_CATEGORIES handles display limits.
        if args.get("groupby"):
            if "first" in args:
                logger.info("Stripping 'first'=%s from aggregate_records with groupby for complete chart data", args.get("first"))
                args.pop("first", None)
        else:
            # Without groupby, aggregate_records returns a single scalar; 'first' is meaningless
            if "first" in args:
                logger.warning("Stripping 'first'=%s from aggregate_records without groupby", args.get("first"))
                args.pop("first", None)

        func_val = args.get("function", "").lower()
        if func_val == "count" and not args.get("field"):
            args["field"] = "*"

        field_val = args.get("field")
        if field_val == "*" and func_val != "count":
            args.pop("field", None)

    try:
        result = await invoke_dab_tool_with_retry(client, tool, args)
        dab_data = extract_payload(result)

        # Auto-retry without select on "Invalid field" error
        if isinstance(dab_data, dict) and dab_data.get("isError"):
            error_msg = dab_data.get("message", "Unknown DAB error")
            if "Invalid field" in error_msg and args.get("select"):
                logger.warning("Retrying %s without select due to: %s", tool, error_msg)
                args.pop("select", None)
                result = await invoke_dab_tool_with_retry(client, tool, args)
                dab_data = extract_payload(result)
                if not (isinstance(dab_data, dict) and dab_data.get("isError")):
                    logger.info("Retry succeeded without select")
                else:
                    error_msg = dab_data.get("message", "Unknown DAB error")

            if isinstance(dab_data, dict) and dab_data.get("isError"):
                logger.error("DAB tool %s returned error: %s", tool, error_msg)
                state["tool_results"][call_id] = {"error": error_msg}
                state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {error_msg}"})
                record_dab_tool_call(tool, "ERROR")
                record_dab_error("DabError", entity=args.get("entity"))
                return

        if isinstance(dab_data, dict):
            result_text = json.dumps(dab_data, default=str)
        else:
            result_text = str(dab_data)
        filtered = filter_tool_results(tool, result_text, auth_context)

        try:
            parsed = json.loads(filtered) if isinstance(filtered, str) else filtered
        except:
            parsed = {"raw_text": filtered}

        state["tool_results"][call_id] = parsed
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": "SUCCESS"})
        logger.info("DAB tool %s succeeded (call_id=%s)", tool, call_id)
        record_dab_tool_call(tool, "SUCCESS")

    except Exception as e:
        logger.error("DAB tool %s failed: %s", tool, e)
        state["tool_results"][call_id] = {"error": str(e)}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {e}"})
        record_dab_tool_call(tool, "ERROR")
        record_dab_error("SystemError", entity=args.get("entity"))


# ═════════════════════════════════════════════════════════════════════════════
# HANA TOOL EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

async def _execute_hana_tool_call(step: Dict, state: Dict, auth_context: Any, tenant_id: str = "LOCALDEV", cached_schema: Dict = None, cached_tools: List[Dict] = None):
    """
    Execute a HANA MCP tool and normalize the result into DAB-style format.

    Pipeline:
      1. Get HANA client from hana_manager (registry pattern)
      2. Call tool (sync via HTTP or STDIO)
      3. Normalize HANA columns/rows -> list[dict] via normalize_hana_result()
      4. Store in state["tool_results"] in the same shape as DAB results
    """
    tool = step.get("tool", "")
    args = step.get("args", {})
    call_id = f"{tool}_{len(state['tool_calls_made'])}"

    client = hana_manager.get_client(tenant_id)

    try:
        raw = client.call_tool(tool, args)
        normalized = normalize_hana_result(raw)

        # If normalization produced a result list, store it directly
        if isinstance(normalized, dict) and "result" in normalized:
            items = normalized.get("result", [])
            if items:
                state["tool_results"][call_id] = normalized
                state["tool_calls_made"].append({"tool": tool, "args": args, "status": "SUCCESS"})
                logger.info("HANA tool %s succeeded (call_id=%s, rows=%d)", tool, call_id, len(items))
            else:
                state["tool_results"][call_id] = {"error": "HANA query returned no data", "rows": 0}
                state["tool_calls_made"].append({"tool": tool, "args": args, "status": "EMPTY"})
                logger.warning("HANA tool %s returned empty result (call_id=%s)", tool, call_id)
        elif isinstance(normalized, dict) and normalized.get("isError"):
            error_msg = normalized.get("message", "Unknown HANA error")
            state["tool_results"][call_id] = {"error": error_msg}
            state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {error_msg}"})
            logger.error("HANA tool %s error: %s", tool, error_msg)
        else:
            state["tool_results"][call_id] = normalized
            state["tool_calls_made"].append({"tool": tool, "args": args, "status": "SUCCESS"})
            logger.info("HANA tool %s succeeded (call_id=%s)", tool, call_id)

    except Exception as e:
        logger.error("HANA tool %s failed: %s", tool, e)
        state["tool_results"][call_id] = {"error": str(e)}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {e}"})