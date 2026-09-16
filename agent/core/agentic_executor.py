"""
agentic_executor.py
Reflexive agent loop -- delegates planning to tool_planner.py,
presentation to summarizer/response_summarizer.py (proven components).
Uses DAB (Data API Builder)
Implements Inform -> Assist -> Offer feedback via intent_category pipeline.

Patches:
  - Chart artifacts injected into LLM context as raw facts (not appended).
  - Client-side dynamic binning for multi-tenant schemas (age, tenure, salary).
  - Fixed aggregate column detection in post_aggregate; stripped 'first' for groupby complete data.
"""
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from agent.core.guards import (
    is_greeting_or_smalltalk,
    greeting_response,
    is_vague_data_question,
    data_summary_response,
)
from agent.core.session_state import load_session_state, save_session_state
from agent.core.progress import emit_progress, emit_chart
from agent.output.chart_payload import build_chartjs_payload
from agent.auth.role_resolver import resolve_auth_role, is_data_about_user
from agent.auth.tool_guards import enforce_tool_args, filter_tool_results
from agent.output.binning_orchestrator import apply_client_side_binning
from agent.output.export_service import (
    derive_export_prefix,
    build_export_metadata,
    generate_chart_export,
    generate_data_export,
)
from agent.integrations.llm_client import call_llm
from agent.config import LARGE_RESULT_THRESHOLD
from agent.integrations.rag_retriever import retrieve_policy_context
from agent.output.chart_generator import generate_chart, extract_chartable_data, _is_aggregate_value_col, _detect_wide_format_metrics
from agent.core.tool_planner import build_tool_plan
from agent.summarizer.response_summarizer import summarize_results
from agent.core.intent_classifier import classify_intent, IntentResult, _is_short_affirmative
from agent.output.excel_exporter import export_to_excel
from agent.integrations.dab_client import dab_manager, invoke_dab_tool_with_retry
from agent.data.response import extract_items, extract_payload
from agent.dab.validation import validate_dab_args
from agent.data.metrics import (
    record_dab_tool_call, record_dab_error,
)
from agent.dab.odata_normalizer import normalize_odata_args
from agent.integrations.hana_client import (
    hana_manager,
    normalize_hana_result,
    validate_hana_tool_args,
    _looks_like_numeric_type_error,
    _wrap_aggregate_fields_with_cast,
    is_hana_available,
)
from agent.integrations.schema_registry import schema_registry_service

logger = logging.getLogger("hr_agent")

# Forecasting imports (guarded to avoid circular deps at module load time)
def _get_forecasting_modules():
    from agent.forecasting.external_data_fetcher import ExternalDataFetcher
    from agent.forecasting.code_generator import CodeGenerator
    from agent.core.sandbox import SubprocessSandbox
    from agent.forecasting.output_parser import parse_and_validate
    return ExternalDataFetcher, CodeGenerator, SubprocessSandbox, parse_and_validate


def _is_aggregate_query_despite_discovery_classification(query: str) -> bool:
    """Heuristic guard: detect aggregate/data-retrieval queries misclassified as data_discovery.

    Returns True when the query clearly asks for data values/aggregates rather than
    asking what data/systems/tables are available.
    """
    q = query.lower()
    aggregate_indicators = [
        "how many", "count", "total ", "sum of", "average ", "avg ",
        "how much", "number of", "how many records", "how many rows",
    ]
    discovery_indicators = [
        "what data", "what's available", "what do you have", "show me what",
        "what tables", "what schemas", "what can you", "list all", "show all available",
        "available data", "accessible data",
    ]

    has_aggregate = any(ind in q for ind in aggregate_indicators)
    has_discovery = any(ind in q for ind in discovery_indicators)
    return has_aggregate and not has_discovery


def _get_executor(tool_name: str):
    """Map a tool name to its executor function.

    Idiomatic replacement for string-matching in the main step loop.
    """
    if tool_name in ("read_records", "aggregate_records", "describe_entities"):
        return _execute_dab_tool_call
    if tool_name == "create_record":
        return _execute_dab_create_call
    if tool_name == "update_record":
        return _execute_dab_update_call
    if tool_name.startswith("hana_"):
        return _execute_hana_tool_call
    return None


def _resolve_refs(obj: Any, ref_map: Dict[str, Any]) -> Any:
    """Recursively resolve ``{"$ref": "step_id.result.path"}`` markers in args.

    ``ref_map`` maps step ids to their normalized results:
      - create_record -> the created record dict
      - read_records  -> the array of records (from ``result`` key)
      - other         -> the full result dict

    Path segments after ``result`` are resolved as dict keys or list indices.
    Unresolvable refs are left unchanged and logged.
    """
    if isinstance(obj, dict):
        if "$ref" in obj and len(obj) == 1:
            ref = obj["$ref"]
            try:
                resolved = _resolve_ref_path(ref, ref_map)
                return resolved
            except Exception as exc:
                logger.warning("STEP_REF: failed to resolve $ref=%s: %s", ref, exc)
                return obj
        return {k: _resolve_refs(v, ref_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_refs(item, ref_map) for item in obj]
    return obj


def _resolve_ref_path(ref: str, ref_map: Dict[str, Any]) -> Any:
    """Resolve a single ``$ref`` path like ``leave_hd.result.id``."""
    parts = ref.split(".")
    if len(parts) < 3 or parts[1] != "result":
        raise ValueError(f"invalid $ref format: {ref}")

    step_id = parts[0]
    value = ref_map.get(step_id)
    if value is None:
        raise ValueError(f"no result stored for step id: {step_id}")

    # The stored value may be in several shapes depending on whether
    # _store_step_ref normalized it:
    #   - Normalized create_record: {"id": 42, "status": "P"}  (no "result" key)
    #   - Raw create_record:        {"result": "<json string>", "args": ...}
    #   - Nested create_record:     {"result": {"result": {"id": 42}}, "args": ...}
    #   - Normalized read_records:  [{"id": 7, ...}]  (a list)
    #   - Raw read_records:         {"result": [{"id": 7, ...}]}
    # Unwrap the outer "result" key (if present), parse JSON strings, and
    # unwrap one level of nested {"result": {...}} so the path walk below
    # operates on the actual record dict or list.
    if isinstance(value, dict) and "result" in value:
        inner = value["result"]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except (json.JSONDecodeError, TypeError):
                pass
        if isinstance(inner, dict) and isinstance(inner.get("result"), dict):
            inner = inner["result"]
        # DAB MCP create_record wraps the record in result.value[0]
        # (OData-style array). Unwrap to the actual record dict.
        if isinstance(inner, dict) and isinstance(inner.get("value"), list) and inner["value"]:
            first = inner["value"][0]
            if isinstance(first, dict):
                inner = first
        if isinstance(inner, (dict, list)):
            value = inner

    # Walk the path starting after ``step_id.result``
    for segment in parts[2:]:
        if isinstance(value, dict):
            value = value.get(segment)
        elif isinstance(value, list):
            try:
                idx = int(segment)
                value = value[idx]
            except (ValueError, IndexError):
                raise ValueError(f"cannot index list with {segment!r} in $ref {ref}")
        else:
            raise ValueError(f"cannot traverse {type(value).__name__} with {segment!r} in $ref {ref}")
        if value is None:
            break

    logger.debug("STEP_REF: resolved %s -> %r (type=%s)", ref, value, type(value).__name__ if value is not None else "None")
    return value


def _store_step_ref(state: Dict, step: Dict, result: Dict) -> None:
    """Normalize and store a step result for later $ref resolution."""
    step_id = step.get("_step_id")
    if not step_id:
        return

    tool = step.get("tool", "")
    normalized = result
    if tool == "read_records" and isinstance(result, dict):
        # read_records returns {"result": [...], ...}; store the array for easy indexing.
        normalized = result.get("result", result)
    elif tool == "create_record" and isinstance(result, dict):
        # _execute_dab_create_call stores {"result": <json_string>, "args": ...}
        # where the "result" value is a JSON-serialized string of the created
        # record. Parse it so $ref resolution (e.g. leave_hd.result.id) can
        # traverse into the record dict instead of failing on a string.
        inner = result.get("result", result)
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except (json.JSONDecodeError, TypeError):
                inner = result
        # DAB MCP create_record may nest the record under a "result" key;
        # unwrap one level of nesting so the raw record dict is stored.
        if isinstance(inner, dict) and isinstance(inner.get("result"), dict):
            inner = inner["result"]
        # DAB MCP create_record wraps the record in result.value[0]
        # (OData-style array). Unwrap to the actual record dict.
        if isinstance(inner, dict) and isinstance(inner.get("value"), list) and inner["value"]:
            first = inner["value"][0]
            if isinstance(first, dict):
                inner = first
        normalized = inner if isinstance(inner, dict) else result

    state.setdefault("_step_refs", {})[step_id] = normalized
    logger.debug("STEP_REF: stored result for %s (tool=%s) keys=%s", step_id, tool,
                 list(normalized.keys()) if isinstance(normalized, dict) else type(normalized).__name__)


def _calculate_variances(data: List[Dict], metric_cols: List[str], x_col: str) -> List[Dict]:
    """Calculate period-over-period variances for financial metrics.

    Handles edge cases:
    - Division by zero when prior period value is 0
    - Missing periods (gaps in time series)
    - NaN/None values
    - Negative values (valid for financial data)

    Returns list of variance dicts with:
      - period: the time period
      - metric: metric name
      - current: current period value
      - prior: prior period value (None if first period)
      - abs_change: absolute change
      - pct_change: percentage change (None if prior is 0 or None)
      - is_material: bool indicating if change is material (>10% and abs > threshold)
    """
    if not data or not metric_cols:
        return []

    variances = []
    for i, row in enumerate(data):
        current_period = row.get(x_col)
        if current_period is None:
            continue

        for metric in metric_cols:
            current_val = row.get(metric)
            if current_val is None:
                continue

            try:
                current_num = float(current_val)
            except (TypeError, ValueError):
                continue

            prior_val = data[i - 1].get(metric) if i > 0 else None
            prior_period = data[i - 1].get(x_col) if i > 0 else None

            prior_num = None
            if prior_val is not None:
                try:
                    prior_num = float(prior_val)
                except (TypeError, ValueError):
                    prior_num = None

            abs_change = None
            pct_change = None
            is_material = False

            if prior_num is not None:
                abs_change = current_num - prior_num
                if prior_num != 0:
                    pct_change = (abs_change / abs(prior_num)) * 100
                    # Material variance: >10% change and absolute change > 1000
                    if abs(pct_change) > 10 and abs(abs_change) > 1000:
                        is_material = True

            variances.append({
                "period": current_period,
                "metric": metric,
                "current": current_num,
                "prior": prior_num,
                "prior_period": prior_period,
                "abs_change": abs_change,
                "pct_change": pct_change,
                "is_material": is_material,
            })

    return variances


def _format_variances_for_prompt(variances: List[Dict], max_items: int = 20) -> str:
    """Format variance data as human-readable text for LLM prompt."""
    if not variances:
        return ""

    lines = ["KEY VARIANCES (period-over-period):"]
    material = [v for v in variances if v.get("is_material")]
    if material:
        lines.append("Material changes (>10% or >1000):")
        for v in material[:max_items]:
            metric = v["metric"].replace("_", " ").title()
            direction = "increase" if v.get("abs_change", 0) > 0 else "decrease"
            pct = f"{v['pct_change']:.1f}%" if v.get("pct_change") is not None else "N/A"
            lines.append(
                f"- {metric}: {direction} of {abs(v.get('abs_change', 0)):,.0f} ({pct}) "
                f"in period {v['period']}"
            )
    else:
        lines.append("No material variances detected.")

    # Add summary statistics per metric
    metrics_seen = {}
    for v in variances:
        m = v["metric"]
        if m not in metrics_seen:
            metrics_seen[m] = []
        if v.get("abs_change") is not None:
            metrics_seen[m].append(v["abs_change"])

    if metrics_seen:
        lines.append("")
        lines.append("Overall trends:")
        for metric, changes in metrics_seen.items():
            if not changes:
                continue
            total_change = sum(changes)
            direction = "upward" if total_change > 0 else "downward" if total_change < 0 else "flat"
            metric_label = metric.replace("_", " ").title()
            lines.append(f"- {metric_label}: overall {direction} trend ({total_change:+,.0f})")

    return "\n".join(lines)


def _step_label(tool: str, entity: str) -> str:
    """Human-readable label for a plan step, used in streaming progress events."""
    if tool == "read_records":
        return f"Reading {entity} records..." if entity else "Reading records..."
    if tool == "aggregate_records":
        return f"Aggregating {entity}..." if entity else "Aggregating data..."
    if tool.startswith("hana_"):
        return f"Querying {entity}..." if entity else "Querying finance data..."
    if tool == "create_record":
        return f"Creating {entity}..." if entity else "Creating record..."
    if tool == "update_record":
        return f"Updating {entity}..." if entity else "Updating record..."
    if tool == "delete_record":
        return f"Deleting {entity}..." if entity else "Deleting record..."
    return f"Executing {tool}..."


async def _run_step_loop(
    steps: List[Dict],
    state: Dict,
    auth_context: Any,
    tenant_id: str,
    cached_schema: Dict,
    cached_tools: List[Dict],
    hana_available: bool = True,
) -> None:
    """Execute plan steps sequentially with $ref resolution and stop conditions.

    Shared by the main agent loop and the leave-confirmation re-execution
    turn. Stops when a step is blocked, an executor is missing, or a
    confirmation checkpoint is hit.
    """
    for step in steps:
        tool = step.get("tool", "")
        # Emit per-step progress for the streaming status line.
        entity = step.get("args", {}).get("entity", "") if isinstance(step.get("args"), dict) else ""
        detail = f"{tool}" + (f":{entity}" if entity else "")
        emit_progress("executing", _step_label(tool, entity), detail=detail)
        # Defensive guard: never attempt a HANA tool when HANA is known down.
        if tool.startswith("hana_") and not hana_available:
            logger.warning("Skipping HANA tool '%s' -- HANA is unavailable", tool)
            state["tool_calls_made"].append({"tool": tool, "args": step.get("args", {}), "status": "HANA_UNAVAILABLE"})
            continue
        executor = _get_executor(tool)
        if executor is None:
            logger.warning("No executor registered for tool '%s' -- skipping", tool)
            state["tool_calls_made"].append({"tool": tool, "args": step.get("args", {}), "status": "NO_EXECUTOR"})
            continue

        # Resolve any $ref markers in step args against prior step results.
        resolved_step = dict(step)
        raw_args = step.get("args", {})
        if raw_args and isinstance(raw_args, dict):
            try:
                resolved_args = _resolve_refs(raw_args, state.get("_step_refs", {}))
                if resolved_args is not raw_args:
                    resolved_step = dict(resolved_step)
                    resolved_step["args"] = resolved_args
            except Exception as exc:
                logger.warning("STEP_REF: failed to resolve refs for %s: %s", tool, exc)

        call_id = f"{tool}_{len(state['tool_calls_made'])}"
        await executor(
            step=resolved_step,
            state=state,
            auth_context=auth_context,
            tenant_id=tenant_id,
            cached_schema=cached_schema,
            cached_tools=cached_tools,
        )

        # Store the step result for future $ref resolution.
        result = state["tool_results"].get(call_id)
        if result is not None:
            _store_step_ref(state, resolved_step, result)

        # If a confirmation checkpoint was hit (leave write intercepted),
        # stop processing further steps — the user must confirm first.
        if state.get("_awaiting_confirmation"):
            logger.info("Step loop: awaiting user confirmation, stopping step execution")
            break

        # Stop if any step was blocked (verification, balance, leave header
        # guard, etc.) — subsequent steps in the chain cannot proceed.
        if state["tool_calls_made"] and "BLOCKED" in state["tool_calls_made"][-1].get("status", ""):
            logger.info("Step loop: step %s blocked, stopping step execution", call_id)
            break


async def _execute_pending_leave_confirmation(
    pending: Dict,
    auth_context: Any,
    cached_tools: List[Dict],
    cached_schema: Dict,
    conversation_history: str,
    user_query: str,
) -> str:
    """Re-execute the stored leave create steps after user confirmation.

    Re-runs the full step chain (employee_leave_hd → employee_leave with
    $ref chaining) through the same executor path so that the V_EMP
    verification gate, balance gate, and idempotency guard all apply.
    """
    from agent.core.intent_classifier import classify_intent
    from agent.core.tool_planner import build_tool_plan
    from agent.summarizer.response_summarizer import summarize_results
    from agent.integrations.llm_client import call_llm
    from agent.config import LARGE_RESULT_THRESHOLD

    tenant_id = auth_context.tenant_id if auth_context and auth_context.tenant_id else "LOCALDEV"
    steps = pending.get("steps", [])
    summary = pending.get("summary", "")

    state = {
        "tool_results": {},
        "tool_calls_made": [],
        "rag_context": "",
        # The user already confirmed this request — the confirmation gate
        # must not re-intercept the re-executed writes.
        "_confirmed_leave_write": True,
    }

    await _run_step_loop(
        steps=steps,
        state=state,
        auth_context=auth_context,
        tenant_id=tenant_id,
        cached_schema=cached_schema,
        cached_tools=cached_tools,
    )

    # Build summarizer input
    tool_results_for_summarizer = {}
    for i, tc in enumerate(state["tool_calls_made"]):
        call_id = f"{tc['tool']}_{i}"
        result = state["tool_results"].get(call_id, {})
        tool_results_for_summarizer[call_id] = {
            "result": json.dumps(result, default=str),
            "args": tc.get("args", {}),
        }

    intent_result = await classify_intent(user_query, conversation_history)
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
        "is_finance_query": intent_result.finance_query,
    }

    answer = summarize_results(
        user_query=user_query,
        tool_results=tool_results_for_summarizer,
        conversation_history=conversation_history,
        chart_config=None,
        call_llm_fn=call_llm,
        large_result_threshold=LARGE_RESULT_THRESHOLD,
        tone_context=tone_context,
        action_context="leave_application_confirmed",
        export_url="",
        wants_export=False,
        code_context="",
    )
    return answer


async def _run_forecasting_pipeline(
    user_query: str,
    conversation_history: str,
    auth_context: Any,
    tenant_id: str,
    plan: Dict,
    intent_result: Any,
    session_state: Dict,
    tone_context: Dict,
) -> str:
    """Execute the end-to-end forecasting pipeline (fetch → code → sandbox → parse → summarize)."""
    ExternalDataFetcher, CodeGenerator, SubprocessSandbox, parse_and_validate = _get_forecasting_modules()

    # Prepare state for forecasting
    forecast_state = {
        "tool_results": {},
        "tool_calls_made": [],
        "rag_context": "",
    }

    # Step 1: fetch_external_data
    enriched = {}
    try:
        client = dab_manager.get_client(tenant_id, getattr(auth_context, "token", None), _resolve_dab_role(auth_context))

        async def _dab_query(tool: str, args: Dict[str, Any]) -> Any:
            return await invoke_dab_tool_with_retry(client, tool, args)

        fetcher = ExternalDataFetcher(dab_query_fn=_dab_query)
        enriched = await fetcher.build_enriched_dataset(tenant_id=tenant_id)
        input_path = fetcher.save_input_json(enriched)
        forecast_state["tool_results"]["fetch_external_data"] = {
            "result": {
                "status": "success",
                "input_path": input_path,
                "hr_rows": len(enriched.get("hr_series", [])),
                "market_rows": len(enriched.get("market_series", [])),
                "merged_rows": len(enriched.get("merged_series", [])),
                "data_sources": enriched.get("metadata", {}).get("data_sources", []),
            }
        }
        forecast_state["tool_calls_made"].append({"tool": "fetch_external_data", "args": {"tenant_id": tenant_id}, "status": "SUCCESS"})
        logger.info("FORECASTING: external data fetched, input saved to %s", input_path)
    except Exception as exc:
        logger.error("FORECASTING: external data fetch failed: %s", exc)
        forecast_state["tool_results"]["fetch_external_data"] = {"error": str(exc)}
        forecast_state["tool_calls_made"].append({"tool": "fetch_external_data", "args": {"tenant_id": tenant_id}, "status": f"ERROR: {exc}"})

    # Step 2: generate_code
    code_result = None
    if enriched:
        try:
            generator = CodeGenerator()
            code_result = generator.generate(
                question=user_query,
                enriched_data=enriched,
                model_type="statsforecast",
            )
            if code_result.success:
                forecast_state["tool_results"]["generate_code"] = {
                    "result": {"status": "success", "model_type": code_result.model_type, "cached": code_result.cached}
                }
                forecast_state["tool_calls_made"].append({"tool": "generate_code", "args": {"model_type": "prophet"}, "status": "SUCCESS"})
                logger.info("FORECASTING: code generated (%d chars)", len(code_result.code))
            else:
                forecast_state["tool_results"]["generate_code"] = {"error": code_result.error}
                forecast_state["tool_calls_made"].append({"tool": "generate_code", "args": {"model_type": "prophet"}, "status": f"ERROR: {code_result.error}"})
        except Exception as exc:
            logger.error("FORECASTING: code generation failed: %s", exc)
            forecast_state["tool_results"]["generate_code"] = {"error": str(exc)}

    # Step 3: execute_sandbox
    sandbox_result = None
    if code_result and code_result.success:
        try:
            sandbox = SubprocessSandbox()
            sandbox_result = sandbox.execute(code_result.code, enriched)
            if sandbox_result.success:
                forecast_state["tool_results"]["execute_sandbox"] = {
                    "result": {
                        "status": "success",
                        "output_path": sandbox_result.output_path,
                        "execution_time_seconds": sandbox_result.execution_time_seconds,
                        "data_sources_used": sandbox_result.data_sources_used,
                    }
                }
                forecast_state["tool_calls_made"].append({"tool": "execute_sandbox", "args": {}, "status": "SUCCESS"})
                logger.info("FORECASTING: sandbox executed in %.2fs", sandbox_result.execution_time_seconds)
            else:
                forecast_state["tool_results"]["execute_sandbox"] = {"error": sandbox_result.error}
                forecast_state["tool_calls_made"].append({"tool": "execute_sandbox", "args": {}, "status": f"ERROR: {sandbox_result.error}"})
                # Retry with fix
                if code_result:
                    logger.info("FORECASTING: retrying code generation with error feedback")
                    fix_result = generator.fix(code_result.code, sandbox_result.error or "Execution failed", enriched)
                    if fix_result.success:
                        sandbox_result = sandbox.execute(fix_result.code, enriched)
                        if sandbox_result.success:
                            forecast_state["tool_results"]["execute_sandbox"] = {
                                "result": {
                                    "status": "success_after_fix",
                                    "output_path": sandbox_result.output_path,
                                    "execution_time_seconds": sandbox_result.execution_time_seconds,
                                    "data_sources_used": sandbox_result.data_sources_used,
                                }
                            }
                            forecast_state["tool_calls_made"][-1]["status"] = "SUCCESS_AFTER_FIX"
                            logger.info("FORECASTING: sandbox succeeded after fix")
        except Exception as exc:
            logger.error("FORECASTING: sandbox execution failed: %s", exc)
            forecast_state["tool_results"]["execute_sandbox"] = {"error": str(exc)}

    # Step 4: validate output + build forecast context
    forecast_output = None
    if sandbox_result and sandbox_result.success and sandbox_result.output_path:
        try:
            with open(sandbox_result.output_path, "r", encoding="utf-8") as f:
                raw_output = json.load(f)
            forecast_output = parse_and_validate(raw_output)
            forecast_state["tool_results"]["forecast_output"] = {
                "result": forecast_output.raw
            }
            forecast_state["tool_calls_made"].append({"tool": "parse_output", "args": {}, "status": "SUCCESS"})
        except Exception as exc:
            logger.error("FORECASTING: output parsing failed: %s", exc)
            forecast_state["tool_results"]["forecast_output"] = {"error": str(exc)}

    # Build market context for summarizer
    market_context_parts = []
    if enriched.get("metadata"):
        meta = enriched["metadata"]
        market_context_parts.append(f"External data sources used: {', '.join(meta.get('external_sources', []))}")
        market_context_parts.append(f"Internal data sources: {', '.join(meta.get('internal_sources', []))}")
    if forecast_output and forecast_output.data_sources_used:
        market_context_parts.append(f"Model data sources: {', '.join(forecast_output.data_sources_used)}")
    if forecast_output and forecast_output.model_info.get("features_used"):
        market_context_parts.append(f"Features used in model: {', '.join(forecast_output.model_info['features_used'])}")
    market_context = "\n".join(market_context_parts)

    # Inject forecast data into tool_results for summarizer
    if forecast_output:
        forecast_state["tool_results"]["__forecast"] = {
            "result": json.dumps(forecast_output.raw, default=str),
            "args": {"type": "forecast"}
        }
    if market_context:
        forecast_state["tool_results"]["__market_context"] = {
            "result": market_context,
            "args": {"type": "market_context"}
        }

    # Build tool_results_for_summarizer from forecast_state
    tool_results_for_summarizer = {}
    for call_id, result in forecast_state["tool_results"].items():
        if isinstance(result, dict) and "result" in result:
            tool_results_for_summarizer[call_id] = {
                "result": json.dumps(result["result"], default=str) if not isinstance(result["result"], str) else result["result"],
                "args": result.get("args", {})
            }
        else:
            tool_results_for_summarizer[call_id] = {
                "result": json.dumps(result, default=str) if isinstance(result, dict) else str(result),
                "args": {}
            }

    # Add special forecasting context
    if "__forecast" in tool_results_for_summarizer:
        tool_results_for_summarizer["__forecast"] = forecast_state["tool_results"]["__forecast"]
    if "__market_context" in tool_results_for_summarizer:
        tool_results_for_summarizer["__market_context"] = forecast_state["tool_results"]["__market_context"]

    action_context = plan.get("action_context", "")

    answer = summarize_results(
        user_query=user_query,
        tool_results=tool_results_for_summarizer,
        conversation_history=conversation_history,
        chart_config=None,
        call_llm_fn=call_llm,
        large_result_threshold=LARGE_RESULT_THRESHOLD,
        tone_context=tone_context,
        action_context=action_context,
        export_url=forecast_state.get("export_url", ""),
        wants_export=plan.get("needs_export", False) or intent_result.wants_export,
        code_context="",
    )

    save_session_state(auth_context, {
        "last_query": user_query,
        "last_intent": intent_result.intent,
        "last_intent_category": intent_result.intent_category,
        "last_data_scope": intent_result.data_scope,
        "last_chart_data": [],
        "last_chart_config": {},
        "last_tool_results": forecast_state.get("tool_results") or session_state.get("last_tool_results", {}),
        "last_export_url": forecast_state.get("export_url", "") or session_state.get("last_export_url", ""),
        "last_forecast_result": forecast_output.raw if forecast_output else session_state.get("last_forecast_result"),
    })
    return answer


def _apply_client_side_binning(state: Dict, plan: Dict) -> Optional[str]:
    """Apply client-side binning to the first suitable tool result.

    Returns the output column name when binning was applied, else None.
    """
    binned_output_column = None
    bin_config = plan.get("client_side_binning")
    if not (bin_config and isinstance(bin_config, dict)):
        return None

    logger.info("CLIENT_SIDE_BINNING: plan includes binning config: %s", bin_config)
    binned_any = False
    for call_id, result in state["tool_results"].items():
        logger.info("CLIENT_SIDE_BINNING: checking call_id=%s result_type=%s", call_id, type(result).__name__)
        items = extract_items(result)
        logger.info("CLIENT_SIDE_BINNING: extracted %d items from %s", len(items), call_id)
        if not items:
            continue

        logger.info("CLIENT_SIDE_BINNING: sample item keys=%s", list(items[0].keys()) if items else [])

        column = bin_config.get("column", "")
        has_count = any("count" in k.lower() or "total" in k.lower() or "sum" in k.lower()
                        for k in (items[0].keys() if items else []))
        has_raw_only = len(items[0].keys() if items else []) == 1 and column in (items[0] if items else {})

        dynamic_config = dict(bin_config)
        if has_raw_only and not has_count:
            dynamic_config["method"] = "pre_aggregate"
            logger.info("CLIENT_SIDE_BINNING: auto-detected pre_aggregate (raw records without count)")
        else:
            dynamic_config["method"] = "post_aggregate"
            logger.info("CLIENT_SIDE_BINNING: auto-detected post_aggregate (grouped data with count)")

        binned = apply_client_side_binning(items, dynamic_config)
        if binned and len(binned) > 0:
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
    return binned_output_column


async def _resolve_rag_and_code(state: Dict, user_query: str, auth_context: Any, tenant_id: str, plan: Dict) -> str:
    """Retrieve RAG context and scan tool results for code context.

    Returns the code-context markdown string (possibly empty).
    """
    if plan.get("rag"):
        state["rag_context"] = retrieve_policy_context(plan.get("rag_query", user_query))

    code_context_md = ""
    tenant_id_for_codes = auth_context.tenant_id if auth_context and auth_context.tenant_id else "default"
    try:
        from agent.dab.code_resolver import scan_for_codes
        code_context_md = await scan_for_codes(state["tool_results"], tenant_id_for_codes)
        if code_context_md:
            logger.info("CODE_RESOLVER: injected code context (%d chars) into LLM", len(code_context_md))
    except Exception as e:
        logger.warning("CODE_RESOLVER: failed to scan results: %s", e)
    return code_context_md


def _resolve_chart_config(
    plan: Dict, state: Dict, session_state: Dict, intent_result: Any,
    user_query: str, binned_output_column: Optional[str],
) -> Tuple[Optional[Dict], List[Dict]]:
    """Resolve chart_config and chartable_data from plan, session, and tool results.

    Returns (chart_config, chartable_data).
    """
    chartable_data = []
    cached_cfg = session_state.get("last_chart_config")
    if intent_result.wants_export and cached_cfg:
        chart_config = dict(cached_cfg)
        logger.info(
            "FOLLOW_UP_EXPORT: restored chart_config from session (type=%s), overriding planner default",
            chart_config.get("type")
        )
        # Only restore cached data when there are no fresh tool results.
        if not state.get("tool_calls_made") and not chartable_data and session_state.get("last_chart_data"):
            chartable_data = list(session_state["last_chart_data"])
            logger.info("FOLLOW_UP_EXPORT: restored %d cached rows", len(chartable_data))
    else:
        chart_config = plan.get("chart")

    if chart_config and binned_output_column:
        chart_config["x_column"] = binned_output_column
        chart_config["title"] = f"{binned_output_column.replace('_', ' ').title()} Distribution"
        logger.info("CHART_DEBUG_EXECUTOR: synced chart_config to binned column=%s title=%s", binned_output_column, chart_config["title"])

    if chart_config and state["tool_results"]:
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

    return chart_config, chartable_data


def _generate_chart_markdown(
    chart_config: Optional[Dict], chartable_data: List[Dict],
    auth_role: str, user_query: str, tenant_id: str,
    is_custom_ui: bool = False,
) -> str:
    """Generate chart markdown from resolved chart config and data.

    When is_custom_ui is True, emits a structured Chart.js payload via the
    hr_chart side-channel (emit_chart) and returns empty markdown -- the
    custom UI renders the chart natively instead of from a markdown image.
    """
    if not chart_config:
        logger.info("CHART_DEBUG_EXECUTOR: no chart_config in plan")
        return ""

    if chart_config and isinstance(chart_config, dict) and chartable_data:
        # If planner left chart metadata empty (common for HANA raw-SQL paths),
        # derive x/y/title from the actual result columns and the user query.
        if not chart_config.get("y_column") and chartable_data:
            first_row = chartable_data[0]
            cols = list(first_row.keys())
            q = user_query.lower()
            metric_keywords = [
                "profit", "revenue", "income", "expense", "cost", "margin",
                "amount", "balance", "salary", "headcount", "count", "total",
                "net", "ebitda", "turnover", "rate", "value", "quantity",
            ]
            matched_col = None
            for kw in metric_keywords:
                candidates = [c for c in cols if kw in c.lower()]
                if candidates:
                    matched_col = candidates[0]
                    break
            if matched_col:
                chart_config["y_column"] = matched_col
                logger.info("CHART_DEBUG_EXECUTOR: derived y_column='%s' from user query", matched_col)
            else:
                # Fallback: first numeric-looking column
                import pandas as pd
                df = pd.DataFrame(chartable_data)
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if numeric_cols:
                    chart_config["y_column"] = numeric_cols[0]
                    logger.info("CHART_DEBUG_EXECUTOR: fallback y_column='%s' from numeric cols", numeric_cols[0])

        # --- Custom UI: emit structured Chart.js payload, skip markdown ---
        if is_custom_ui:
            chart_payload = build_chartjs_payload(chart_config, chartable_data, title=chart_config.get("title"))
            if chart_payload:
                emit_chart(chart_payload)
                emit_progress("charting", "Rendering chart...")
            # Return empty markdown -- the custom UI renders via hr_chart.
            # Still include the data table so the UI can show raw numbers.
            from agent.output.chart_generator import generate_data_table
            return generate_data_table(chartable_data) if chartable_data else ""

        chart_markdown = generate_chart(
            data=chartable_data,
            chart_type=chart_config.get("type", "bar"),
            x_column=chart_config.get("x_column"),
            y_column=chart_config.get("y_column"),
            title=chart_config.get("title"),
            y_label=chart_config.get("y_label"),
            auth_role=auth_role,
            include_table=True,
            multi_series=chart_config.get("multi_series", False),
            metrics=chart_config.get("metrics", []),
            tenant_id=tenant_id,
        )
        logger.info("CHART_DEBUG_EXECUTOR: generated chart_markdown len=%d", len(chart_markdown))
    elif not chart_config and chartable_data:
        # Defensive fallback: if planner missed chart intent but we have chartable data,
        # generate a default bar chart for HANA / aggregate queries that look chartable.
        q = user_query.lower()
        looks_like_chart_request = any(
            kw in q for kw in ("chart", "graph", "visualize", "plot", "bar", "pie", "line")
        )
        has_multiple_cols = bool(chartable_data) and isinstance(chartable_data[0], dict) and len(chartable_data[0]) >= 2
        if looks_like_chart_request and has_multiple_cols:
            logger.info("CHART_DEBUG_EXECUTOR: fallback chart generation for missed chart intent")

            # --- Custom UI: emit structured payload for fallback chart too ---
            if is_custom_ui:
                fallback_config = {"type": "bar", "x_column": None, "y_column": None,
                                   "title": None, "y_label": None, "multi_series": False,
                                   "metrics": []}
                chart_payload = build_chartjs_payload(fallback_config, chartable_data)
                if chart_payload:
                    emit_chart(chart_payload)
                    emit_progress("charting", "Rendering chart...")
                from agent.output.chart_generator import generate_data_table
                return generate_data_table(chartable_data) if chartable_data else ""

            chart_markdown = generate_chart(
                data=chartable_data,
                chart_type="bar",
                x_column=None,
                y_column=None,
                title=None,
                y_label=None,
                auth_role=auth_role,
                include_table=True,
            )
            logger.info("CHART_DEBUG_EXECUTOR: fallback generated chart_markdown len=%d", len(chart_markdown))
        else:
            chart_markdown = ""
    else:
        chart_markdown = ""

    return chart_markdown


def _handle_export(
    plan: Dict, intent_result: Any, state: Dict, chart_config: Optional[Dict],
    chartable_data: List[Dict], auth_context: Any, auth_role: str,
) -> None:
    """Generate export URLs (chart export or data-only Excel) into state."""
    if not ((plan.get("needs_export") or intent_result.wants_export) and chart_config
            and isinstance(chart_config, dict) and chartable_data):
        pass
    else:
        metadata = build_export_metadata(chart_config, auth_context, auth_role, "", intent_result, state.get("tool_calls_made", []))
        export_url = generate_chart_export(chartable_data, chart_config, auth_role, metadata, state.get("tool_calls_made", []))
        if export_url:
            state["export_url"] = export_url
        else:
            prefix = derive_export_prefix(chart_config, (state.get("tool_calls_made", [{}])[0].get("args", {}) or {}).get("entity", ""))
            export_url = export_to_excel(chartable_data, prefix=prefix or "export")
            if export_url:
                state["export_url"] = export_url
                logger.info("EXECUTOR_EXPORT: generated data-only Excel export: %s", export_url)

    if (plan.get("needs_export") or intent_result.wants_export) and not state.get("export_url"):
        is_self_access = (
            auth_context
            and getattr(auth_context, "permissions", None)
            and "read:self" in auth_context.permissions
        )
        for call_id, result in state["tool_results"].items():
            items = extract_items(result)
            if items and isinstance(items, list) and len(items) > 0:
                is_self_access = is_self_access or is_data_about_user(items, auth_context)
                export_url = generate_data_export(items, strict_privacy=not is_self_access)
                if export_url:
                    state["export_url"] = export_url
                    logger.info(
                        "EXECUTOR_FULLINFO_EXPORT: generated data-only Excel export for %d records: %s",
                        len(items), export_url,
                    )
                break


def _build_tool_results_for_summarizer(
    state: Dict, chart_markdown: str, chart_config: Optional[Dict],
    chartable_data: List[Dict],
) -> Dict:
    """Build the tool_results dict consumed by the summarizer."""
    tool_results_for_summarizer = {}
    for i, tc in enumerate(state["tool_calls_made"]):
        call_id = f"{tc['tool']}_{i}"
        result = state["tool_results"].get(call_id, {})
        tool_results_for_summarizer[call_id] = {
            "result": json.dumps(result, default=str),
            "args": tc.get("args", {}),
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

    # Inject variance analysis for multi-metric financial reports
    variance_text = ""
    if chartable_data and isinstance(chart_config, dict) and chart_config.get("multi_series"):
        metrics = chart_config.get("metrics", []) or _detect_wide_format_metrics(chartable_data, None)
        x_col = next(iter(chartable_data[0].keys())) if chartable_data else "period"
        for candidate in ["POPER", "MONTH", "PERIOD", "YEAR", "QUARTER"]:
            if candidate in chartable_data[0]:
                x_col = candidate
                break
        variances = _calculate_variances(chartable_data, metrics, x_col)
        variance_text = _format_variances_for_prompt(variances)
        if variance_text:
            tool_results_for_summarizer["__variances"] = {
                "result": variance_text,
                "args": {"type": "variance_analysis"}
            }
            logger.info("CHART_DEBUG_VARIANCE: injected __variances len=%d", len(variance_text))

    return tool_results_for_summarizer


def _plan_has_write_steps(plan: Dict) -> bool:
    """True when the plan contains any write steps (create/update/delete)."""
    return any(
        s.get("tool") in ("create_record", "update_record", "delete_record")
        for s in plan.get("steps", [])
        if isinstance(s, dict)
    )


async def _handle_chart_clarification(
    plan: Dict, user_query: str, conversation_history: str, tone_context: Dict,
) -> Optional[str]:
    """Ask the user to clarify an ambiguous chart request before executing steps.

    Returns a clarification answer string when clarification is needed,
    else None (caller proceeds with normal execution).
    """
    if not plan.get("chart_intent_clarification_needed"):
        return None
    if _plan_has_write_steps(plan):
        return None

    clarification_reason = plan.get("chart_intent_clarification_reason", "your request could mean different things")
    logger.info("CHART_INTENT_CLARIFICATION: %s", clarification_reason)
    try:
        clarification_answer = summarize_results(
            user_query=user_query,
            tool_results={},
            conversation_history=conversation_history,
            chart_config=None,
            call_llm_fn=call_llm,
            large_result_threshold=LARGE_RESULT_THRESHOLD,
            tone_context=tone_context,
            action_context="chart_intent_clarification",
            export_url="",
            wants_export=False,
            code_context="",
        )
        return clarification_answer
    except Exception as exc:
        logger.warning("CHART_INTENT_CLARIFICATION: summarizer failed (%s), using fallback", exc)
        return (
            f"I want to make sure I show you the right chart. "
            f"{clarification_reason.capitalize()}. "
            f"Could you clarify what exactly you'd like to see?"
        )


async def _handle_pending_slot_clarification(
    plan: Dict, user_query: str, conversation_history: str, tone_context: Dict,
    auth_context: Any = None, tenant_id: str = "LOCALDEV",
    cached_schema: Dict = None, cached_tools: List[Dict] = None,
    hana_available: bool = True,
) -> Optional[str]:
    """Surface the pending_slot's user_question when a field is missing a value.

    Before asking, executes any READ-ONLY steps already present in the plan
    (DAB read_records / aggregate_records / describe_entities only -- never
    writes) so the clarification is grounded in real data when available.
    Incomplete leave applications intentionally contain no entitlement read;
    the action injector asks for the missing leave type and date first.

    Returns a clarification answer string when clarification is needed,
    else None (caller proceeds with normal execution).
    """
    if not plan.get("pending_slots"):
        return None
    if _plan_has_write_steps(plan):
        return None

    pending = plan["pending_slots"][0]
    user_question = pending.get("user_question", "Could you provide more details?")
    entity = pending.get("entity", "the record")
    field = pending.get("field", "")
    logger.info(
        "PENDING_SLOT: %s.%s needs clarification: %s",
        entity, field, user_question,
    )

    # Execute read-only steps so the clarification presents real data (e.g.
    # the applicant's leave entitlement) instead of asking blind. Write
    # steps are excluded by _plan_has_write_steps above; HANA tools are
    # excluded because this path is for grounding the question, not
    # running the full plan.
    tool_results_for_summarizer: Dict = {}
    read_only_steps = [
        s for s in plan.get("steps", [])
        if isinstance(s, dict) and s.get("tool") in ("read_records", "aggregate_records", "describe_entities")
    ]
    if read_only_steps and auth_context is not None:
        state = {
            "tool_results": {},
            "tool_calls_made": [],
            "rag_context": "",
        }
        try:
            await _run_step_loop(
                steps=read_only_steps,
                state=state,
                auth_context=auth_context,
                tenant_id=tenant_id,
                cached_schema=cached_schema or {},
                cached_tools=cached_tools,
                hana_available=hana_available,
            )
        except Exception as exc:
            logger.warning("PENDING_SLOT: read-only grounding steps failed: %s", exc)
        if state["tool_results"]:
            for i, tc in enumerate(state["tool_calls_made"]):
                call_id = f"{tc['tool']}_{i}"
                result = state["tool_results"].get(call_id, {})
                tool_results_for_summarizer[call_id] = {
                    "result": json.dumps(result, default=str),
                    "args": tc.get("args", {}),
                }
            logger.info(
                "PENDING_SLOT: grounded clarification with %d read-only step result(s)",
                len(tool_results_for_summarizer),
            )

    # Propagate pending_slots into tone_context so the summarizer's
    # build_response_structure() can emit the right STAGE 2 guidance.
    tone_context = dict(tone_context)
    tone_context["pending_slots"] = plan["pending_slots"]
    try:
        clarification_answer = summarize_results(
            user_query=user_query,
            tool_results=tool_results_for_summarizer,
            conversation_history=conversation_history,
            chart_config=None,
            call_llm_fn=call_llm,
            large_result_threshold=LARGE_RESULT_THRESHOLD,
            tone_context=tone_context,
            action_context="pending_slot_clarification",
            export_url="",
            wants_export=False,
            code_context="",
        )
        return clarification_answer
    except Exception as exc:
        logger.warning("PENDING_SLOT: summarizer failed (%s), using fallback", exc)
        return user_question


# _is_plausible_slot_value and _QUESTION_STARTERS now live in
# agent.actions.base_injector (shared with simple_update_strategy).


async def _handle_pending_update(
    pending: Dict, user_query: str, auth_context: Any,
    cached_tools: List[Dict], cached_schema: Dict,
    conversation_history: str,
) -> Optional[str]:
    """Bind a follow-up reply to a pending update field and execute it.

    Mirrors the leave-confirmation re-execution pattern: when a prior turn
    stored a pending action (field mentioned without a value), the user's
    current reply is treated as the value for that field.

    Returns the response string when the pending action was consumed
    (value bound + update executed, or re-asked), or None when the pending
    action does not apply to a simple_update entity (caller falls through
    to normal planning).
    """
    from agent.core.session_state import clear_pending_action
    from agent.actions.strategies.simple_update_strategy import handle_pending_update
    from agent.actions.base_injector import is_plausible_slot_value

    entity = pending.get("entity", "")
    field = pending.get("field", "")
    tenant_id = auth_context.tenant_id if auth_context and auth_context.tenant_id else "LOCALDEV"

    q_lower = user_query.strip().lower()

    # Explicit decline — cancel the pending update.
    if q_lower in ("no", "cancel", "stop", "never mind", "nevermind", "forget it"):
        clear_pending_action(auth_context)
        logger.info("PENDING_UPDATE: user declined pending update for %s.%s", entity, field)
        return "Update cancelled. No changes were made."

    # Ambiguous affirmation ("yes"/"ok") is not a value — re-ask.
    if _is_short_affirmative(user_query):
        question = pending.get("user_question") or f"What would you like to set your {field.replace('_', ' ')} to?"
        logger.info("PENDING_UPDATE: ambiguous affirmation for %s.%s — re-asking", entity, field)
        return question

    # Question-shaped or long replies are new requests, not bare values.
    # Clear the stale pending action so it doesn't hijack the new request
    # when the planner's post-processors (inject_simple_update) reload it.
    if not is_plausible_slot_value(user_query):
        clear_pending_action(auth_context)
        logger.info(
            "PENDING_UPDATE: query %r is not a plausible slot value for %s.%s — "
            "clearing stale pending action, falling through to normal planning",
            user_query, entity, field,
        )
        return None

    # Delegate to the strategy layer: bind the reply to the pending field,
    # build read+update steps, and return them (or a re-ask message).
    result = handle_pending_update(
        user_query, entity, field, auth_context,
        cached_schema, cached_tools, conversation_history, tenant_id,
    )
    if result is None:
        # Not a simple_update entity — fall through to normal planning.
        return None

    steps, response = result
    if steps is None:
        # Value not understood — re-ask, keep the pending action.
        return response

    # Execute the read + update steps.
    state = {
        "tool_results": {},
        "tool_calls_made": [],
        "rag_context": "",
        "_pending_write_steps": steps,
    }
    await _run_step_loop(
        steps=steps,
        state=state,
        auth_context=auth_context,
        tenant_id=tenant_id,
        cached_schema=cached_schema,
        cached_tools=cached_tools,
    )

    # The pending action was fulfilled (value bound + steps built) — clear it
    # so a stale slot doesn't hijack a future turn.
    clear_pending_action(auth_context)
    logger.info("PENDING_UPDATE: fulfilled and cleared %s.%s", entity, field)

    # Summarize the result.
    if not state["tool_results"]:
        return "I wasn't able to update that field. Please try again."

    code_context_md = await _resolve_rag_and_code(state, user_query, auth_context, tenant_id, {"steps": steps})
    tool_results_for_summarizer = _build_tool_results_for_summarizer(
        state, "", None, [],
    )
    intent_result = await classify_intent(user_query, conversation_history)
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
        "is_finance_query": intent_result.finance_query,
    }
    answer = summarize_results(
        user_query=user_query,
        tool_results=tool_results_for_summarizer,
        conversation_history=conversation_history,
        chart_config=None,
        call_llm_fn=call_llm,
        large_result_threshold=LARGE_RESULT_THRESHOLD,
        tone_context=tone_context,
        action_context="profile_update_completed",
        export_url="",
        wants_export=False,
        code_context=code_context_md,
    )
    return answer


async def _handle_implicit_update(
    user_query: str, auth_context: Any,
    cached_tools: List[Dict], cached_schema: Dict,
    conversation_history: str,
) -> Optional[str]:
    """Handle an implicit life-event statement ("I'm married now").

    Uses the strategy layer to extract field+value via LLM fallback, then
    routes through the profile-update confirmation gate. The update is NOT
    executed — instead a confirmation message is returned so the user can
    confirm before the write happens.

    Returns the confirmation response string, or None when no field+value
    could be resolved (caller falls through to normal planning).
    """
    from agent.actions.strategies.simple_update_strategy import (
        extract_update_fields,
        profile_update_confirmation_gate,
        is_personal_state_statement,
        DEFAULT_FIELD_PATTERNS,
    )
    from agent.actions.base_injector import get_entity_config, get_entity_fields_from_schema

    entity_name = "employee_general"
    config = get_entity_config(entity_name)
    if not config or config.get("strategy") != "simple_update":
        return None

    allowed = set(config.get("self_updateable_fields", []))
    if not allowed:
        return None

    if entity_name not in cached_schema:
        return None

    tid = auth_context.tenant_id if auth_context and auth_context.tenant_id else "LOCALDEV"

    # Extract field+value via LLM fallback (implicit_update=True)
    updates, _mentioned, _slot = extract_update_fields(
        user_query,
        list(allowed),
        DEFAULT_FIELD_PATTERNS,
        conversation_history=conversation_history,
        tenant_id=tid,
        implicit_update=True,
    )

    if not updates:
        # LLM couldn't resolve a field+value from the life-event statement.
        # Fall through to normal planning.
        return None

    # Route through the confirmation gate — stores steps in session state
    # and returns (steps, summary) for the summarizer to present.
    gate_result = profile_update_confirmation_gate(
        entity_name, updates, auth_context, cached_schema,
        cached_tools, conversation_history, tid,
    )
    if gate_result is None:
        return None

    _steps, summary = gate_result

    # Build a confirmation response using the summarizer
    intent_result = await classify_intent(user_query, conversation_history)
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
        "is_finance_query": intent_result.finance_query,
    }

    answer = summarize_results(
        user_query=user_query,
        tool_results={},
        conversation_history=conversation_history,
        chart_config=None,
        call_llm_fn=call_llm,
        large_result_threshold=LARGE_RESULT_THRESHOLD,
        tone_context=tone_context,
        action_context="profile_update_confirmation",
        export_url="",
        wants_export=False,
        code_context="",
    )
    return answer


async def _execute_pending_update_confirmation(
    pending: Dict, auth_context: Any,
    cached_tools: List[Dict], cached_schema: Dict,
    conversation_history: str, user_query: str,
) -> str:
    """Re-execute the stored profile-update steps after user confirmation.

    Mirrors _execute_pending_leave_confirmation: re-runs the full step chain
    (read + update with $ref chaining) through the same executor path.
    """
    from agent.core.intent_classifier import classify_intent
    from agent.core.tool_planner import build_tool_plan
    from agent.summarizer.response_summarizer import summarize_results
    from agent.integrations.llm_client import call_llm
    from agent.config import LARGE_RESULT_THRESHOLD

    tenant_id = auth_context.tenant_id if auth_context and auth_context.tenant_id else "LOCALDEV"
    steps = pending.get("steps", [])
    summary = pending.get("summary", "")

    state = {
        "tool_results": {},
        "tool_calls_made": [],
        "rag_context": "",
        # The user already confirmed this update — the confirmation gate
        # must not re-intercept the re-executed writes.
        "_confirmed_update_write": True,
    }

    await _run_step_loop(
        steps=steps,
        state=state,
        auth_context=auth_context,
        tenant_id=tenant_id,
        cached_schema=cached_schema,
        cached_tools=cached_tools,
    )

    # Build summarizer input
    tool_results_for_summarizer = {}
    for i, tc in enumerate(state["tool_calls_made"]):
        call_id = f"{tc['tool']}_{i}"
        result = state["tool_results"].get(call_id, {})
        tool_results_for_summarizer[call_id] = {
            "result": json.dumps(result, default=str),
            "args": tc.get("args", {}),
        }

    intent_result = await classify_intent(user_query, conversation_history)
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
        "is_finance_query": intent_result.finance_query,
    }

    answer = summarize_results(
        user_query=user_query,
        tool_results=tool_results_for_summarizer,
        conversation_history=conversation_history,
        chart_config=None,
        call_llm_fn=call_llm,
        large_result_threshold=LARGE_RESULT_THRESHOLD,
        tone_context=tone_context,
        action_context="profile_update_confirmed",
        export_url="",
        wants_export=False,
        code_context="",
    )
    return answer


def _handle_dashboard(plan: Dict) -> Optional[str]:
    """Return a dashboard URL response for multi-chart dashboard plans, else None."""
    if not plan.get("multi_chart"):
        return None
    if not plan.get("dashboard_queries"):
        logger.info("MULTI_CHART_DASHBOARD: multi_chart flag set but no dashboard_queries provided")
        return None

    dashboard_queries = plan.get("dashboard_queries", [])
    # URL-encode the JSON payload so browsers parse it correctly, and
    # use the absolute agent base URL (same host that serves
    # /dashboard.html and /exports/).
    from urllib.parse import quote
    from agent.config import AGENT_BASE_URL
    encoded_queries = quote(json.dumps(dashboard_queries))
    dashboard_url = f"{AGENT_BASE_URL}/dashboard.html?queries={encoded_queries}"
    logger.info("MULTI_CHART_DASHBOARD: returning dashboard URL with %d queries", len(dashboard_queries))
    # Markdown link (same convention as export URLs) so chat clients
    # render a clean clickable label instead of the raw URL.
    return (
        f"I've built a multi-chart dashboard for you. "
        f"[View it here]({dashboard_url})"
    )


async def run_reflexive_agent(
    user_query: str,
    conversation_history: str,
    auth_context: Any,
    cached_tools: List[Dict],
    cached_schema: Dict,
    is_custom_ui: bool = False,
) -> str:
    # -- Pending leave confirmation check (must run before any keyword/intent
    # gate so a short "yes"/"no" follow-up is not rejected by greeting or
    # smalltalk guards). --
    from agent.core.session_state import load_pending_leave_confirmation, clear_pending_leave_confirmation
    pending_confirm = load_pending_leave_confirmation(auth_context)
    if pending_confirm:
        if _is_short_affirmative(user_query):
            # User confirmed — re-execute the stored leave steps.
            logger.info(
                "CONFIRMATION_TURN: user affirmed pending leave application for %s",
                pending_confirm.get("entity"),
            )
            answer = await _execute_pending_leave_confirmation(
                pending_confirm, auth_context, cached_tools, cached_schema,
                conversation_history, user_query,
            )
            clear_pending_leave_confirmation(auth_context)
            return answer
        elif user_query.strip().lower() in ("no", "cancel", "stop", "never mind", "nevermind"):
            logger.info("CONFIRMATION_TURN: user declined pending leave application")
            clear_pending_leave_confirmation(auth_context)
            return "Leave application cancelled. No changes were made."
        # If the user's reply is neither a clear affirmation nor a decline,
        # fall through to normal processing — the pending confirmation stays
        # in session state so the next turn can still resolve it.

    # -- Pending update-slot check (must run before any keyword/intent gate
    # so a bare value reply like "married" is not rejected by the
    # action-request or greeting guards). Mirrors the leave-confirmation
    # pattern above: the user previously asked to update a field but did not
    # provide a value; their current reply is that value. --
    from agent.core.session_state import load_pending_action
    pending_action = load_pending_action(auth_context)
    if pending_action:
        answer = await _handle_pending_update(
            pending_action, user_query, auth_context,
            cached_tools, cached_schema, conversation_history,
        )
        if answer is not None:
            return answer
        # None => the pending action doesn't apply to a simple_update entity
        # (e.g. a leave days slot); fall through to normal planning.

    # -- Pending update-confirmation check (must run before any keyword/intent
    # gate so a short "yes"/"no" follow-up is not rejected by greeting or
    # smalltalk guards). Mirrors the leave-confirmation pattern: when a
    # prior turn stored a profile-update confirmation (implicit life-event
    # statement), the user's current reply confirms or declines. --
    from agent.core.session_state import load_pending_update_confirmation, clear_pending_update_confirmation
    pending_update_confirm = load_pending_update_confirmation(auth_context)
    if pending_update_confirm:
        if _is_short_affirmative(user_query):
            logger.info(
                "CONFIRMATION_TURN: user affirmed pending profile update for %s",
                pending_update_confirm.get("entity"),
            )
            answer = await _execute_pending_update_confirmation(
                pending_update_confirm, auth_context, cached_tools, cached_schema,
                conversation_history, user_query,
            )
            clear_pending_update_confirmation(auth_context)
            return answer
        elif user_query.strip().lower() in ("no", "cancel", "stop", "never mind", "nevermind", "forget it"):
            logger.info("CONFIRMATION_TURN: user declined pending profile update")
            clear_pending_update_confirmation(auth_context)
            return "Update cancelled. No changes were made."
        # If the user's reply is neither a clear affirmation nor a decline,
        # fall through to normal processing — the pending confirmation stays
        # in session state so the next turn can still resolve it.

    # -- Implicit life-event statement interception --
    # First-person statements like "I'm married now" imply a profile update
    # without an explicit action verb. These bypass the action-keyword and
    # intent gates so the LLM extraction can resolve field+value, then
    # confirm before committing.
    from agent.actions.strategies.simple_update_strategy import is_personal_state_statement
    if is_personal_state_statement(user_query):
        answer = await _handle_implicit_update(
            user_query, auth_context, cached_tools, cached_schema,
            conversation_history,
        )
        if answer is not None:
            return answer
        # None => no field+value could be resolved; fall through to normal planning.

    is_gs, gs_label = is_greeting_or_smalltalk(user_query)
    if is_gs:
        logger.info("L1 guard triggered: %s", gs_label)
        return greeting_response(gs_label)

    if is_vague_data_question(user_query):
        logger.info("L2 guard triggered: vague data question")
        return data_summary_response(cached_schema)

    session_state = load_session_state(auth_context)

    emit_progress("intent", "Understanding your request...")
    intent_result = await classify_intent(user_query, conversation_history)
    logger.info(
        "Intent classified: intent=%s category=%s urgency=%s emotional=%s empathy=%s confidence=%.2f action_oriented=%s",
        intent_result.intent, intent_result.intent_category, intent_result.urgency_level,
        intent_result.emotional_state, intent_result.needs_empathy, intent_result.confidence,
        intent_result.action_oriented
    )

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
        "is_finance_query": intent_result.finance_query,
    }

    if (
        intent_result.intent_category == "data_discovery"
        and _is_aggregate_query_despite_discovery_classification(user_query)
    ):
        logger.info(
            "INTENT_CORRECTION: aggregate query misclassified as data_discovery; correcting planner context"
        )
        tone_context = dict(tone_context)
        tone_context["intent_category"] = "aggregate_data"
        try:
            from agent.core.intent_config import get_category_config
            corrected_config = get_category_config("aggregate_data")
            tone_context["chart_eligible"] = corrected_config.get("chart_eligible", True)
        except Exception as exc:
            logger.warning("INTENT_CORRECTION: failed to refresh chart_eligible: %s", exc)
            tone_context["chart_eligible"] = True

    tenant_id = auth_context.tenant_id if auth_context and auth_context.tenant_id else "LOCALDEV"

    # If HANA was observed to be down (e.g. at startup), skip HANA tool discovery
    # and live schema-registry discovery during the request so we don't retry
    # connections against a dead server on every user query.
    hana_available = is_hana_available()
    if not hana_available:
        logger.info("HANA unavailable -- excluding HANA tools from planner and skipping live schema discovery")

    logger.info("Planning with tool_planner for: %s", user_query)
    emit_progress("planning", "Planning how to answer...")
    plan = await build_tool_plan(
        user_query=user_query, conversation_history=conversation_history,
        auth_context=auth_context, cached_tools=cached_tools,
        cached_schema=cached_schema, rag_available=True, tone_context=tone_context,
        hana_schema_registry=schema_registry_service.get_registry(tenant_id) if hana_available else {},
    )

    # -- Pending-slot clarification -- when the action injector detected a field
    # mentioned without a value (e.g., "apply 1 day leave" with no date/leave
    # type), surface the pending_slot's user_question so the agent asks for the
    # missing value instead of silently proceeding or returning a generic
    # message. Runs BEFORE the chart-clarification handler because the planner's
    # action-clarification gate reuses chart_intent_clarification_needed for
    # pending_slots -- the chart handler would otherwise swallow these turns.
    # The handler executes the plan's read-only steps first so the question is
    # grounded in real data (e.g. the applicant's leave entitlement).
    # Guard: never fire when the plan contains write steps — if a write step
    # was produced, the value was already resolved and no clarification is needed.
    clarification = await _handle_pending_slot_clarification(
        plan, user_query, conversation_history, tone_context,
        auth_context=auth_context, tenant_id=tenant_id,
        cached_schema=cached_schema, cached_tools=cached_tools,
        hana_available=hana_available,
    )
    if clarification is not None:
        return clarification

    # -- Chart intent clarification -- ask the user to clarify ambiguous chart requests
    # before executing any steps. This prevents wasted tool calls on wrong chart configs.
    # Guard: never fire when the plan contains write steps (create/update/delete) —
    # a chart-clarification question is meaningless for an action request (e.g., a
    # leave application) and would block the action pipeline from executing.
    clarification = await _handle_chart_clarification(
        plan, user_query, conversation_history, tone_context,
    )
    if clarification is not None:
        return clarification

    # Handle multi-chart dashboard requests via dynamic dashboard.
    # Named-dashboard plans carry BOTH dashboard_queries and data-retrieval
    # steps (the steps execute to verify data availability); the dashboard
    # link is the user-facing result either way.
    dashboard_answer = _handle_dashboard(plan)
    if dashboard_answer is not None:
        return dashboard_answer

    if plan.get("direct_answer"):
        if intent_result.wants_export and not (intent_result.wants_export and session_state.get("last_chart_data")):
            logger.info("Direct answer from planner")
            return plan["direct_answer"]

    if not plan.get("steps"):
        if intent_result.wants_export and session_state.get("last_chart_data"):
            logger.info("FOLLOW_UP_EXPORT: empty plan but session has cached data; proceeding to export")
            state = {
                "tool_results": session_state.get("last_tool_results", {}),
                "tool_calls_made": [],
                "rag_context": "",
                "export_url": session_state.get("last_export_url", ""),
            }
            chartable_data = session_state.get("last_chart_data", [])
            chart_config = session_state.get("last_chart_config", {})
        else:
            logger.info("No steps -- clarification needed")
            return plan.get("reasoning", "Could you clarify what you're looking for?")

    # Initialized for both the no-steps export path and the step-loop path;
    # _resolve_chart_config consumes it below.
    binned_output_column = None

    if plan.get("steps"):
        emit_progress("executing", "Retrieving data from HR records...")
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # FORECASTING PIPELINE -- handle forecasting queries end-to-end
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        is_forecasting = any(
            step.get("tool") in ("fetch_external_data", "generate_code", "execute_sandbox", "summarize_forecast")
            for step in plan.get("steps", [])
        )
        if is_forecasting:
            answer = await _run_forecasting_pipeline(
                user_query=user_query,
                conversation_history=conversation_history,
                auth_context=auth_context,
                tenant_id=tenant_id,
                plan=plan,
                intent_result=intent_result,
                session_state=session_state,
                tone_context=tone_context,
            )
            return answer

        # Normal non-forecasting step loop (dashboard requests handled earlier)
        state = {
            "tool_results": {},
            "tool_calls_made": [],
            "rag_context": "",
            # Full plan step chain, so the leave confirmation gate can store
            # every step (hd + detail) for the confirmation-turn re-execution.
            "_pending_write_steps": plan["steps"],
        }

        await _run_step_loop(
            steps=plan["steps"],
            state=state,
            auth_context=auth_context,
            tenant_id=tenant_id,
            cached_schema=cached_schema,
            cached_tools=cached_tools,
            hana_available=hana_available,
        )

        binned_output_column = _apply_client_side_binning(state, plan)

    if not state["tool_results"] and not (intent_result.wants_export and session_state.get("last_chart_data")):
        return "I wasn't able to retrieve any data for that request."

    code_context_md = await _resolve_rag_and_code(state, user_query, auth_context, tenant_id, plan)

    chart_config, chartable_data = _resolve_chart_config(
        plan, state, session_state, intent_result, user_query, binned_output_column,
    )

    auth_role, fallback_used, perms, roles = resolve_auth_role(auth_context)

    if fallback_used:
        logger.warning(
            "AUTH_ROLE_FALLBACK: auth_role resolved via fallback -- "
            "permissions=%s internal_roles=%s. "
            "Investigate why permissions were not populated by auth layer.",
            sorted(perms) if perms else None,
            sorted(roles) if roles else None
        )

    logger.info("AUTH_ROLE_RESOLVED: auth_role=%s permissions=%s internal_roles=%s",
                auth_role,
                sorted(perms) if perms else None,
                sorted(roles) if roles else None)

    chart_markdown = _generate_chart_markdown(chart_config, chartable_data, auth_role, user_query, tenant_id, is_custom_ui=is_custom_ui)

    _handle_export(plan, intent_result, state, chart_config, chartable_data, auth_context, auth_role)

    tool_results_for_summarizer = _build_tool_results_for_summarizer(
        state, chart_markdown, chart_config, chartable_data,
    )

    action_context = plan.get("action_context", "")

    emit_progress("summarizing", "Composing your answer...")
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

    save_session_state(auth_context, {
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


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# DAB TOOL EXECUTION
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

def _resolve_dab_role(auth_context: Any) -> str:
    """Map the agent's resolved auth role to a DAB role name.

    employee -> HRMS_EMPLOYEE, manager -> HRMS_MANAGER, admin -> HRMS_HR.
    Used for the X-MS-API-ROLE header so DAB authorizes writes.
    """
    auth_role, _, _, _ = resolve_auth_role(auth_context)
    return {
        "employee": "HRMS_EMPLOYEE",
        "manager": "HRMS_MANAGER",
        "admin": "HRMS_HR",
    }.get(auth_role, "HRMS_EMPLOYEE")


def _dab_role_for_tool(auth_context: Any, tool: str) -> str:
    """Select the DAB role header for a tool call.

    DAB only applies a custom role when the X-MS-API-ROLE header is present;
    without it a valid token still resolves to the built-in 'authenticated'
    role, which has no read/create on these entities. We always send the
    resolved role (HRMS_EMPLOYEE / HRMS_MANAGER / HRMS_HR) so both reads and
    writes are authorized against the entity permission grants. The agent
    still enforces row-level self-service filtering at the tool-args layer
    (see agent.auth.tool_guards.enforce_tool_args).
    """
    return _resolve_dab_role(auth_context)


def _record_blocked(state: Dict, call_id: str, tool: str, args: Dict, error: str, error_type: str) -> None:
    """Record a blocked DAB tool call (auth or schema validation failure)."""
    state["tool_results"][call_id] = {"error": error}
    state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"BLOCKED: {error}"})
    record_dab_tool_call(tool, "BLOCKED")
    record_dab_error(error_type, entity=args.get("entity"))


def _record_dab_error(state: Dict, call_id: str, tool: str, args: Dict, error_msg: str, entity: str = None) -> None:
    """Record a DAB-level error response (isError payload from the DAB API)."""
    state["tool_results"][call_id] = {"error": error_msg}
    state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {error_msg}"})
    record_dab_tool_call(tool, "ERROR")
    record_dab_error("DabError", entity=entity if entity is not None else args.get("entity"))


def _record_success(state: Dict, call_id: str, tool: str, args: Dict, result_text: str) -> None:
    """Record a successful DAB tool call."""
    state["tool_results"][call_id] = {"result": result_text, "args": args}
    state["tool_calls_made"].append({"tool": tool, "args": args, "status": "SUCCESS"})
    record_dab_tool_call(tool, "SUCCESS")


def _record_system_error(state: Dict, call_id: str, tool: str, args: Dict, exc: Exception) -> None:
    """Record an unexpected exception during DAB tool execution."""
    state["tool_results"][call_id] = {"error": str(exc)}
    state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {exc}"})
    record_dab_tool_call(tool, "ERROR")
    record_dab_error("SystemError", entity=args.get("entity"))


def _normalize_list_arg(args: Dict, key: str, as_string: bool = False, drop_values: Tuple = ()) -> None:
    """Normalize a list-typed DAB arg from stray string/list forms, in place.

    - Comma-separated string -> split, strip, drop empties
    - List -> strip each entry, drop empties
    - Any other type -> drop the arg (server default applies)
    - Empty result, or a single entry in drop_values (e.g. "*" for select)
      -> drop the arg

    When as_string, re-join to a comma-separated string (for 'select');
    otherwise store a clean list (for 'orderby'/'groupby').
    """
    val = args.get(key)
    if val is None:
        return
    if isinstance(val, str):
        parsed = [v.strip() for v in val.split(",") if v.strip()]
    elif isinstance(val, list):
        parsed = [str(v).strip() for v in val if str(v).strip()]
    else:
        args.pop(key, None)
        return
    if not parsed or (len(parsed) == 1 and parsed[0] in drop_values):
        args.pop(key, None)
    elif as_string:
        args[key] = ",".join(parsed)
    else:
        args[key] = parsed


def _normalize_aggregate_orderby(args: Dict) -> None:
    """Normalize aggregate_records 'orderby' to a direction string ("asc"/"desc").

    DAB aggregate_records 'orderby' is a direction string, NOT an array of
    sort expressions (that is read_records-only). The shipped DAB source
    validates it via GetString() and rejects arrays with UnexpectedError.
    Normalize any stray list/string forms to a valid direction, or drop the
    parameter to use the server default.
    """
    orderby_val = args.get("orderby")
    if orderby_val is None:
        return
    if isinstance(orderby_val, str):
        lowered = orderby_val.strip().lower()
        if lowered in ("asc", "desc"):
            args["orderby"] = lowered
        elif lowered.endswith("asc"):
            args["orderby"] = "asc"
        else:
            # Sort-expression strings like "salary desc" are invalid here;
            # "desc" is the server default, so drop the arg.
            args.pop("orderby", None)
    elif isinstance(orderby_val, list):
        # Arrays (e.g. ["m desc"], ["count desc"]) are invalid for
        # aggregate_records; salvage direction from the entries.
        joined = " ".join(str(o).strip().lower() for o in orderby_val if str(o).strip())
        if joined.endswith("asc") and not joined.endswith("desc"):
            args["orderby"] = "asc"
        else:
            args.pop("orderby", None)  # "desc" or unknown -> default
    else:
        args.pop("orderby", None)


async def _execute_dab_tool_call(step: Dict, state: Dict, auth_context: Any, tenant_id: str = "LOCALDEV", cached_schema: Dict = None, cached_tools: List[Dict] = None):
    tool = step.get("tool")
    args = step.get("args", {})
    call_id = f"{tool}_{len(state['tool_calls_made'])}"
    client = dab_manager.get_client(tenant_id, getattr(auth_context, "token", None), _dab_role_for_tool(auth_context, tool))

    normalize_odata_args(args, cached_schema)

    logger.debug("Executing DAB tool: %s with args: %s", tool, args)

    args, error = enforce_tool_args(tool, args, auth_context)
    if error:
        _record_blocked(state, call_id, tool, args, error, "AuthError")
        return

    args, schema_error = validate_dab_args(tool, args, cached_tools)
    if schema_error:
        _record_blocked(state, call_id, tool, args, schema_error, "ValidationError")
        return

    if tool == "read_records":
        _normalize_list_arg(args, "select", as_string=True, drop_values=("*",))
        _normalize_list_arg(args, "orderby")

    elif tool == "aggregate_records":
        _normalize_list_arg(args, "groupby")
        _normalize_aggregate_orderby(args)

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

        if "first" in args:
            if args.get("groupby"):
                logger.info("Stripping 'first'=%s from aggregate_records with groupby for complete chart data", args.get("first"))
            else:
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
                _record_dab_error(state, call_id, tool, args, error_msg)
                return

        if isinstance(dab_data, dict):
            result_text = json.dumps(dab_data, default=str)
        else:
            result_text = str(dab_data)
        filtered = filter_tool_results(tool, result_text, auth_context)

        try:
            parsed = json.loads(filtered) if isinstance(filtered, str) else filtered
        except (json.JSONDecodeError, TypeError):
            parsed = {"raw_text": filtered}

        state["tool_results"][call_id] = parsed
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": "SUCCESS"})
        logger.info("DAB tool %s succeeded (call_id=%s)", tool, call_id)
        record_dab_tool_call(tool, "SUCCESS")

    except Exception as e:
        logger.error("DAB tool %s failed: %s", tool, e)
        _record_system_error(state, call_id, tool, args, e)


async def _execute_dab_create_call(step: Dict, state: Dict, auth_context: Any, tenant_id: str = "LOCALDEV", cached_schema: Dict = None, cached_tools: List[Dict] = None):
    tool = step.get("tool")
    args = step.get("args", {})
    call_id = f"{tool}_{len(state['tool_calls_made'])}"
    client = dab_manager.get_client(tenant_id, getattr(auth_context, "token", None), _resolve_dab_role(auth_context))

    # enforce_tool_args injects employee_no and validates required fields
    args, error = enforce_tool_args(tool, args, auth_context)
    if error:
        _record_blocked(state, call_id, tool, args, error, "AuthError")
        return

    # Validate args against schema
    args, schema_error = validate_dab_args(tool, args, cached_tools)
    if schema_error:
        _record_blocked(state, call_id, tool, args, schema_error, "ValidationError")
        return

    try:
        record = args.get("data", {})
        entity = args.get("entity", "")
        if not isinstance(record, dict):
            raise ValueError("create_record 'data' argument must be an object")

        # Idempotency guard: prevent duplicate create_record actions in the same session
        action_fingerprint = hashlib.sha256(
            f"{tool}:{entity}:{json.dumps(record, sort_keys=True, default=str)}".encode()
        ).hexdigest()[:16]
        executed_actions = state.get("_executed_actions", set())
        if action_fingerprint in executed_actions:
            logger.warning("Duplicate create_record detected, skipping: %s", action_fingerprint)
            state["tool_results"][call_id] = {"result": json.dumps({"skipped": True, "reason": "duplicate action"}), "args": args}
            state["tool_calls_made"].append({"tool": tool, "args": args, "status": "SKIPPED: duplicate action"})
            return
        executed_actions.add(action_fingerprint)
        state["_executed_actions"] = executed_actions

        # Leave write gates (confirmation, verification, balance, hd_id
        # chaining) live in the action-strategy layer per the architecture.
        from agent.actions.strategies import leave_gates

        if leave_gates.is_leave_write(entity):
            if not leave_gates.resolve_hd_id(entity, record, args, state, call_id, tool):
                return
            if not await leave_gates.confirmation_gate(
                entity, record, args, state, call_id, tool,
                auth_context, tenant_id, step,
            ):
                return
            if not await leave_gates.employee_verify_gate(
                entity, record, args, state, call_id, tool, client,
            ):
                return
            if not await leave_gates.balance_gate(
                entity, record, args, state, call_id, tool, client,
            ):
                return

        result = await invoke_dab_tool_with_retry(client, "create_record", {"entity": entity, "data": record})
        dab_data = extract_payload(result)

        if isinstance(dab_data, dict) and dab_data.get("isError"):
            error_msg = dab_data.get("message", "Unknown DAB error")
            logger.error("DAB create_record returned error: %s", error_msg)
            _record_dab_error(state, call_id, tool, args, error_msg, entity=entity)
            return

        # After a successful employee_leave_hd create, capture its id so the
        # next employee_leave step can use it as hd_id.
        await leave_gates.capture_leave_hd_id(
            entity, record, state, client, dab_data,
            invoke_read_records=lambda qargs: invoke_dab_tool_with_retry(client, "read_records", qargs),
        )

        result_text = json.dumps(dab_data, default=str) if isinstance(dab_data, dict) else str(dab_data)
        _record_success(state, call_id, tool, args, result_text)
        logger.info("DAB create_record succeeded (call_id=%s)", call_id)

    except Exception as exc:
        logger.error("DAB create_record failed: %s", exc)
        _record_system_error(state, call_id, tool, args, exc)


async def _execute_dab_update_call(step: Dict, state: Dict, auth_context: Any, tenant_id: str = "LOCALDEV", cached_schema: Dict = None, cached_tools: List[Dict] = None):
    tool = step.get("tool")
    args = step.get("args", {})
    call_id = f"{tool}_{len(state['tool_calls_made'])}"
    client = dab_manager.get_client(tenant_id, getattr(auth_context, "token", None), _resolve_dab_role(auth_context))

    normalize_odata_args(args, cached_schema)

    logger.debug("Executing DAB tool: %s with args: %s", tool, args)

    args, error = enforce_tool_args(tool, args, auth_context)
    if error:
        _record_blocked(state, call_id, tool, args, error, "AuthError")
        return

    args, schema_error = validate_dab_args(tool, args, cached_tools)
    if schema_error:
        _record_blocked(state, call_id, tool, args, schema_error, "ValidationError")
        return

    try:
        keys = args.get("keys", {})
        fields = args.get("fields", {})
        entity = args.get("entity", "")
        if not isinstance(keys, dict):
            raise ValueError("update_record 'keys' argument must be an object")
        if not isinstance(fields, dict):
            raise ValueError("update_record 'fields' argument must be an object")

        # Profile-update confirmation gate: when the injector marked this step
        # with _confirmation_pending (implicit life-event statement), block the
        # write unless the confirmation turn set _confirmed_update_write.
        if step.get("_confirmation_pending") and not state.get("_confirmed_update_write"):
            logger.info(
                "UPDATE_CONFIRMATION_GATE: blocking update_record for %s — awaiting user confirmation",
                entity,
            )
            state["_awaiting_confirmation"] = True
            state["tool_results"][call_id] = {
                "result": json.dumps({
                    "confirmation_required": True,
                    "confirmation_type": "profile_update",
                    "entity": entity,
                    "summary": step.get("_confirmation_summary", ""),
                    "message": "Please confirm this profile update before it is applied.",
                }),
                "args": args,
            }
            state["tool_calls_made"].append({
                "tool": tool, "args": args,
                "status": "CONFIRMATION_REQUIRED",
            })
            from agent.data.metrics import record_dab_tool_call
            record_dab_tool_call(tool, "CONFIRMATION_REQUIRED")
            return

        result = await invoke_dab_tool_with_retry(client, "update_record", {"entity": entity, "keys": keys, "fields": fields})
        dab_data = extract_payload(result)

        if isinstance(dab_data, dict) and dab_data.get("isError"):
            error_msg = dab_data.get("message", "Unknown DAB error")
            logger.error("DAB update_record returned error: %s", error_msg)
            _record_dab_error(state, call_id, tool, args, error_msg, entity=entity)
            return

        result_text = json.dumps(dab_data, default=str) if isinstance(dab_data, dict) else str(dab_data)
        _record_success(state, call_id, tool, args, result_text)
        logger.info("DAB update_record succeeded (call_id=%s)", call_id)

    except Exception as exc:
        logger.error("DAB update_record failed: %s", exc)
        _record_system_error(state, call_id, tool, args, exc)


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
# HANA TOOL EXECUTION
# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

async def _try_numeric_cast_retry(tool: str, args: Dict[str, Any], state: Dict, call_id: str, error_text: str, client: Any) -> bool:
    """Retry hana_execute_query with CAST(... AS DECIMAL) after a numeric-type error.

    Returns True if retry succeeded, False otherwise.
    """
    if not _looks_like_numeric_type_error(error_text):
        return False
    if tool != "hana_execute_query":
        return False
    if not isinstance(args, dict) or not isinstance(args.get("query"), str):
        return False

    sanitized_query = _wrap_aggregate_fields_with_cast(args["query"])
    if sanitized_query == args["query"]:
        return False

    retry_args = dict(args)
    retry_args["query"] = sanitized_query
    try:
        raw = client.call_tool(tool, retry_args)
        normalized = normalize_hana_result(raw, tool)
        if isinstance(normalized, dict) and "result" in normalized and normalized.get("result"):
            state["tool_results"][call_id] = normalized
            state["tool_calls_made"].append({"tool": tool, "args": retry_args, "status": "SUCCESS_AFTER_CAST"})
            logger.info("HANA tool %s succeeded after numeric cast retry (call_id=%s)", tool, call_id)
            return True
    except Exception as retry_e:
        logger.error("HANA tool %s retry after cast failed: %s", tool, retry_e)
    return False


async def _execute_hana_tool_call(step: Dict, state: Dict, auth_context: Any, tenant_id: str = "LOCALDEV", cached_schema: Dict = None, cached_tools: List[Dict] = None):
    tool = step.get("tool", "")
    args = step.get("args", {})
    call_id = f"{tool}_{len(state['tool_calls_made'])}"

    client = hana_manager.get_client(tenant_id)
    registry = schema_registry_service.get_registry(tenant_id)

    if tool == "hana_list_tables" and not args.get("schema_name"):
        logger.warning("hana_list_tables called without schema_name -- blocked")
        state["tool_results"][call_id] = {"error": "schema_name is required for hana_list_tables. Use a schema from the authorized list above."}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": "ERROR: missing schema_name"})
        return

    # -- Registry diagnostics --------------------------------------------------
    registry_size = len(registry) if registry else 0
    registry_tables = 0
    if registry:
        for s, tables in registry.items():
            registry_tables += len(tables)
    logger.info(
        "HANA_REGISTRY: schemas=%d total_tables=%d for tenant=%s",
        registry_size, registry_tables, tenant_id,
    )
    if registry and args.get("schema_name"):
        schema_key = args["schema_name"].upper()
        tables_in_schema = registry.get(schema_key, registry.get(args["schema_name"], []))
        logger.info(
            "HANA_REGISTRY_LOOKUP: schema=%s tables=%s",
            args["schema_name"], tables_in_schema,
        )

    validation_error = validate_hana_tool_args(tool, args, registry)
    if validation_error:
        logger.warning("HANA tool %s blocked by registry validation: %s", tool, validation_error)
        state["tool_results"][call_id] = {"error": validation_error}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {validation_error}"})
        return

    logger.info(
        "HANA_EXECUTE: tool=%s schema=%s table=%s",
        tool, args.get("schema_name"), args.get("table_name"),
    )

    try:
        raw = client.call_tool(tool, args)
        normalized = normalize_hana_result(raw, tool)

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
            logger.error("HANA tool %s error: %s | raw=%s", tool, error_msg, {k: v for k, v in normalized.items() if k != "result"})
            if not await _try_numeric_cast_retry(tool, args, state, call_id, error_msg, client):
                pass  # Error already recorded above
        else:
            state["tool_results"][call_id] = normalized
            state["tool_calls_made"].append({"tool": tool, "args": args, "status": "SUCCESS"})
            logger.info("HANA tool %s succeeded (call_id=%s)", tool, call_id)
    except Exception as e:
        error_text = str(e)
        logger.error("HANA tool %s failed: %s", tool, error_text)
        state["tool_results"][call_id] = {"error": error_text}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {error_text}"})
        await _try_numeric_cast_retry(tool, args, state, call_id, error_text, client)




