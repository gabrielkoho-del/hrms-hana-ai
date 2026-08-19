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
from typing import Any, Dict, List, Optional

from agent.core.guards import (
    is_greeting_or_smalltalk,
    greeting_response,
    is_vague_data_question,
    data_summary_response,
)
from agent.core.session_state import load_session_state, save_session_state
from agent.auth.role_resolver import resolve_auth_role, is_data_about_user
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
from agent.output.excel_exporter import export_to_excel, export_to_excel_with_chart
from agent.integrations.dab_client import dab_manager, invoke_dab_tool_with_retry
from agent.data.response import extract_items, extract_payload, is_dab_error
from agent.dab.validation import validate_dab_args
from agent.data.metrics import (
    record_dab_tool_call, record_dab_error,
    record_chart_generated
)
from agent.dab.odata_normalizer import normalize_odata_args
from agent.integrations.hana_client import hana_manager, normalize_hana_result, extract_hana_items, validate_hana_tool_args, _looks_like_numeric_type_error, _wrap_aggregate_fields_with_cast
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


# -"EUR-"EUR-"EUR Tool execution registry -"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR
# Maps tool names to executor functions. Idiomatic replacement for string-matching in main loop.
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
    if tool_name.startswith("superset_"):
        return _execute_superset_tool_call
    return None


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


async def run_reflexive_agent(
    user_query: str,
    conversation_history: str,
    auth_context: Any,
    cached_tools: List[Dict],
    cached_schema: Dict,
) -> str:
    is_gs, gs_label = is_greeting_or_smalltalk(user_query)
    if is_gs:
        logger.info("L1 guard triggered: %s", gs_label)
        return greeting_response(gs_label)

    if is_vague_data_question(user_query):
        logger.info("L2 guard triggered: vague data question")
        return data_summary_response(cached_schema)

    session_state = load_session_state(auth_context)

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

    tenant_id = auth_context.tenant_id if auth_context and auth_context.tenant_id else "LOCALDEV"

    logger.info("Planning with tool_planner for: %s", user_query)
    plan = await build_tool_plan(
        user_query=user_query, conversation_history=conversation_history,
        auth_context=auth_context, cached_tools=cached_tools,
        cached_schema=cached_schema, rag_available=True, tone_context=tone_context,
        hana_schema_registry=schema_registry_service.get_registry(tenant_id),
    )

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

    if plan.get("steps"):
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # FORECASTING PIPELINE -- handle forecasting queries end-to-end
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        is_forecasting = any(
            step.get("tool") in ("fetch_external_data", "generate_code", "execute_sandbox", "summarize_forecast")
            for step in plan.get("steps", [])
        )
        if is_forecasting:
            logger.info("FORECASTING_PIPELINE: executing end-to-end forecast")
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
                client = dab_manager.get_client(tenant_id)

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

            # Skip normal step loop; proceed to summarization
            state = forecast_state
            chart_config = None
            chartable_data = []
            code_context_md = ""
            binned_output_column = None
            variance_text = ""
            tool_results_for_summarizer = {}
            for call_id, result in state["tool_results"].items():
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
                tool_results_for_summarizer["__forecast"] = state["tool_results"]["__forecast"]
            if "__market_context" in tool_results_for_summarizer:
                tool_results_for_summarizer["__market_context"] = state["tool_results"]["__market_context"]

            action_context = plan.get("action_context", "")

            # Call summarizer with market context
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
                "last_forecast_result": forecast_output.raw if forecast_output else session_state.get("last_forecast_result"),
            })
            return answer

        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        # SUPERSET DASHBOARD PIPELINE -- handle dashboard requests end-to-end
        # -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
        is_dashboard = (
            tone_context.get("intent_category") == "dashboard_building"
            or any(
                step.get("tool", "") in ("superset_generate_dashboard", "superset_generate_chart")
                for step in plan.get("steps", [])
            )
        )
        if is_dashboard:
            logger.info("SUPERSET_DASHBOARD_PIPELINE: executing dashboard workflow")
            from agent.output.superset_dashboard import SupersetDashboardAgent

            dashboard_agent = SupersetDashboardAgent()
            dashboard_result = await dashboard_agent.handle(
                user_query=user_query,
                auth_context=auth_context,
                tone_context=tone_context,
            )

            # Persist dashboard result in state for summarizer
            state = {
                "tool_results": {},
                "tool_calls_made": [],
                "rag_context": "",
            }
            state["tool_results"]["__superset_dashboard"] = {
                "result": json.dumps(dashboard_result, default=str),
                "args": {"type": "superset_dashboard"},
            }
            state["tool_calls_made"].append({
                "tool": "superset_dashboard_agent",
                "args": {"user_query": user_query},
                "status": "SUCCESS" if dashboard_result.get("status") != "error" else f"ERROR: {dashboard_result.get('error')}",
            })

            # Skip normal step loop; proceed to summarization
            chart_config = None
            chartable_data = []
            code_context_md = ""
            binned_output_column = None
            variance_text = ""
            tool_results_for_summarizer = {}
            for call_id, result in state["tool_results"].items():
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

            if "__superset_dashboard" in tool_results_for_summarizer:
                tool_results_for_summarizer["__superset_dashboard"] = state["tool_results"]["__superset_dashboard"]

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

            save_session_state(auth_context, {
                "last_query": user_query,
                "last_intent": intent_result.intent,
                "last_intent_category": intent_result.intent_category,
                "last_data_scope": intent_result.data_scope,
                "last_chart_data": chartable_data if chartable_data else session_state.get("last_chart_data"),
                "last_chart_config": chart_config if chart_config else session_state.get("last_chart_config"),
                "last_tool_results": state.get("tool_results") or session_state.get("last_tool_results", {}),
                "last_export_url": state.get("export_url", "") or session_state.get("last_export_url", ""),
                # Preserve pending dashboard previews across the confirmation turn
                "superset_pending_dashboard": session_state.get("superset_pending_dashboard"),
            })
            return answer

        # Normal non-forecasting step loop
        state = {
            "tool_results": {},
            "tool_calls_made": [],
            "rag_context": "",
        }

        for step in plan["steps"]:
            tool = step.get("tool", "")
            executor = _get_executor(tool)
            if executor is None:
                logger.warning("No executor registered for tool '%s' -- skipping", tool)
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

    binned_output_column = None
    bin_config = plan.get("client_side_binning")
    if bin_config and isinstance(bin_config, dict):
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

    if plan.get("rag"):
        state["rag_context"] = retrieve_policy_context(plan.get("rag_query", user_query))

    if not state["tool_results"] and not (intent_result.wants_export and session_state.get("last_chart_data")):
        return "I wasn't able to retrieve any data for that request."

    code_context_md = ""
    tenant_id_for_codes = auth_context.tenant_id if auth_context and auth_context.tenant_id else "default"
    try:
        from agent.dab.code_resolver import scan_for_codes
        code_context_md = await scan_for_codes(state["tool_results"], tenant_id_for_codes)
        if code_context_md:
            logger.info("CODE_RESOLVER: injected code context (%d chars) into LLM", len(code_context_md))
    except Exception as e:
        logger.warning("CODE_RESOLVER: failed to scan results: %s", e)

    chart_markdown = ""
    if 'chartable_data' not in locals() or not chartable_data:
        chartable_data = []

    cached_cfg = session_state.get("last_chart_config")
    if intent_result.wants_export and cached_cfg:
        chart_config = dict(cached_cfg)
        logger.info(
            "FOLLOW_UP_EXPORT: restored chart_config from session (type=%s), overriding planner default",
            chart_config.get("type")
        )
        # Only restore cached data when there are no fresh tool results.
        # If new tool calls were made, the extraction loop below will populate chartable_data.
        if not state.get("tool_calls_made") and not chartable_data and session_state.get("last_chart_data"):
            chartable_data = list(session_state["last_chart_data"])
            logger.info("FOLLOW_UP_EXPORT: restored %d cached rows", len(chartable_data))
    elif 'chart_config' not in locals() or not chart_config:
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

    if chart_config and isinstance(chart_config, dict) and chartable_data:
        # If planner left chart metadata empty (common for HANA raw-SQL paths),
        # derive x/y/title from the actual result columns and the user query.
        if not chart_config.get("y_column") and chartable_data:
            first_row = chartable_data[0]
            cols = list(first_row.keys())
            q = user_query.lower()
            # Match query keywords against available columns to pick the intended metric.
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
                # Fallback: first numeric-looking column that is not the x-axis/time column
                import pandas as pd
                df = pd.DataFrame(chartable_data)
                numeric_cols = [
                    c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])
                ]
                if numeric_cols:
                    chart_config["y_column"] = numeric_cols[0]
                    logger.info("CHART_DEBUG_EXECUTOR: fallback y_column='%s' from numeric cols", numeric_cols[0])

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

    if not chart_config:
        logger.info("CHART_DEBUG_EXECUTOR: no chart_config in plan")


    if (plan.get("needs_export") or intent_result.wants_export) and chart_config and isinstance(chart_config, dict) and chartable_data:
        metadata = build_export_metadata(chart_config, auth_context, auth_role, user_query, intent_result, state.get("tool_calls_made", []))
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

    # Inject variance analysis for multi-metric financial reports
    variance_text = ""
    if chartable_data and isinstance(chart_config, dict) and chart_config.get("multi_series"):
        metrics = chart_config.get("metrics", []) or _detect_wide_format_metrics(chartable_data, None)
        x_col = next(iter(chartable_data[0].keys())) if chartable_data else "period"
        # Find x column
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

async def _execute_dab_tool_call(step: Dict, state: Dict, auth_context: Any, tenant_id: str = "LOCALDEV", cached_schema: Dict = None, cached_tools: List[Dict] = None):
    from agent.main import enforce_tool_args, filter_tool_results

    tool = step.get("tool")
    args = step.get("args", {})
    call_id = f"{tool}_{len(state['tool_calls_made'])}"
    client = dab_manager.get_client(tenant_id)

    normalize_odata_args(args, cached_schema)

    logger.debug("Executing DAB tool: %s with args: %s", tool, args)

    args, error = enforce_tool_args(tool, args, auth_context)
    if error:
        state["tool_results"][call_id] = {"error": error}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"BLOCKED: {error}"})
        record_dab_tool_call(tool, "BLOCKED")
        record_dab_error("AuthError", entity=args.get("entity"))
        return

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

        if args.get("groupby"):
            if "first" in args:
                logger.info("Stripping 'first'=%s from aggregate_records with groupby for complete chart data", args.get("first"))
                args.pop("first", None)
        else:
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


# -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
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

    # -"EUR-"EUR Registry diagnostics -"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR-"EUR
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


async def _execute_superset_tool_call(step: Dict, state: Dict, auth_context: Any, tenant_id: str = "LOCALDEV", cached_schema: Dict = None, cached_tools: List[Dict] = None):
    """Execute a Superset MCP tool call.

    For non-dashboard tools (list/get/info/sql), calls MCP directly.
    For chart/dashboard tools, calls MCP directly to avoid duplicating
    the top-level dashboard pipeline's preview-first workflow.
    """
    from agent.integrations.superset_client import SupersetMCPClient

    tool = step.get("tool", "")
    args = step.get("args", {})
    call_id = f"{tool}_{len(state['tool_calls_made'])}"

    try:
        client = SupersetMCPClient()
        result = client.call_tool(tool.replace("superset_", ""), args)
        state["tool_results"][call_id] = {"result": result, "args": args}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": "SUCCESS"})
        logger.info("SUPERSET tool %s succeeded (call_id=%s)", tool, call_id)

    except Exception as exc:
        error_text = str(exc)
        logger.error("Superset tool %s failed: %s", tool, error_text)
        state["tool_results"][call_id] = {"error": error_text}
        state["tool_calls_made"].append({"tool": tool, "args": args, "status": f"ERROR: {error_text}"})

