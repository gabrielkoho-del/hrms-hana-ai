"""Chart, RAG, export, and binning metadata inference.

Infer chart, RAG, client-side binning, and other metadata from query and steps.
"""
import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent.core.config_loader import (
    get_chart_type_patterns_for_tenant as get_chart_type_patterns,
    get_export_keywords_for_tenant as get_export_keywords,
    get_general_chart_pattern_for_tenant as get_general_chart_pattern,
    get_rag_keywords_for_tenant as get_rag_keywords,
)
from agent.core.intent_classifier import ChartIntentResult
from agent.output.binning import BINNING_MAP, _resolve_binning_column
from agent.output.chart_metadata import extract_metrics_hybrid
from agent.output.dashboard_builder import build_dashboard_queries
from agent.hana.temporal import is_financial_metric_query, get_available_fiscal_years, resolve_temporal_context
from agent.integrations.hana_client import is_hana_available
from agent.integrations.llm_client import call_llm

logger = logging.getLogger("hr_agent")

# Confidence threshold below which we ask the user for clarification.
_CHART_INTENT_CONFIDENCE_THRESHOLD = 0.65


# =============================================================================
# STRUCTURED CHART INTENT CLASSIFICATION -- LLM-based, replaces pure-regex detection
# =============================================================================


async def classify_chart_intent(
    user_query: str,
    conversation_history: str = "",
    tone_context: Optional[Dict[str, Any]] = None,
    tenant_id: str = "default",
) -> ChartIntentResult:
    """Classify the user's chart/dashboard intent using a structured LLM call.

    This replaces pure-regex multi-chart detection with a proper LLM classifier that can
    handle ambiguous cases like "show me performance" (one chart or several?).

    Returns a ChartIntentResult with:
      - intent: "single_chart" | "multi_chart" | "modify_existing" | "kpi_only"
      - confidence: 0.0-1.0
      - charts: list of detected {dimension, metric, chart_type} pairs
      - needs_clarification: True when confidence is low or query is ambiguous
      - clarification_reason: human-readable explanation
      - raw_classification: full LLM output for debugging

    Falls back to feature-shedding (kpi_only, no clarification) if the LLM call fails.
    """
    tone_context = tone_context or {}
    query_lower = user_query.lower().strip()

    system_prompt = """You are a chart and dashboard intent classifier for an HR analytics AI assistant.

Classify the user query into ONE of these intents:

- single_chart: User wants exactly ONE chart (e.g., "show me headcount by department", "plot salary distribution", "bar chart of leave by type").
- multi_chart: User explicitly wants MULTIPLE charts in a dashboard (e.g., "show me an overview", "give me a dashboard", "compare headcount AND salary across departments", "show me X and Y and Z").
- modify_existing: User wants to add/remove/change charts on an existing dashboard they are already viewing.
- kpi_only: User only wants KPI summary numbers, no charts.

RULES:
- "dashboard", "overview", "all of the above", "everything", "full picture" → multi_chart
- "show me X and Y", "compare A and B" → multi_chart (each X/Y/A/B is a separate chart)
- "show me an overview of everything" → multi_chart
- "give me a breakdown" → multi_chart
- Named dashboards (executive workforce, headcount, compensation) → multi_chart
- Single metric/dimension → single_chart
- "just the numbers", "no chart", "KPI summary" → kpi_only
- If user mentions modifying an existing view → modify_existing
- If ambiguous ("how's the org doing?"), default to multi_chart with low confidence and set needs_clarification=true
- Action-oriented queries (apply, request, book, take, submit leave; update profile; resign; enroll in benefits) are NOT chart requests. Classify as kpi_only with high confidence (0.9) and needs_clarification=false — the response is text-only and the action pipeline handles it. NEVER set needs_clarification=true for action requests.
- If the query is a question about data/numbers but not explicitly asking for a chart (e.g., "how many employees", "what is my balance"), classify as kpi_only with moderate confidence.

For multi_chart, identify each distinct chart as a {dimension, metric, chart_type} object.
For single_chart, return one chart.
For kpi_only, return an empty charts array.

Return JSON only with this schema:
{
  "intent": "single_chart|multi_chart|modify_existing|kpi_only",
  "confidence": 0.0-1.0,
  "charts": [
     {"dimension": "field name or null", "metric": "field name or null", "chart_type": "bar|line|pie|area|doughnut|scatter|heatmap|radar|null"}
  ],
  "needs_clarification": true|false,
  "clarification_reason": "string or null"
}

Examples:
- "show me headcount by department" → {"intent": "single_chart", "confidence": 0.95, "charts": [{"dimension": "department", "metric": "headcount", "chart_type": "bar"}], "needs_clarification": false, "clarification_reason": null}
- "show me executive workforce overview" → {"intent": "multi_chart", "confidence": 0.95, "charts": [...], "needs_clarification": false, "clarification_reason": null}
- "how's the org doing?" → {"intent": "multi_chart", "confidence": 0.45, "charts": [...], "needs_clarification": true, "clarification_reason": "Ambiguous query could mean different breakdowns"}
- "just the numbers" → {"intent": "kpi_only", "confidence": 0.9, "charts": [], "needs_clarification": false, "clarification_reason": null}
- "want to apply full day annual leave on Sep 30" → {"intent": "kpi_only", "confidence": 0.9, "charts": [], "needs_clarification": false, "clarification_reason": null}"""

    history_snippet = conversation_history[-300:] if conversation_history else "None"
    user_prompt = (
        f"Recent conversation:\n{history_snippet}\n\n"
        f"Query: {user_query}\n\n"
        f"Return JSON only."
    )

    try:
        choice = await asyncio.wait_for(
            asyncio.to_thread(call_llm, system_prompt, user_prompt, temperature=0.1, max_tokens=300),
            timeout=15.0,
        )
        if not choice:
            raise ValueError("Empty LLM response")

        content = choice.get("message", {}).get("content", "{}").strip()
        # Strip markdown code fences
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        raw = json.loads(content)

        intent = str(raw.get("intent", "single_chart"))
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
        charts = raw.get("charts", [])
        needs_clarification = bool(raw.get("needs_clarification", False))
        clarification_reason = str(raw.get("clarification_reason") or "")

        # Validate intent value
        valid_intents = {"single_chart", "multi_chart", "modify_existing", "kpi_only"}
        if intent not in valid_intents:
            logger.warning("classify_chart_intent: invalid intent '%s', defaulting to single_chart", intent)
            intent = "single_chart"

        # Confidence threshold gate: if the LLM's confidence is below the
        # threshold, force needs_clarification=True so the planner asks the
        # user before proceeding with a potentially wrong chart config.
        if confidence < _CHART_INTENT_CONFIDENCE_THRESHOLD and not needs_clarification:
            needs_clarification = True
            if not clarification_reason:
                clarification_reason = (
                    f"Low confidence ({confidence:.2f}) in chart intent classification; "
                    f"clarification recommended."
                )
            logger.info(
                "classify_chart_intent: confidence %.2f below threshold %.2f -- "
                "forcing needs_clarification=True",
                confidence, _CHART_INTENT_CONFIDENCE_THRESHOLD,
            )

        logger.info(
            "classify_chart_intent: intent=%s confidence=%.2f charts=%d needs_clarification=%s",
            intent, confidence, len(charts), needs_clarification,
        )

        return ChartIntentResult(
            intent=intent,
            confidence=confidence,
            charts=charts,
            needs_clarification=needs_clarification,
            clarification_reason=clarification_reason,
            raw_classification=raw,
        )

    except Exception as exc:
        logger.warning("classify_chart_intent: LLM call failed (%s), falling back to feature-shedding", exc)
        return _classify_chart_intent_fallback(query_lower, tone_context)


def _classify_chart_intent_fallback(
    query_lower: str,
    tone_context: Optional[Dict[str, Any]] = None,
) -> ChartIntentResult:
    """Fallback when the LLM chart-intent classifier is unavailable.

    Implements pure feature-shedding per resilient-agent best practice
    (LangChain RunnableWithFallbacks, LangGraph reliability handbook,
    AWS resilient-agents patterns): chart classification is a decorative
    capability, so on LLM failure we decline to classify and return
    kpi_only with needs_clarification=False.

    This eliminates behavioral divergence between the LLM path and the
    fallback path — the previous regex fallback guessed multi_chart /
    single_chart / kpi_only via keyword matching, which could disagree
    with the LLM's judgment and trigger the executor's clarification
    gate (needs_clarification=True), blocking core action pipelines
    such as leave applications.

    Chart requests still work because infer_metadata() has independent
    deterministic detection (explicit_chart_type regex, general_chart_intent,
    chart_eligible from intent category) that gracefully degrades a
    dashboard request to a single chart instead of breaking it.
    """
    tone_context = tone_context or {}

    # Action requests (apply leave, update profile, etc.) are NOT chart
    # requests. Classify as kpi_only (text-only response) with no
    # clarification — needs_clarification=True would trigger the executor's
    # clarification gate and block the action pipeline from executing.
    try:
        from agent.actions.base_injector import is_action_request
        if is_action_request(query_lower):
            return ChartIntentResult(
                intent="kpi_only",
                confidence=0.9,
                charts=[],
                needs_clarification=False,
                clarification_reason="",
                raw_classification={"source": "regex_fallback_action"},
            )
    except Exception as exc:
        logger.warning("_classify_chart_intent_fallback: is_action_request check failed (%s)", exc)

    # Pure feature-shedding: decline to classify chart intent. The LLM
    # classifier is the primary path; on failure we shed the decorative
    # chart-classification capability rather than guessing, which could
    # diverge from the LLM's judgment and block downstream pipelines.
    return ChartIntentResult(
        intent="kpi_only",
        confidence=0.5,
        charts=[],
        needs_clarification=False,
        clarification_reason="",
        raw_classification={"source": "regex_fallback_feature_shedding"},
    )





def _detect_chart_intent(query: str, tenant_id: str = "default") -> Dict[str, Optional[str]]:
    """Detect explicit chart type from user query using configured patterns.

    Returns a dict with:
      - explicit_chart_type: matched chart type or None
      - general_chart_intent: bool from general chart pattern
    """
    query_lower = query.lower()
    chart_type_patterns = get_chart_type_patterns(tenant_id)
    explicit_chart_type = None
    for pattern_info in chart_type_patterns:
        if re.search(pattern_info["pattern"], query_lower, re.I):
            explicit_chart_type = pattern_info["type"]
            break

    general_chart_pattern = get_general_chart_pattern(tenant_id)
    general_chart_intent = bool(re.search(general_chart_pattern, query_lower, re.I))
    return {
        "explicit_chart_type": explicit_chart_type,
        "general_chart_intent": general_chart_intent,
    }


def _validate_chart_metadata(chart: Optional[Dict]) -> Optional[Dict]:
    """Validate chart metadata and return a safe version or None.

    A chart config is considered valid only when it has a recognized type and
    at least one of x_column / y_column is non-empty. Invalid configs are
    replaced with None so the executor can fall back to auto-detection or skip
    chart generation gracefully.
    """
    if not chart or not isinstance(chart, dict):
        return None

    valid_types = {"bar", "barh", "pie", "doughnut", "line", "area", "hist", "box", "scatter", "heatmap", "gauge", "radar"}
    chart_type = chart.get("type")
    if chart_type not in valid_types:
        logger.warning("CHART_VALIDATION: invalid chart type '%s' -- dropping chart metadata", chart_type)
        return None

    x_col = (chart.get("x_column") or "").strip()
    y_col = (chart.get("y_column") or "").strip()

    if not x_col and not y_col:
        logger.warning("CHART_VALIDATION: chart metadata has no x_column or y_column -- dropping")
        return None

    # Ensure required keys exist with safe defaults
    return {
        "type": chart_type,
        "x_column": x_col,
        "y_column": y_col,
        "y_label": chart.get("y_label") or "",
        "title": chart.get("title") or "",
        "gauge_min": chart.get("gauge_min"),
        "gauge_max": chart.get("gauge_max"),
        "gauge_threshold": chart.get("gauge_threshold"),
        "trend_line": chart.get("trend_line", False),
        "multi_series": chart.get("multi_series", False),
        "metrics": chart.get("metrics", []),
    }


def _derive_y_label(func: str, entity: str, field: str) -> str:
    """Derive a human-readable y-axis label from aggregation function and field."""
    func_labels = {
        "count": "Count",
        "avg": "Average",
        "sum": "Total",
        "min": "Minimum",
        "max": "Maximum",
    }
    base = func_labels.get(func, func.title())
    field_label = field.replace("_", " ").title() if field else "Value"
    return f"{base} {field_label}"


async def infer_metadata(user_query: str, steps: List[Dict], tone_context: Optional[Dict], tenant_id: str = "default") -> Dict:
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

    has_aggregate = any(s.get("tool") == "aggregate_records" for s in steps)
    has_read = any(s.get("tool") == "read_records" for s in steps)
    has_hana = any(s.get("tool", "").startswith("hana_") for s in steps)

    # Detect explicit chart type request from user query (must happen before HANA early return)
    chart_type_patterns = get_chart_type_patterns(tenant_id)
    explicit_chart_type = None
    for pattern_info in chart_type_patterns:
        if re.search(pattern_info["pattern"], query_lower, re.I):
            explicit_chart_type = pattern_info["type"]
            break

    general_chart_pattern = get_general_chart_pattern(tenant_id)
    general_chart_intent = bool(re.search(general_chart_pattern, query_lower, re.I))

    if explicit_chart_type:
        logger.info("Planner: explicit chart type detected from query: %s", explicit_chart_type)

    # Also treat tone_context chart_eligible=True as implicit chart intent.
    chart_eligible = tone_context.get("chart_eligible", False)
    if not general_chart_intent and chart_eligible and has_hana:
        general_chart_intent = True

    # Detect multi-metric financial report intent using hybrid extraction.
    unique_metrics = await extract_metrics_hybrid(user_query)
    is_multi_metric = len(unique_metrics) >= 2
    if is_multi_metric:
        logger.info("Planner: multi-metric report detected: %s", unique_metrics)

    # multi_chart is always set by the LLM-based classify_chart_intent call in build_tool_plan
    # before infer_metadata runs. No regex fallback is needed.
    multi_chart = False
    chart_intent = tone_context.get("chart_intent")
    if chart_intent is not None:
        # Use structured LLM classification result
        multi_chart = chart_intent.intent == "multi_chart"
        kpi_only = chart_intent.intent == "kpi_only"
        if multi_chart:
            logger.info(
                "Planner: multi-chart intent from LLM classifier (confidence=%.2f, needs_clarification=%s)",
                chart_intent.confidence, chart_intent.needs_clarification,
            )
            metadata["multi_chart"] = True
            metadata["reasoning"] = (
                f"Multi-chart dashboard intent from LLM classifier (confidence={chart_intent.confidence:.2f}); "
                "using dynamic dashboard with multiple queries."
            )
            if chart_intent.needs_clarification:
                metadata["chart_intent_clarification_needed"] = True
                metadata["chart_intent_clarification_reason"] = chart_intent.clarification_reason
            # Generate structured dashboard queries from the user's request
            from agent.output.dashboard_builder import build_dashboard_queries
            metadata["dashboard_queries"] = build_dashboard_queries(user_query, steps, tenant_id)
        elif kpi_only:
            logger.info("Planner: kpi_only intent from LLM classifier")
            metadata["kpi_only"] = True
            metadata["reasoning"] = "KPI-only intent from LLM classifier."
        elif chart_intent.needs_clarification:
            # single_chart but needs clarification
            logger.info(
                "Planner: single chart with clarification needed (confidence=%.2f): %s",
                chart_intent.confidence, chart_intent.clarification_reason,
            )
            metadata["chart_intent_clarification_needed"] = True
            metadata["chart_intent_clarification_reason"] = chart_intent.clarification_reason

    # HANA queries use raw SQL; skip DAB chart/binning metadata inference
    # BUT preserve explicit chart requests so executor can still generate the chart
    if has_hana and not has_aggregate and not has_read:
        metadata["reasoning"] = f"Selected {len(steps)} HANA tool(s) for direct SQL query."
        if explicit_chart_type or general_chart_intent or multi_chart:
            chart_type = explicit_chart_type or "bar"
            # Do NOT create chart metadata with empty required fields.
            # The executor will auto-detect x/y columns from the returned data.
            metadata["chart"] = None
            metadata["reasoning"] += f" Chart intent detected (type={chart_type}); executor will derive columns from data."
        return metadata

    # -- Chart inference (intent category + data shape, NOT keywords) --
    chart_eligible = tone_context.get("chart_eligible", False) if tone_context else False

    # Must also have aggregate step with groupby OR read_records with 2+ columns OR HANA data
    has_aggregate_with_groupby = any(
        s.get("tool") == "aggregate_records" and s.get("args", {}).get("groupby")
        for s in steps
    )
    has_read_with_2col = any(
        s.get("tool") == "read_records" and
        len([f for f in (s.get("args", {}).get("select", "") or "").split(",") if f.strip()]) >= 2
        for s in steps
    )
    has_hana_with_data = has_hana and any(
        s.get("tool", "").startswith("hana_") for s in steps
    )
    has_groupby_like = has_aggregate_with_groupby or has_read_with_2col or has_hana_with_data

    # Row count guard: 1-5 rows = no chart (too small), 6-50 = chart, 50+ = chart + export
    # We don't know row count yet, so we plan the chart and let executor decide later
    wants_chart = (chart_eligible or multi_chart or explicit_chart_type or general_chart_intent) and has_groupby_like

    if wants_chart:
        chart_type = "bar"
        x_col = ""
        y_col = ""
        title = "Distribution"
        y_label = ""
        multi_series = False
        metrics = []

        for step in steps:
            if step["tool"] == "aggregate_records":
                args = step.get("args", {})
                groupby = args.get("groupby", [])
                func = args.get("function", "count")
                field = args.get("field", "")

                # Heatmap special handling: needs 2 groupby dimensions.
                # Use the first two groupby fields as row/column dimensions;
                # the value column is auto-detected by the renderer.
                if explicit_chart_type == "heatmap" and len(groupby) >= 2:
                    chart_type = "heatmap"
                    x_col = groupby[0]
                    y_col = groupby[1]
                    y_label = _derive_y_label(func, args.get("entity", ""), field)
                    title = f"{x_col.replace('_', ' ').title()} by {y_col.replace('_', ' ').title()}"
                    break

                # If user asked for heatmap but we don't have 2 groupby dims,
                # fall back to a standard chart type instead of producing an
                # unusable heatmap config.
                if explicit_chart_type == "heatmap" and len(groupby) < 2:
                    chart_type = "bar"
                    if groupby:
                        x_col = groupby[-1]
                    y_col = "count" if func == "count" else (field or "value")
                    y_label = _derive_y_label(func, args.get("entity", ""), field)
                    title = f"{x_col.replace('_', ' ').title()} Distribution" if x_col else "Distribution"
                    break

                if groupby:
                    x_col = groupby[-1]  # Most granular

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
                    multi_series = True

                # Add filter context to title
                having = args.get("having", "")
                filter_str = args.get("filter", "")
                context_suffix = ""
                if having:
                    # having may be an object ({'gt': 100}) or a legacy string
                    having_str = having if isinstance(having, str) else json.dumps(having)
                    context_suffix = f" (filtered: {having_str})"
                elif filter_str:
                    context_suffix = " (filtered)"
                title = f"{x_col.replace('_', ' ').title()} Distribution{context_suffix}" if x_col else "Distribution"
                break

            elif step["tool"] == "read_records":
                # Handle read_records with 2-column select as chartable data
                args = step.get("args", {})
                select_val = args.get("select", "")
                if select_val:
                    fields = [f.strip() for f in select_val.split(",") if f.strip()]
                    if len(fields) == 2:
                        # Heuristic: first non-numeric-looking field is x, second is y
                        # The executor will refine this via _pick_columns
                        x_col = fields[0]
                        y_col = fields[1]
                        chart_type = explicit_chart_type or "bar"
                        title = f"{x_col.replace('_', ' ').title()} Distribution"
                        # Do not break here; aggregate_records takes precedence if present

        # Attach multi-series metadata if detected
        if is_multi_metric:
            multi_series = True
            metrics = unique_metrics

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
            "multi_series": multi_series,
            "metrics": metrics,
        }

    # -- RAG inference --
    rag_keywords = get_rag_keywords(tenant_id)
    if any(kw in query_lower for kw in rag_keywords) or category == "policy_info":
        metadata["rag"] = True
        metadata["rag_query"] = user_query

    # -- Export intent inference --
    export_keywords = get_export_keywords(tenant_id)
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
