"""System prompt builders for DAB native tool calling and JSON fallback modes."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.config import DEFAULT_MAX_ROWS, LARGE_RESULT_THRESHOLD, UNLIMITED_ROWS
from agent.core.utils import schema_cache

logger = logging.getLogger("hr_agent")


def build_tone_aware_guidance(tone_context: Dict) -> str:
    if not tone_context:
        return ""
    guidance_parts = []
    category = tone_context.get("intent_category", "policy_info")
    urgency = tone_context.get("urgency_level", "routine")
    emotional = tone_context.get("emotional_state", "neutral")
    needs_empathy = tone_context.get("needs_empathy", False)

    if category in ("personal_data", "emergency"):
        guidance_parts.append(
            "- PERSONAL DATA QUERY: Fetch the user's personal record FIRST. "
            "For leave: fetch BOTH entitlement AND balance. "
            "For profile: fetch all relevant fields. "
            "Never present generic policy as personal data."
        )

    if category == "action_request":
        guidance_parts.append(
            "- ACTION REQUEST: The user wants to DO something. "
            "Fetch all prerequisites they need to complete the action: "
            "eligibility, current status, required approvals, contact info. "
            "Include an 'action_context' describing the specific action."
        )

    if category == "grievance":
        guidance_parts.append(
            "- GRIEVANCE QUERY: Be sensitive. Fetch relevant records (if any) "
            "and include grievance officer or HR contact info in action_context."
        )

    if category == "aggregate_data":
        guidance_parts.append(
            "- AGGREGATE QUERY: The user wants organizational data. "
            "Use aggregate_records when possible. Respect permission filters. "
            "If the schema lacks pre-computed bins (age_group, tenure_group, salary_band), "
            "fetch raw values and set client_side_binning for dynamic binning."
        )

    if category == "workforce_analytics":
        guidance_parts.append(
            "- WORKFORCE PRODUCTIVITY QUERY: Present results as a concise KPI table with columns: KPI | Current Value | Target | Status. "
            "Use bold KPI names and clear status indicators. "
            "Available computations from existing data: HR-to-Employee Ratio and Absenteeism Rate. "
            "For unsupported metrics, state 'N/A - requires [specific data source]' rather than guessing. "
            "End with a brief 'Why it matters' sentence explaining the HR business impact."
        )

    if category == "policy_info":
        guidance_parts.append(
            "- POLICY QUERY: The user wants to know a rule or procedure. "
            "Set rag: true to fetch policy documents. Keep data queries minimal."
        )

    if category == "data_discovery":
        guidance_parts.append(
            "- DATA DISCOVERY QUERY: The user asks what data is available. "
            "Summarize the authorized DAB entities and SAP HANA schemas/tables from the system prompt. "
            "Do not call tools. Offer a direct summary listing the available data sources."
        )

    if needs_empathy or emotional in ("anxious", "distressed", "frustrated"):
        guidance_parts.append(
            "- The user is in an emotional state. Prioritize fetching their personal data (profile, leave balance, "
            "manager info) so the response can be personalized and supportive, not generic policy text."
        )

    if urgency in ("urgent", "time_sensitive"):
        guidance_parts.append(
            "- This is time-sensitive. Prioritize tools that give immediate actionable information. "
            "Avoid tools that return large datasets requiring analysis."
        )

    if not guidance_parts:
        return ""

    return "\nTONE-AWARE CONTEXT (use these hints to reason about tool selection):\n" + "\n".join(guidance_parts)


def build_dab_tool_schemas(cached_schema: Dict) -> List[Dict]:
    entity_examples = ", ".join(list(cached_schema.keys())[:5]) if cached_schema else "employee"

    read_records_schema = {
        "type": "function",
        "function": {
            "name": "read_records",
            "description": (
                "Query records from a DAB entity using OData filters. "
                "Use eq, ne, gt, ge, lt, le, and, or, not. "
                "Text filters do NOT support contains/LIKE — use exact eq or fetch broader. "
                "For dates, use pre-computed fields like hire_year/hire_month — NEVER year(hire_date). "
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
    from agent.integrations.hana_client import get_cached_hana_tool_schemas
    return get_cached_hana_tool_schemas(tenant_id=tenant_id)


def build_all_tool_schemas(cached_schema: Dict, include_hana: bool = True, tenant_id: Optional[str] = None) -> List[Dict]:
    schemas = list(build_dab_tool_schemas(cached_schema))
    if include_hana:
        schemas.extend(build_hana_tool_schemas(tenant_id=tenant_id))
    return schemas


def build_tool_calling_system_prompt(
    schema_block: str,
    entity_list: str,
    user_context: str,
    rag_status: str,
    tone_guidance: str = "",
    hana_schema_registry: Optional[Dict[str, List[str]]] = None,
    hana_semantic_block: Optional[str] = None,
) -> str:
    parts = [
        "You are an HR AI agent with DAB data access. Use the available tools to fetch data. "
        "Call the appropriate tool when you need to query the database. "
        "You may call multiple tools if needed. "
        "For greetings or smalltalk, do not call any tools.",
        "",
        schema_block,
        "",
        *_build_hana_prompt_block(hana_schema_registry),
        "",
    ]

    if hana_semantic_block:
        parts.extend(["", hana_semantic_block, ""])

    parts.extend([
        "Entities: " + entity_list,
        "",
        "RAG: " + rag_status + ".",
    ])

    if user_context:
        parts.extend(["", user_context])
    if tone_guidance:
        parts.extend(["", tone_guidance])

    parts.extend([
        "",
        "ENTITY NAMES ARE CASE-SENSITIVE AND MUST MATCH EXACTLY. The entity 'Employees' is NOT the same as 'employees' or 'employee'. Use the exact PascalCase names shown in the schema above. NEVER lowercase, NEVER snake_case, NEVER pluralize or singularize. If the schema says 'Employees', you MUST use 'Employees' exactly.",
        "",
        "FIELD NAMES ARE CASE-SENSITIVE AND MUST MATCH EXACTLY. The field 'EMAIL' is NOT the same as 'email'. The field 'FULL_NAME' is NOT 'full_name'. Use the EXACT field names shown in the schema above — copy them character-for-character into select, filter, orderby, groupby, and field parameters. This applies to ALL OData arguments: select='EMAIL,FULL_NAME', filter='EMAIL eq value', orderby=['BASIC_SALARY desc'], groupby=['DEPARTMENT']. NEVER guess or normalize field names.",
        "",
        "CHART RULES:",
        "- Pre-binned data (age_group, salary_range) -> bar. Raw numeric distribution -> hist.",
        "- Mermaid supports: bar, barh, pie, line. Matplotlib ONLY: hist, box.",
        "- Under CHART_MODE=auto, hist/box always route to matplotlib.",
        "- No charts for: personal data, single value, <5 rows, action queries, policy questions.",
        "- Time dimension -> line. Proportion <=8 cats -> pie. Otherwise -> barh (default).",
        "",
        "DYNAMIC BINNING (multi-tenant):",
        "If schema lacks pre-computed bins (age_group, salary_band): fetch raw values via aggregate_records groupby [raw_col] first:100, then set client_side_binning in plan.",
        "",
        "CRITICAL RULES:",
        "- Personal queries (my leave, my salary): fetch DB record FIRST. Policy is REFERENCE ONLY. Never substitute policy for personal data.",
        "- Self-referential: filter by auth_context email/emp_id. NEVER query others if read:self.",
        "- If personal record is 0 rows: state 'I checked your records and do not see [X] on file.' Then MAY cite general policy as reference only.",
        "",
        "TMS OVERTIME / ABSENCE PATTERN (V_TMS_OVERTIME + V_EMP):",
        "When the user asks about absence days, absentees, attendance summary, or overtime-related employee lists from V_TMS_OVERTIME, use this exact two-step pattern:",
        "Step 1: Call aggregate_records on entity=V_TMS_OVERTIME with function='count', field='Indicator', groupby=['employee_no'], and filter=\"Indicator ne null\". This returns one row per employee with their absent-day count. Use first='-1' only if the user explicitly asks for all employees; otherwise use a reasonable default like first='50'.",
        "Step 2: From the aggregate result, extract the employee_no values and call read_records on entity=V_EMP with filter=\"EMPLOYEE_NO in (<comma-separated list>)\" and a select list of presentable fields such as EMPLOYEE_NO, EMPLOYEE_NAME, GENDER, DEPARTMENT_CODE, POSITION_CODE, DATE_JOINED. If the employee list is long, paginate with first=50 and provide the after cursor for continuation.",
        "Step 3: Join the two result sets in Python using employee_no as the key. Build a hash map from the V_EMP results, then annotate each V_TMS_OVERTIME aggregate row with employee name and department. This is an O(n) in-memory join with negligible memory cost.",
        "RULES FOR THIS PATTERN:",
        "- NEVER call read_records on V_TMS_OVERTIME without a filter on actual_date or employee_no; it is a large daily attendance view and will return too many rows.",
        "- NEVER return raw V_TMS_OVERTIME daily rows to the user for absence summary questions; always aggregate first.",
        "- The Indicator column contains 'Absent' when the employee was absent; count non-null Indicator values to compute absent days.",
        "- Use actual_date for date filtering if the user specifies a period (e.g., actual_date ge 2025-01-01 and actual_date le 2025-12-31).",
        "- Do NOT attempt to join V_TMS_OVERTIME and V_EMP inside the database; do the join in Python after both tool calls return.",
    ])

    return "\n".join(parts)


def _build_hana_prompt_block(
    hana_schema_registry: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    schema_lines = ["SAP HANA ACCESS — available schemas/tables:"]
    registry = hana_schema_registry or {}
    if registry:
        schema_lines.append("- Authorized schemas and tables (discovered at startup):")
        for schema_name, tables in registry.items():
            normalized_tables = [str(t).upper() for t in tables]
            preview = ", ".join(normalized_tables[:5]) + (" ..." if len(normalized_tables) > 5 else "")
            schema_lines.append(f"  - {schema_name.upper()}: {preview if preview else '(empty)'}")
        schema_lines.append(
            "- DATA AVAILABILITY QUESTIONS: When the user asks what data you have, what's available, or similar discovery questions, respond with the summary above instead of calling tools. "
            "If the user asks for a specific count, total, sum, average, or data retrieval from a known table or schema, USE THE APPROPRIATE TOOL to fetch the data."
        )
        schema_lines.append(
            "- CRITICAL: Always pass schema_name explicitly from the list of AUTHORIZED SCHEMAS above. "
            "DO NOT guess or invent schema names like 'FINANCE'. "
            "For tables already listed under an authorized schema, query them directly with hana_execute_query. "
            "Only call hana_list_tables if you need to verify whether a table exists or discover additional tables."
        )
    else:
        schema_lines.append(
            "- Use hana_list_schemas at the start of the session to enumerate authorized schemas."
        )
    schema_lines.extend([
        "- Use hana_execute_query for SQL queries against SAP HANA.",
        "- Use hana_explain_table / hana_list_tables with explicit schema_name for schema discovery.",
        "- HANA tools return {'columns': [...], 'rows': {...}}",
        "- HANA queries do NOT support OData filters; write plain SQL.",
        "- HANA IDENTIFIER RULES: Unquoted identifiers are case-insensitive and stored as UPPERCASE. Quoted identifiers are case-sensitive. Always use UPPERCASE unquoted identifiers in SQL, e.g., SELECT COUNT(*) FROM BKPF or SELECT COUNT(*) FROM DBADMIN.BKPF. NEVER use lowercase quoted identifiers like 'bkpf' or 'DBADMIN.bkpf'; HANA will reject them.",
        "- FINANCIAL QUERY TIME HANDLING: When user asks for annual financial metrics (gross profit, margin, revenue, EBITDA, net profit, expenses, financial ratios) WITHOUT specifying a year, default to the MOST RECENT COMPLETE FISCAL YEAR, not the current year. Current year data is often incomplete. If current year is 2026, prefer 2025 for annual financials unless user explicitly requests 2026. For quarterly/monthly data, use the most recent complete period. Always mention the year/period used in your answer so the user knows what data was queried.",
        "- HANA DATE/MONTH EXTRACTION: Never use SUBSTRING(BUDAT, 5, 2) to extract month from dates. BUDAT is a DATE field; in HANA SQL it is NOT a plain 'YYYYMMDD' string. Use MONTH(BUDAT) for numeric month (1-12), or TO_VARCHAR(BUDAT, 'MM') for zero-padded month strings ('01'-'12'). For posting period, use POPER directly — it already contains the fiscal period (1-12 or 1-16).",
        "- MULTI-METRIC FINANCIAL REPORTS: When the user asks for multiple metrics (e.g., Revenue, Gross Profit, EBITDA, Net Profit) in one report, return a WIDE format result with one row per time period and one column per metric. Example: SELECT POPER, SUM(CASE WHEN RACCT IN ('800000','805000') THEN -HSL ELSE 0 END) AS REVENUE, SUM(CASE WHEN RACCT BETWEEN '420000' AND '480000' THEN HSL ELSE 0 END) AS GROSS_PROFIT, ... FROM DBADMIN.FAGLFLEXA WHERE GJAHR = '2025' GROUP BY POPER ORDER BY POPER. Do NOT return separate result sets for each metric. Include a METRIC column only if stacking vertically; prefer horizontal wide format for charts.",
    ])
    return schema_lines


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA FORMATTING — selective enrichment + token pruning
# ═════════════════════════════════════════════════════════════════════════════

# Config knobs for schema formatting
MAX_ENTITIES = 20
MAX_FIELDS_PER_ENTITY = 25
MAX_DISTINCT_INLINE = 10
MAX_DISTINCT_VALUES = 10
TOKEN_BUDGET = 5000  # Schema token budget for planner prompt


def _format_field(field: Dict, include_distinct: bool = True) -> str:
    """Format a single field: name:type or name:type{val1,val2}."""
    fname = field.get("name", "")
    ftype = field.get("type", "")
    if not fname:
        return ""

    parts = [f"{fname}:{ftype}"]

    if include_distinct:
        distinct = field.get("distinct_values")
        if isinstance(distinct, (list, tuple)) and 0 < len(distinct) <= MAX_DISTINCT_INLINE:
            vals = ",".join(str(v) for v in distinct[:MAX_DISTINCT_VALUES])
            parts.append(f"{{{vals}}}")

    return "".join(parts)


def _format_entity(entity_name: str, entity: Dict, include_desc: bool = True, include_distinct: bool = True) -> str:
    """Format one entity line: Entity -> field1:type, field2:type{val1,val2}, ..."""
    fields = entity.get("fields", entity.get("columns", [])) if isinstance(entity, dict) else []
    field_strs = []
    for f in fields[:MAX_FIELDS_PER_ENTITY]:
        if isinstance(f, dict):
            s = _format_field(f, include_distinct=include_distinct)
            if s:
                field_strs.append(s)
        else:
            field_strs.append(str(f))
    if len(fields) > MAX_FIELDS_PER_ENTITY:
        field_strs.append("...")

    line = f"  {entity_name} -> {', '.join(field_strs)}"
    if include_desc:
        desc = entity.get("description", "") if isinstance(entity, dict) else ""
        if desc and len(desc) <= 60:
            line += f"  # {desc}"
    return line


def _format_schema_with_selected_fields(schema: Dict, selected_fields: List[str]) -> str:
    """Format schema showing only selected fields + count of remaining fields per entity."""
    lines = ["Entities:"]
    for entity_name, entity in schema.items():
        if not isinstance(entity, dict):
            continue
        fields = entity.get("fields", entity.get("columns", []))
        field_strs = []
        remaining = []
        for f in fields:
            if isinstance(f, dict):
                fname = f.get("name", "")
                if fname in selected_fields:
                    s = _format_field(f, include_distinct=True)
                    if s:
                        field_strs.append(s)
                else:
                    remaining.append(fname)
            else:
                remaining.append(str(f))

        if remaining:
            field_strs.append(f"... ({len(remaining)} more: {', '.join(remaining[:5])}{'...' if len(remaining) > 5 else ''})")

        line = f"  {entity_name} -> {', '.join(field_strs)}"
        desc = entity.get("description", "")
        if desc and len(desc) <= 60:
            line += f"  # {desc}"
        lines.append(line)
    return "\n".join(lines)


def format_schema_for_prompt(schema: Dict, token_budget: int = TOKEN_BUDGET, user_query: str = "") -> str:
    """Production-grade schema formatter with progressive token pruning."""
    if not schema:
        return "Database schema: unavailable"

    entities = [(name, ent) for name, ent in schema.items() if isinstance(ent, dict)]
    entities = entities[:MAX_ENTITIES]

    def _build(include_desc: bool, include_distinct: bool) -> str:
        lines = ["Entities:"]
        for name, ent in entities:
            lines.append(_format_entity(name, ent, include_desc, include_distinct))
        return "\n".join(lines)

    # Try progressively leaner formats until under token budget
    for desc, distinct in [(True, True), (False, True), (False, False)]:
        text = _build(desc, distinct)
        if len(text) // 3 <= token_budget:  # rough token estimate
            return text

    # If still over budget, truncate entities aggressively
    while len(entities) > 3 and len(_build(False, False)) // 3 > token_budget:
        entities = entities[: len(entities) - 1]

    result = _build(False, False)
    if len(result) // 3 > token_budget:
        logger.warning(
            "Schema token budget exhausted even after aggressive pruning. "
            "Consider splitting schema across multiple calls."
        )
    return result


async def format_schema_for_prompt_cached(
    schema: Dict, tenant_id: str = "default", user_query: str = ""
) -> str:
    """Multi-tenant-aware cached schema formatter with 5-min TTL."""
    if not schema:
        return "Database schema: unavailable"

    total_fields = sum(
        len(e.get("fields", e.get("columns", [])))
        for e in schema.values() if isinstance(e, dict)
    )

    # For large schemas with user query, use semantic field retrieval
    if user_query and total_fields > 25:
        try:
            relevant = await _retrieve_relevant_fields(
                user_query, schema, tenant_id=tenant_id, top_k=MAX_FIELDS_PER_ENTITY
            )
            selected_fields = [r["name"] for r in relevant if "name" in r]
            if selected_fields:
                formatted = _format_schema_with_selected_fields(schema, selected_fields)
                logger.info(
                    "Schema formatted with semantic retrieval: %d/%d fields selected",
                    len(selected_fields), total_fields
                )
                return formatted
        except Exception as e:
            logger.warning("Semantic schema retrieval failed: %s. Falling back to full schema.", e)

    # Cache key excludes user_query for base schema; semantic retrieval is fast
    base_key = f"{tenant_id}:{json.dumps(schema, sort_keys=True, default=str)}"
    cached = schema_cache.get(base_key)
    if cached is not None:
        return cached

    formatted = format_schema_for_prompt(schema)
    schema_cache.set(base_key, formatted)
    return formatted


async def _retrieve_relevant_fields(
    user_query: str,
    schema: Dict,
    tenant_id: str = "default",
    top_k: int = 8,
) -> List[Dict]:
    """Retrieve relevant fields using embedding-based semantic retrieval."""
    total_fields = sum(
        len(e.get("fields", e.get("columns", [])))
        for e in schema.values() if isinstance(e, dict)
    )

    # For small schemas, return all fields (no retrieval needed)
    if total_fields <= 25:
        all_fields = []
        for entity_name, entity in schema.items():
            if not isinstance(entity, dict):
                continue
            for field in entity.get("fields", entity.get("columns", [])):
                if isinstance(field, dict):
                    all_fields.append({
                        "entity": entity_name,
                        "name": field.get("name", ""),
                        "type": field.get("type", ""),
                        "description": field.get("description", ""),
                        "score": 1.0,
                    })
        return all_fields

    # For large schemas, use embedding retrieval via schema_index
    try:
        from agent.integrations.schema_index import search_relevant_fields
        return await search_relevant_fields(user_query, schema, tenant_id=tenant_id, top_k=top_k)
    except Exception as e:
        logger.warning("Embedding retrieval failed: %s. Returning full schema.", e)
        # Degrade gracefully: return all fields
        all_fields = []
        for entity_name, entity in schema.items():
            if not isinstance(entity, dict):
                continue
            for field in entity.get("fields", entity.get("columns", [])):
                if isinstance(field, dict):
                    all_fields.append({
                        "entity": entity_name,
                        "name": field.get("name", ""),
                        "type": field.get("type", ""),
                        "description": field.get("description", ""),
                        "score": 1.0,
                    })
        return all_fields


def build_user_context_rules(auth_context: Any) -> str:
    """Build user context rules for prompt injection."""
    if not auth_context or not auth_context.authenticated:
        return ""

    rules = []
    if auth_context.email:
        rules.append(
            "- Your email identity is: EMAIL eq '" + auth_context.email + "'. "
            "Use this value for filtering on the EMAIL field."
        )
    if auth_context.emp_id:
        rules.append(
            "- Your employee ID is: " + auth_context.emp_id + ". "
            "For LOCALDEV tenant, use emp_id field. "
            "For RDEMOROCKFORT tenant, use EMPLOYEE_NO field. "
            "Both support leading zeros (e.g., '000024')."
        )

    if "read:all_employees" not in auth_context.permissions:
        if "read:subordinates" in auth_context.permissions:
            rules.append(
                "- MANAGER ROLE: When querying employees, filter to subordinates only: "
                "manager_id eq YOUR_EMP_ID. You may also include your own record."
            )
        elif "read:self" in auth_context.permissions:
            rules.append(
                "- EMPLOYEE ROLE (SELF-ONLY): You are STRICTLY LIMITED to querying ONLY the user's own profile. "
                "ALL queries MUST include a self-filter using your email or emp_id. "
                "NEVER query other employees' data. If the user asks about others, refuse and explain."
            )

    return "\n".join(rules)
