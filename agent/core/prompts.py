"""System prompt builders for DAB native tool calling and JSON fallback modes."""
from typing import Dict, List, Optional, Any

from agent.config import DEFAULT_MAX_ROWS, LARGE_RESULT_THRESHOLD, UNLIMITED_ROWS

logger = None  # logging.getLogger("hr_agent") — imported in tool_planner.py


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
    ])
    return schema_lines
