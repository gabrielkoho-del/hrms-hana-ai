"""System prompt builders for DAB native tool calling and JSON fallback modes."""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.config import DEFAULT_MAX_ROWS, LARGE_RESULT_THRESHOLD, UNLIMITED_ROWS
from agent.core.utils import schema_cache

logger = logging.getLogger("hr_agent")


def build_tone_aware_guidance(tone_context: Dict, leave_codes: Optional[List[str]] = None) -> str:
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
        leave_codes_str = ", ".join(leave_codes) if leave_codes else "ANL, MCL, MAT, UPL, CL, HPL, RPL, EXM, WFH"
        guidance_parts.append(
            "- ACTION REQUEST: The user wants to DO something. "
            f"CRITICAL: For ANY leave-related action request (apply leave, book leave, request leave, take leave, etc.), you MUST follow this exact flow:\n"
            "  STEP 1: Fetch the employee's leave entitlement/balance using read_records on employee_leave_entitlement or v_employee_leave_summary.\n"
            "  STEP 2: In your response text, state the available balance in plain language (e.g. 'You have 22 days of Annual Leave available'). Format whole numbers without decimals (e.g. '22 days', not '22.0 days').\n"
            "  STEP 3: Immediately ask the user for the specific leave date and leave type. Say something like: 'Which date would you like to take your leave?' and 'What type of leave would you like to apply for?'\n"
            "  STEP 4: Do NOT say 'What would you like to do next?' or offer generic options like 'Draft a leave application email' or 'Check the status of your previous leave requests'. These are NOT appropriate for an action request.\n"
            "  STEP 5: Do NOT assume Annual Leave (ANL) or today's date — always confirm leave type and date with the user.\n"
            f"  STEP 6: Only call create_record on employee_leave when ALL of these required fields are confirmed:\n"
            f"           - leave_code (valid values from codesetup: {leave_codes_str})\n"
            "           - date_from (start date, YYYY-MM-DD)\n"
            "           - date_to (end date, YYYY-MM-DD)\n"
            "           - days (number of leave days, supports decimals for half-day)\n"
            "           - period_type_from (1=Full day, 2=Half day AM, 3=Half day PM)\n"
            "           - period_type_to (1=Full day, 2=Half day AM, 3=Half day PM)\n"
            "           - emergency (Y=Yes, N=No)\n"
            "           - status (P=Pending)\n"
            "           - employee_no (from authenticated user)\n"
            "           - hd_id (from the employee_leave_hd record created first)\n"
            "  STEP 7: Create employee_leave_hd first, then use its returned id as hd_id for employee_leave.\n"
            "For other actions: use the appropriate create_record or update_record tool. "
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

    create_record_schema = {
        "type": "function",
        "function": {
            "name": "create_record",
            "description": (
                "Create a new record in a DAB entity (tables only). "
                "Use for employee actions like applying for leave. "
                "The employee_no field will be auto-populated from the user's identity for self-service actions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to create record in (e.g., employee_leave)"
                    },
                    "data": {
                        "type": "object",
                        "description": "Record fields as key-value pairs. For leave: employee_no, leave_code, date_from, date_to, days, status, reason_code."
                    }
                },
                "required": ["entity", "data"]
            }
        }
    }

    update_record_schema = {
        "type": "function",
        "function": {
            "name": "update_record",
            "description": (
                "Update an existing record in a DAB entity by key. "
                "Use for modifying existing records like updating a leave request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name to update record in"
                    },
                    "keys": {
                        "type": "object",
                        "description": "Primary key fields to identify the record to update (e.g., {\"id\": 42})."
                    },
                    "fields": {
                        "type": "object",
                        "description": "Field names and new values to update."
                    }
                },
                "required": ["entity", "keys", "fields"]
            }
        }
    }

    return [read_records_schema, aggregate_records_schema, describe_entities_schema, create_record_schema, update_record_schema]


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
    field_semantics_block: Optional[str] = None,
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
    if field_semantics_block:
        parts.extend(["", field_semantics_block, ""])

    parts.extend([
        "",
        "ENTITY NAMES ARE CASE-SENSITIVE AND MUST MATCH EXACTLY. The entity 'Employees' is NOT the same as 'employees' or 'employee'. Use the exact PascalCase names shown in the schema above. NEVER lowercase, NEVER snake_case, NEVER pluralize or singularize. If the schema says 'Employees', you MUST use 'Employees' exactly.",
        "",
        "FIELD NAMES ARE CASE-SENSITIVE AND MUST MATCH EXACTLY. The field 'EMAIL' is NOT the same as 'email'. The field 'FULL_NAME' is NOT 'full_name'. Use the EXACT field names shown in the schema above -- copy them character-for-character into select, filter, orderby, groupby, and field parameters. This applies to ALL OData arguments: select='EMAIL,FULL_NAME', filter='EMAIL eq value', orderby=['BASIC_SALARY desc'], groupby=['DEPARTMENT']. NEVER guess or normalize field names.",
        "",
        "LEAVE ENTITLEMENT QUERIES: When the user asks about leave entitlement, leave balance, how much leave they have, or similar personal leave questions, query the 'v_employee_leave_summary' or 'v_employee_leave_entitlement' entity. Filter by EMPLOYEE_NO (or employee_no) and year. Do NOT query V_EMP or Employees for leave entitlement data -- those entities do not contain leave balances.",
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
    schema_lines = ["SAP HANA ACCESS -- available schemas/tables:"]
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
        "- HANA DATE/MONTH EXTRACTION: Never use SUBSTRING(BUDAT, 5, 2) to extract month from dates. BUDAT is a DATE field; in HANA SQL it is NOT a plain 'YYYYMMDD' string. Use MONTH(BUDAT) for numeric month (1-12), or TO_VARCHAR(BUDAT, 'MM') for zero-padded month strings ('01'-'12'). For posting period, use POPER directly -- it already contains the fiscal period (1-12 or 1-16).",
        "- MULTI-METRIC FINANCIAL REPORTS: When the user asks for multiple metrics (e.g., Revenue, Gross Profit, EBITDA, Net Profit) in one report, return a WIDE format result with one row per time period and one column per metric. Example: SELECT POPER, SUM(CASE WHEN RACCT IN ('800000','805000') THEN -HSL ELSE 0 END) AS REVENUE, SUM(CASE WHEN RACCT BETWEEN '420000' AND '480000' THEN HSL ELSE 0 END) AS GROSS_PROFIT, ... FROM DBADMIN.FAGLFLEXA WHERE GJAHR = '2025' GROUP BY POPER ORDER BY POPER. Do NOT return separate result sets for each metric. Include a METRIC column only if stacking vertically; prefer horizontal wide format for charts.",
    ])
    return schema_lines


def format_field_semantics_for_prompt(
    cached_schema: Dict,
    dimension_map: Optional[Dict[str, List[str]]] = None,
    leave_codes: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Build a concise field-semantics reference for the system prompt.

    Maps common HR query concepts to the correct DAB fields so the LLM
    picks the right column upfront instead of guessing.
    """
    if not cached_schema:
        return ""

    # Build a lookup of all field names across all entities (lowercase -> exact)
    all_fields: Dict[str, str] = {}
    for entity_data in cached_schema.values():
        if not isinstance(entity_data, dict):
            continue
        for f in entity_data.get("fields", entity_data.get("columns", [])):
            if isinstance(f, dict):
                name = f.get("name", "")
                if name:
                    all_fields[name.lower()] = name

    lines = ["FIELD SEMANTICS (use these exact field names):"]

    # Helper to add a semantic line only if the field exists in the schema
    def _add(concept: str, field_name: str, warning: str = "") -> None:
        if field_name.lower() in all_fields:
            exact = all_fields[field_name.lower()]
            if warning:
                lines.append(f"- {concept}: use {exact}. {warning}")
            else:
                lines.append(f"- {concept}: use {exact}")

    # Track which fields have been added to avoid duplicates
    _added_fields: set = set()

    def _add_unique(concept: str, field_name: str, warning: str = "") -> None:
        if field_name.lower() in all_fields and field_name.lower() not in _added_fields:
            _added_fields.add(field_name.lower())
            _add(concept, field_name, warning)

    # Employment / status
    _add("active/inactive/current/former employees", "EMPLOYEE_STATUS",
         "NOT confirmation_status (that is probation status PROB/CONF). "
         "Filter values are codes, not English words: ACTV=active, RESG=resigned, ABSC=absent, JOIN=joined, TERM=terminated, DECE=deceased.")
    _add("probation/confirmation status", "CONFIRMATION_STATUS",
         "This is probation status, not employment status. "
         "Filter values are codes, not English words: CONF=confirmed, PROB=probation.")
    _add("gender", "GENDER",
         "Filter values are codes, not English words: M=Male, F=Female, S=unknown.")
    _add("marital status", "MARITAL_STATUS",
         "Filter values are codes, not English words: S=Single, M=Married, D=Divorced, C=Widowed, 0=Unknown.")

    # Identifiers
    _add("employee number / ID", "EMPLOYEE_NO")
    _add("employee name", "EMPLOYEE_NAME")
    _add("email address", "EMAIL")

    # Demographics
    _add("birth date / age", "BIRTH_DATE")
    _add("gender", "GENDER")
    _add("marital status", "MARITAL_STATUS")
    _add("nationality", "NATIONALITY_CODE")
    _add("race/ethnicity", "RACE")

    # Dates
    _add("join date / tenure", "DATE_JOINED")
    _add("original join date", "FIRST_DATE_JOINED")
    _add("resignation date", "DATE_RESIGNED")
    _add("confirmation date (end of probation)", "DATE_CONFIRM")

    # Organization — derived from _DIMENSION_MAP so the LLM knows the
    # correct column for each dimension keyword.
    if dimension_map:
        _dim_labels = {
            "department": "department",
            "branch": "branch",
            "company": "company",
            "division": "division",
            "section": "section",
            "position": "position",
            "gender": "gender",
            "nationality": "nationality",
            "marital": "marital status",
            "status": "employee status",
            "location": "location",
        }
        for kw, cols in dimension_map.items():
            preferred = cols[0]
            label = _dim_labels.get(kw, kw)
            _add_unique(label, preferred)

    # Employment details
    _add("employment category", "EMPLOYMENT_CATEGORY")
    _add("employee level / grade", "EMPLOYEE_LEVEL")
    _add("grade / pay grade", "GRADE_CODE")
    _add("category code", "CATEGORY_CODE")
    _add("cost center", "COST_CENTER")
    _add("profit center", "PROFIT_CENTER")

    # Leave enum values — these are codes, not English words.
    lines.append("")
    lines.append("LEAVE ENUM VALUES (use these exact codes in filters and create_record):")
    lines.append(
        "- employee_leave.status / employee_leave_hd.status: P=Pending, A=Approved, R=Rejected, C=Cancelled. "
        "Do NOT use English words like 'Pending' or 'Approved' in OData filters."
    )
    lines.append(
        "- employee_leave.emergency: Y=Yes, N=No."
    )
    lines.append(
        "- employee_leave.period_type_from / period_type_to: 1=Full day, 2=Half day (AM), 3=Half day (PM)."
    )
    if leave_codes:
        # Dynamically fetched from codesetup WHERE type='Leave Type'
        code_str = ", ".join(f"{code}={desc}" for code, desc in leave_codes[:50])
        lines.append(
            f"- employee_leave.leave_code / employee_leave_entitlement.leave_code: {code_str}. "
            "Query codesetup WHERE type='Leave Type' for the canonical list."
        )
    else:
        lines.append(
            "- employee_leave.leave_code / employee_leave_entitlement.leave_code: "
            "ANL=Annual, MCL=Medical Clinic, MAT=Maternity, COM=Compassionate, UPL=Unpaid, WFH=Work from home, "
            "HPL=Hospitalization, RPL=Replacement, CL=Childcare, EXM=Examination, CAL=Call, HOS=Hospitalization, "
            "ABS=Absent, ADL=Accidental, CPL=Compassionate, PAT=Paternity, SPL=Sick, PTL=Paternity, "
            "MRL=Medical, NS=No Show, CC=Career Change, ECC=Emergency, UICL=Unpaid, UML=Unpaid, SPTL=Sick, "
            "RPH=Rest Day, ANL-CT=Annual Carry-over, ANL_ESS=Annual ESS, UPL1/2=Unpaid half-day, "
            "UPL1=Unpaid, UPL_No_ESS=Unpaid, Leave_No_ESS=Leave, RLC1/RLC2=Related, PILF/PILH=Related, "
            "SMTL=Sick, CCL1/CCL2=Childcare, HL=Hospitalization. "
            "Query codesetup WHERE type='Leave Type' for the canonical list."
        )
    lines.append(
        "- employee_leave_entitlement.entitlement_type: Y=Yearly/annual entitlement."
    )

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)


# =============================================================================
# SCHEMA FORMATTING -- selective enrichment + token pruning
# =============================================================================

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
