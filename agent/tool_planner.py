# agent/tool_planner.py
"""Tool planning module for DAB (Data API Builder) — native tool calling + JSON fallback.

Production-grade dual-path architecture:
  Path 1 (primary): Native OpenAI tool calling — model emits structured tool_calls.
  Path 2 (fallback): Text-based JSON parsing — for models/tool configs that don't support tool calling.

All previous fixes preserved:
  • asyncio.to_thread for non-blocking LLM calls
  • Non-greedy JSON regex
  • Aligned client-side binning example
  • Schema-driven entity normalization
"""
import asyncio
import json
import os
import re
import time
import logging
from typing import List, Dict, Optional, Any

from agent.llm_client import call_llm
from agent.config import DEFAULT_MAX_ROWS, LARGE_RESULT_THRESHOLD, UNLIMITED_ROWS

try:
    from agent.hana_client import HANA_TOOL_NAMES
    _HANA_TOOLS_AVAILABLE = True
except Exception:
    HANA_TOOL_NAMES = []
    _HANA_TOOLS_AVAILABLE = False

logger = logging.getLogger("hr_agent")


# ═════════════════════════════════════════════════════════════════════════════
# TOKENIZER — Gemini SentencePiece approximation
# ═════════════════════════════════════════════════════════════════════════════

def _count_tokens(text: str) -> int:
    """Gemini SentencePiece approximation: ~3.5 chars per token."""
    return int(len(text) / 3.5)


# ═════════════════════════════════════════════════════════════════════════════
# TTL CACHE — Multi-tenant safe with automatic expiry
# ═════════════════════════════════════════════════════════════════════════════
class _TTLCache:
    """Lightweight TTL cache for formatted schema strings. No external deps."""

    def __init__(self, ttl: int = 300, maxsize: int = 100):
        self.ttl = ttl
        self.maxsize = maxsize
        self._store: Dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> Optional[str]:
        if key not in self._store:
            return None
        value, ts = self._store[key]
        if time.time() - ts > self.ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str):
        if len(self._store) >= self.maxsize:
            oldest_key = min(self._store, key=lambda k: self._store[k][1])
            del self._store[oldest_key]
        self._store[key] = (value, time.time())

    def clear(self):
        self._store.clear()


_schema_cache = _TTLCache(ttl=300, maxsize=100)  # 5 min TTL, 100 entries


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA EMBEDDING RETRIEVAL — Lightweight in-memory index
# ═════════════════════════════════════════════════════════════════════════════

async def _retrieve_relevant_fields(
    user_query: str,
    schema: Dict,
    tenant_id: str = "default",
    top_k: int = 8,
) -> List[Dict]:
    """Retrieve relevant fields using embedding-based semantic retrieval.

    Falls back to full schema if embedding fails or schema is small (<50 fields).
    """
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
        from agent.schema_index import search_relevant_fields
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


# ═════════════════════════════════════════════════════════════════════════════
# BINNING CONFIGURATION — Flexible column-name matching for multi-tenant schemas
# ═════════════════════════════════════════════════════════════════════════════

BINNING_MAP = {
    "age": {
        "aliases": {
            "age", "date_of_birth", "dob", "birth_date", "birthdate",
            "dateofbirth", "birthday", "date_birth", "d_o_b"
        },
        "config": {
            "method": "post_aggregate",
            "column": "age",
            "bins": [18, 25, 35, 45, 55, 100],
            "labels": ["18-25", "26-35", "36-45", "46-55", "56+"]
        }
    },
    "salary": {
        "aliases": {
            "salary", "basic_salary", "gross_salary", "net_salary",
            "monthly_salary", "annual_salary", "pay", "wage",
            "compensation", "remuneration", "basic_pay", "total_salary"
        },
        "config": {
            "method": "post_aggregate",
            "column": "salary",
            "bins": [0, 3000, 5000, 8000, 12000, 999999],
            "labels": ["<3K", "3-5K", "5-8K", "8-12K", "12K+"]
        }
    },
    "tenure": {
        "aliases": {
            "tenure", "years_of_service", "service_years", "length_of_service",
            "employment_duration", "years_with_company", "company_tenure",
            "service_length", "service_duration"
        },
        "config": {
            "method": "post_aggregate",
            "column": "tenure",
            "bins": [0, 1, 3, 5, 10, 100],
            "labels": ["<1yr", "1-3yr", "3-5yr", "5-10yr", "10yr+"]
        }
    }
}


def _resolve_binning_column(raw_col: str) -> Optional[str]:
    """Map a raw column name to its canonical binning type using aliases.

    Exact match first, then suffix/prefix match for compound names
    (e.g., "basic_salary" -> "salary").
    """
    col_lower = raw_col.lower().strip().replace(" ", "_")
    # Exact match
    for canonical, meta in BINNING_MAP.items():
        if col_lower in meta["aliases"]:
            return canonical
    # Secondary: suffix/prefix match (e.g., "basic_salary" contains "salary")
    for canonical, meta in BINNING_MAP.items():
        if col_lower.endswith(f"_{canonical}") or col_lower.startswith(f"{canonical}_"):
            return canonical
    return None


def _derive_y_label(func: str, entity: str, field: str) -> str:
    """Derive human-readable y-axis label from aggregation context."""
    entity_singular = entity.rstrip("s") if entity else "Employee"
    if func == "count":
        return f"Number of {entity_singular.title()}s"
    if func == "sum":
        return f"Total {field.replace('_', ' ').title()}"
    if func == "avg":
        return f"Average {field.replace('_', ' ').title()}"
    if func == "min":
        return f"Minimum {field.replace('_', ' ').title()}"
    if func == "max":
        return f"Maximum {field.replace('_', ' ').title()}"
    return field.replace("_", " ").title() if field else "Value"


# ═════════════════════════════════════════════════════════════════════════════
# EMPTY PLAN & NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════

def _empty_plan() -> Dict:
    return {
        "steps": [],
        "chart": None,
        "rag": False,
        "rag_query": "",
        "needs_export": False,
        "direct_answer": "",
        "reasoning": "",
        "action_context": "",
        "client_side_binning": None,
    }


def _normalize_plan(parsed: Dict) -> Dict:
    empty = _empty_plan()
    for key in empty:
        if key not in parsed:
            parsed[key] = empty[key]
    if not isinstance(parsed.get("steps"), list):
        parsed["steps"] = []
    return parsed


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA INTROSPECTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _has_column(schema: Dict, entity: str, column: str) -> bool:
    if not schema or entity not in schema:
        return False
    entity_def = schema.get(entity, {})
    fields = entity_def.get("fields", entity_def.get("columns", []))
    field_names = {f.get("name", "").lower() for f in fields if isinstance(f, dict)}
    return column.lower() in field_names


# ═════════════════════════════════════════════════════════════════════════════
# SCHEMA FORMATTING — PRODUCTION-GRADE: selective enrichment + token pruning
# ═════════════════════════════════════════════════════════════════════════════

# Config knobs for schema formatting
MAX_ENTITIES = 20
MAX_FIELDS_PER_ENTITY = 25
MAX_DISTINCT_INLINE = 10
MAX_DISTINCT_VALUES = 10
TOKEN_BUDGET = 5000  # Schema token budget for planner prompt; Gemini 3.1 Flash Lite has 1M context, so this is conservative (~0.5%)


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
    """Production-grade schema formatter with progressive token pruning.

    Strategy (in order of pruning severity):
      1. Full format: descriptions + distinct values + all fields
      2. Remove descriptions
      3. Remove distinct values
      4. Truncate to MAX_FIELDS_PER_ENTITY fields
      5. Truncate to MAX_ENTITIES entities
    """
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
        if _count_tokens(text) <= token_budget:
            return text

    # If still over budget, truncate entities aggressively
    while len(entities) > 3 and _count_tokens(_build(False, False)) > token_budget:
        entities = entities[: len(entities) - 1]

    result = _build(False, False)
    if _count_tokens(result) > token_budget:
        logger.error(
            "Schema token budget exhausted even after aggressive pruning (%d tokens > %d). "
            "Consider splitting schema across multiple calls or using schema summaries.",
            _count_tokens(result), token_budget
        )
    return result


async def format_schema_for_prompt_cached(
    schema: Dict, tenant_id: str = "default", user_query: str = ""
) -> str:
    """Multi-tenant-aware cached schema formatter with 5-min TTL.

    When user_query is provided and schema is large (>25 fields), uses embedding-based
    semantic retrieval to select only relevant fields for the prompt.
    """
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
    cached = _schema_cache.get(base_key)
    if cached is not None:
        return cached

    formatted = format_schema_for_prompt(schema)
    _schema_cache.set(base_key, formatted)
    return formatted


# ═════════════════════════════════════════════════════════════════════════════
# USER CONTEXT INJECTION
# ═════════════════════════════════════════════════════════════════════════════
def build_user_context_rules(auth_context: Any) -> str:
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


# ═════════════════════════════════════════════════════════════════════════════
# DAB FILTER VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
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


# ═════════════════════════════════════════════════════════════════════════════
# JSON PARSING (fallback path)
# ═════════════════════════════════════════════════════════════════════════════
def parse_json_from_text(text: str) -> Dict:
    code_block_match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return _empty_plan()

    try:
        parsed = json.loads(match.group())
        return _normalize_plan(parsed)
    except json.JSONDecodeError:
        pass

    return _extract_plan_with_regex(text)


def _extract_plan_with_regex(text: str) -> Dict:
    plan = _empty_plan()

    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', text)
    if reasoning_match:
        plan["reasoning"] = reasoning_match.group(1)

    direct_match = re.search(r'"direct_answer"\s*:\s*"([^"]*)"', text)
    if direct_match:
        plan["direct_answer"] = direct_match.group(1)
        return plan

    steps_match = re.search(r'"steps"\s*:\s*(\[.*?\])', text, re.DOTALL)
    if steps_match:
        try:
            plan["steps"] = json.loads(steps_match.group(1))
        except:
            pass

    logger.warning("Used minimal regex fallback. Steps: %d", len(plan["steps"]))
    return plan


# ═════════════════════════════════════════════════════════════════════════════
# TONE-AWARE TOOL SELECTION
# ═════════════════════════════════════════════════════════════════════════════
def build_tone_aware_guidance(tone_context: Dict) -> str:
    if not tone_context:
        return ""

    guidance_parts = []
    category = tone_context.get("intent_category", "policy_info")
    urgency = tone_context.get("urgency_level", "routine")
    emotional = tone_context.get("emotional_state", "neutral")
    needs_empathy = tone_context.get("needs_empathy", False)
    action_oriented = tone_context.get("action_oriented", False)

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

    if category == "policy_info":
        guidance_parts.append(
            "- POLICY QUERY: The user wants to know a rule or procedure. "
            "Set rag: true to fetch policy documents. Keep data queries minimal."
        )

    if needs_empathy or emotional in ("anxious", "distressed", "frustrated"):
        guidance_parts.append(
            "- The user is in an emotional state. Prioritize fetching their personal data (profile, leave balance, "
            "manager info) so the response can be personalized and supportive, not generic policy text."
        )

    if urgency in ("urgent", "distressed", "time_sensitive"):
        guidance_parts.append(
            "- This is time-sensitive. Prioritize tools that give immediate actionable information. "
            "Avoid tools that return large datasets requiring analysis."
        )

    if not guidance_parts:
        return ""

    return "\nTONE-AWARE CONTEXT (use these hints to reason about tool selection):\n" + "\n".join(guidance_parts)


# ═════════════════════════════════════════════════════════════════════════════
# NATIVE TOOL CALLING — OpenAI-compatible schemas for DAB tools
# ═════════════════════════════════════════════════════════════════════════════

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


def build_hana_tool_schemas() -> List[Dict]:
    """Build OpenAI function-calling schemas for SAP HANA MCP tools."""
    hana_execute_query = {
        "type": "function",
        "function": {
            "name": "hana_execute_query",
            "description": (
                "Execute SQL against SAP HANA finance database. "
                "Use for finance/GL queries, cost centers, balance sheets, etc. "
                "Supports SELECT/WITH; results include columns and rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute against HANA"
                    },
                    "maxRows": {
                        "type": "number",
                        "description": "Max rows to return"
                    },
                    "includeTotal": {
                        "type": "boolean",
                        "description": "If true, also return total row count"
                    }
                },
                "required": ["query"]
            }
        }
    }

    hana_describe_table = {
        "type": "function",
        "function": {
            "name": "hana_describe_table",
            "description": "Describe the structure of a HANA table (columns, types).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table name"},
                    "schema_name": {"type": "string", "description": "Schema name (optional)"},
                    "catalog_database": {"type": "string", "description": "MDC catalog database (optional)"}
                },
                "required": ["table_name"]
            }
        }
    }

    hana_list_tables = {
        "type": "function",
        "function": {
            "name": "hana_list_tables",
            "description": "List tables in a HANA schema with optional prefix filter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "schema_name": {"type": "string", "description": "Schema name (optional)"},
                    "prefix": {"type": "string", "description": "Table name prefix filter (optional)"},
                    "limit": {"type": "number", "description": "Max tables to return (optional)"},
                    "offset": {"type": "number", "description": "Pagination offset (optional)"}
                },
                "required": []
            }
        }
    }

    hana_get_sample_data = {
        "type": "function",
        "function": {
            "name": "hana_get_sample_data",
            "description": "Fetch sample rows from a HANA table (SELECT TOP N).",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table name"},
                    "schema_name": {"type": "string", "description": "Schema name (optional)"},
                    "limit": {"type": "number", "description": "Number of rows (default 10, max 1000)"}
                },
                "required": ["table_name"]
            }
        }
    }

    return [hana_execute_query, hana_describe_table, hana_list_tables, hana_get_sample_data]


def build_all_tool_schemas(cached_schema: Dict, include_hana: bool = True) -> List[Dict]:
    """Merge DAB and HANA tool schemas for the planner prompt.

    Args:
        cached_schema: DAB entity schema map.
        include_hana: If True, append HANA schemas (requires HANA server reachable).
    """
    schemas = list(build_dab_tool_schemas(cached_schema))
    if include_hana:
        schemas.extend(build_hana_tool_schemas())
    return schemas


_ALL_DAB_TOOLS = {"read_records", "aggregate_records", "describe_entities"}
_ALL_HANA_TOOLS = set(HANA_TOOL_NAMES)


def extract_steps_from_tool_calls(tool_calls: List[Dict], allowed_tools: Optional[set] = None) -> List[Dict]:
    """Convert LLM tool_calls to plan steps.

    Args:
        tool_calls: Raw tool_calls from LLM response.
        allowed_tools: Set of tool names to permit. If None, allow DAB + HANA core tools.
    """
    if allowed_tools is None:
        allowed_tools = _ALL_DAB_TOOLS | _ALL_HANA_TOOLS

    steps = []
    for tc in tool_calls:
        tool_name = tc["function"]["name"]
        if tool_name not in allowed_tools:
            logger.warning("LLM tried to call unauthorized tool '%s' — skipping", tool_name)
            continue
        try:
            args = json.loads(tc["function"]["arguments"])
            steps.append({"tool": tool_name, "args": args})
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse tool arguments for %s: %s", tool_name, e)
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


# ═════════════════════════════════════════════════════════════════════════════
# METADATA INFERENCE — Heuristic chart/rag/binning from query + steps
# ═════════════════════════════════════════════════════════════════════════════

def infer_metadata(user_query: str, steps: List[Dict], tone_context: Optional[Dict]) -> Dict:
    """Infer chart, RAG, client-side binning, and other metadata from query and steps.

    This is deterministic and fast — no extra API call needed.
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

    # HANA queries use raw SQL; skip DAB chart/binning metadata inference
    if has_hana and not has_aggregate and not has_read:
        metadata["reasoning"] = f"Selected {len(steps)} HANA tool(s) for finance data."
        return metadata

    # ── Chart inference (intent category + data shape, NOT keywords) ──
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

    # Detect explicit chart type request from user query
    explicit_chart_type = None
    query_lower = user_query.lower()
    if any(kw in query_lower for kw in ("bar chart", "bar graph", "bar plot", "column chart", "column graph")):
        explicit_chart_type = "bar"
    elif any(kw in query_lower for kw in ("pie chart", "pie graph", "pie plot", "donut chart")):
        explicit_chart_type = "pie"
    elif any(kw in query_lower for kw in ("line chart", "line graph", "trend chart", "time series")):
        explicit_chart_type = "line"
    elif any(kw in query_lower for kw in ("histogram", "hist chart", "distribution chart")):
        explicit_chart_type = "hist"
    elif any(kw in query_lower for kw in ("box plot", "boxplot", "box chart")):
        explicit_chart_type = "box"
    if explicit_chart_type:
        logger.info("Planner: explicit chart type detected from query: %s", explicit_chart_type)

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
        }

    # ── RAG inference ──
    rag_keywords = ["policy", "rule", "handbook", "procedure", "guideline", "entitled", "eligible", "how do i", "how to"]
    if any(kw in query_lower for kw in rag_keywords) or category == "policy_info":
        metadata["rag"] = True
        metadata["rag_query"] = user_query

    # ── Export intent inference ──
    export_keywords = ("export", "download", "save", "excel", "spreadsheet",
                         "xlsx", "workbook", "file", "send me", "give me the data")
    wants_export = (
        any(kw in query_lower for kw in export_keywords)
        and category == "aggregate_data"
        and (has_aggregate or has_read)
    )
    if wants_export:
        metadata["needs_export"] = True

    # ── Client-side binning inference ──
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

    # ── Action context ──
    metadata["action_context"] = tone_context.get("action_context", "")

    # ── Reasoning ──
    metadata["reasoning"] = (
        f"Selected {len(steps)} tool(s) based on query intent ({category}). "
        f"Chart={'yes' if metadata['chart'] else 'no'}, RAG={'yes' if metadata['rag'] else 'no'}, "
        f"Export={'yes' if metadata['needs_export'] else 'no'}."
    )

    return metadata


# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

def build_tool_calling_system_prompt(
    schema_block: str,
    entity_list: str,
    user_context: str,
    rag_status: str,
    tone_guidance: str = "",
) -> str:
    """Compact system prompt for native tool calling mode.

    Much shorter than JSON mode prompt because tool schemas carry parameter definitions.
    """
    parts = [
        "You are an HR AI agent with DAB data access. Use the available tools to fetch data. "
        "Call the appropriate tool when you need to query the database. "
        "You may call multiple tools if needed. "
        "For greetings or smalltalk, do not call any tools.",
        "",
        schema_block,
        "",
        "Entities: " + entity_list,
        "",
        "RAG: " + rag_status + ".",
    ]

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
        "HANA (SAP HANA finance data):",
        "- Use hana_execute_query for finance/GL/cost-center SQL queries.",
        "- Use hana_describe_table / hana_list_tables for HANA schema discovery.",
        "- HANA tools return {'columns': [...], 'rows': [...]}; the system normalizes them automatically.",
        "- HANA queries do NOT support OData filters; write plain SQL.",
        "",
        "CRITICAL RULES:",
        "- Personal queries (my leave, my salary): fetch DB record FIRST. Policy is REFERENCE ONLY. Never substitute policy for personal data.",
        "- Self-referential: filter by auth_context email/emp_id. NEVER query others if read:self.",
        "- If personal record is 0 rows: state 'I checked your records and do not see [X] on file.' Then MAY cite general policy as reference only.",
    ])

    return "\n".join(parts)


def build_json_fallback_system_prompt(
    schema_block: str,
    entity_list: str,
    user_context: str,
    rag_status: str,
    tone_guidance: str = "",
) -> str:
    """Full system prompt for JSON fallback mode (when tool calling fails)."""
    parts = [
        "You are an HR AI agent with DAB data access. Reason: DISCOVER entity -> REASON fields/filters -> QUERY with correct tool. First char = {. Last char = }. NO markdown fences. NO text outside JSON.",
        "",
        schema_block,
        "",
        "Entities: " + entity_list,
        "",
        "TOOLS:",
        "- read_records(entity, select, filter, orderby, first): OData filter uses eq, ne, gt, ge, lt, le, and, or, not. Text filters do NOT support contains/LIKE — use exact eq or fetch broader. orderby: [\"salary desc\"]. first: max rows (default " + str(DEFAULT_MAX_ROWS) + ", use " + str(UNLIMITED_ROWS) + " for 'all').",
        "- aggregate_records(entity, function, field, groupby, orderby, filter, first, having): function = count|sum|avg|min|max. Use * for count. groupby for grouped results. first for max groups. DATE: use pre-computed hire_year/hire_month — NEVER year(hire_date).",
        "- describe_entities(): Discover fields. Use only when uncertain.",
        "",
        "RAG: " + rag_status + ".",
    ]

    if user_context:
        parts.extend(["", user_context])
    if tone_guidance:
        parts.extend(["", tone_guidance])

    parts.extend([
        "",
        "ENTITY NAMES ARE CASE-SENSITIVE AND MUST MATCH EXACTLY. The entity 'Employees' is NOT the same as 'employees' or 'employee'. Use the exact PascalCase names shown in the schema above. NEVER lowercase, NEVER snake_case, NEVER pluralize or singularize. If the schema says 'Employees', you MUST use 'Employees' exactly.",
        "",
        "CHART RULES:",
        "- Pre-binned data (age_group, salary_range) -> bar/barh. Raw numeric distribution -> hist.",
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
        "OUTPUT FORMAT — RAW JSON ONLY:",
        '{"reasoning": "brief analysis", "steps": [{"tool": "read_records", "args": {...}}], "chart": {"type": "bar", "x_column": "", "y_column": "", "title": ""}, "rag": false, "rag_query": "", "needs_export": false, "direct_answer": "", "action_context": "", "client_side_binning": null}',
        "",
        "client_side_binning format (when schema lacks pre-computed bins):",
        '{"method": "post_aggregate", "column": "age", "bins": [18, 25, 35, 45, 55, 100], "labels": ["18-25", "26-35", "36-45", "46-55", "56+"]}',
        "",
        "RULES: steps=[] for greetings. chart can be null. client_side_binning can be null. Never omit keys. First char = {. Last char = }."
    ])

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# PLAN EXTRACTOR — Shared between tool calling and JSON fallback
# ═════════════════════════════════════════════════════════════════════════════

def _build_plan_from_choice(choice) -> Optional[Dict]:
    """Extract and validate a plan from an LLM choice (JSON fallback path)."""
    if not choice:
        return None
    msg = choice.get("message", {})

    content = msg.get("content", "")
    if not content:
        logger.warning("Planner: LLM returned empty content. finish_reason=%s", msg.get("finish_reason"))
        return None

    parsed = parse_json_from_text(content)
    allowed_tools = {"read_records", "aggregate_records", "describe_entities"}
    filtered_steps = [s for s in parsed.get("steps", []) if s.get("tool") in allowed_tools]
    if len(filtered_steps) != len(parsed.get("steps", [])):
        logger.warning("Filtered out unauthorized tools from parsed plan")

    plan = {
        "steps": filtered_steps,
        "direct_answer": parsed.get("direct_answer", ""),
        "chart": parsed.get("chart"),
        "rag": parsed.get("rag", False),
        "rag_query": parsed.get("rag_query", ""),
        "needs_export": parsed.get("needs_export", False),
        "reasoning": parsed.get("reasoning", ""),
        "action_context": parsed.get("action_context", ""),
        "client_side_binning": parsed.get("client_side_binning"),
    }

    if not plan["steps"] and not plan["direct_answer"]:
        logger.warning("Planner: parsed plan has no steps and no direct_answer")
        return None

    if plan["steps"]:
        logger.info("Parsed tool plan: %s | reasoning: %s | action_context: %s | client_side_binning: %s",
                   [s["tool"] for s in plan["steps"]],
                   plan.get("reasoning", "")[:200],
                   plan.get("action_context", "")[:100],
                   "YES" if plan.get("client_side_binning") else "NO")
    if plan.get("direct_answer"):
        logger.info("Direct answer from parsed JSON | reasoning: %s", plan.get("reasoning", "")[:200])

    return plan


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PLANNER — Dual-path: native tool calling (primary) -> JSON fallback
# ═════════════════════════════════════════════════════════════════════════════
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
) -> Dict:
    """Build an agentic, permission-aware tool plan for DAB.

    Dual-path architecture:
      1. Native tool calling (primary) — model emits structured tool_calls.
      2. JSON text parsing (fallback) — for compatibility or tool-calling failures.
    """
    if not cached_tools:
        return _empty_plan()

    tenant_id = getattr(auth_context, "tenant_id", "default") if auth_context else "default"
    schema_block = await format_schema_for_prompt_cached(
        cached_schema, tenant_id=tenant_id, user_query=user_query
    )
    schema_tokens = _count_tokens(schema_block)
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
        full_query = "Previous conversation:\n" + conversation_history + "\n\nCurrent question: " + user_query

    # ═══════════════════════════════════════════════════════════════════════
    # PATH 1: Native tool calling (primary)
    # ═══════════════════════════════════════════════════════════════════════
    all_tools = build_all_tool_schemas(cached_schema, include_hana=True)
    system_tc = build_tool_calling_system_prompt(
        schema_block, entity_list, user_context, rag_status, tone_guidance
    )

    logger.info("Planner: Path 1 — native tool calling (system=%d tokens, tools=%d)", _count_tokens(system_tc), len(all_tools))

    # Planner tier: gemini-3.5-flash (250K TPM, 15 RPM, 500 RPD free tier)
    # Estimated ~3,500–5,500 tokens per call depending on schema size
    choice = await asyncio.to_thread(
        call_llm, system_tc, full_query,
        tools=all_tools, temperature=0.1, json_mode=False,
        tier="planner", estimated_tokens=5500
    )

    steps = []
    if choice and choice.get("message", {}).get("tool_calls"):
        steps = extract_steps_from_tool_calls(choice["message"]["tool_calls"])
        steps = _normalize_entity_names(steps, cached_schema)
        logger.info("Planner: Tool calling produced %d steps: %s", len(steps), [s["tool"] for s in steps])

    if steps:
        # Infer metadata from query + steps (deterministic, no extra API call)
        metadata = infer_metadata(user_query, steps, tone_context)

        # If the model also provided content with additional metadata, merge it
        content = choice.get("message", {}).get("content", "")
        if content:
            try:
                parsed = parse_json_from_text(content)
                if parsed.get("chart"):
                    metadata["chart"] = parsed["chart"]
                if parsed.get("rag"):
                    metadata["rag"] = parsed["rag"]
                if parsed.get("rag_query"):
                    metadata["rag_query"] = parsed["rag_query"]
                if parsed.get("client_side_binning"):
                    metadata["client_side_binning"] = parsed["client_side_binning"]
                if parsed.get("action_context"):
                    metadata["action_context"] = parsed["action_context"]
                if parsed.get("reasoning"):
                    metadata["reasoning"] = parsed["reasoning"]
            except Exception as e:
                logger.debug("Failed to parse content metadata: %s", e)

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

        logger.info("Planner: Path 1 SUCCESS — %d steps, chart=%s, rag=%s, binning=%s",
                   len(plan["steps"]),
                   "yes" if plan["chart"] else "no",
                   "yes" if plan["rag"] else "no",
                   "yes" if plan["client_side_binning"] else "no")
        return plan

    # ═══════════════════════════════════════════════════════════════════════
    # PATH 2: JSON text parsing (fallback)
    # ═══════════════════════════════════════════════════════════════════════
    logger.warning("Planner: Path 1 failed (no tool steps), attempting Path 2 — JSON fallback")

    system_json = build_json_fallback_system_prompt(
        schema_block, entity_list, user_context, rag_status, tone_guidance
    )

    choice = await asyncio.to_thread(
        call_llm, system_json, full_query,
        tools=None, temperature=0.1, json_mode=False,
        tier="planner", estimated_tokens=5500
    )

    plan = _build_plan_from_choice(choice)
    if plan and plan.get("steps"):
        plan["steps"] = _normalize_entity_names(plan["steps"], cached_schema)
    if plan:
        logger.info("Planner: Path 2 SUCCESS — JSON fallback produced valid plan")
        return plan

    logger.error("Planner: Both paths failed. Returning empty plan.")
    return _empty_plan()