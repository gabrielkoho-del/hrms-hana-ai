# agent/tool_planner.py
"""Tool planning module for DAB (Data API Builder) — native tool calling.

Production-grade architecture:
  Native OpenAI tool calling — model emits structured tool_calls.

All previous fixes preserved:
  • asyncio.to_thread for non-blocking LLM calls
  • Schema-driven entity normalization
  • Aligned client-side binning example
"""
import asyncio
import hashlib
import json
import os
import re
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any

import jsonschema
import tiktoken

from agent.integrations.llm_client import call_llm
from agent.integrations.hana_client import normalize_hana_result
from agent.config import DEFAULT_MAX_ROWS, LARGE_RESULT_THRESHOLD, UNLIMITED_ROWS, PLANNER_ESTIMATED_TOKENS, CONVERSATION_HISTORY_TOKEN_BUDGET
from agent.core.prompts import (
    build_tone_aware_guidance,
    build_dab_tool_schemas,
    build_hana_tool_schemas,
    build_all_tool_schemas,
    build_tool_calling_system_prompt,
)
from agent.output.binning import (
    BINNING_MAP,
    _resolve_binning_column,
    _derive_y_label,
)

logger = logging.getLogger("hr_agent")


def _load_hana_semantic_schema() -> Optional[Dict]:
    """Load HANA semantic descriptions from hana-semantics-hr.json."""
    from pathlib import Path
    env_path = os.getenv("HANA_SEMANTICS_PATH", "")
    if env_path:
        semantics_path = Path(env_path)
        if not semantics_path.is_absolute():
            semantics_path = Path(__file__).resolve().parent.parent.parent / semantics_path
    else:
        semantics_path = Path(__file__).resolve().parent.parent.parent / "hana-mcp-server" / "config" / "hana-semantics-hr.json"

    if not semantics_path.exists():
        logger.debug("HANA semantics file not found: %s", semantics_path)
        return None

    try:
        with open(semantics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("tables"):
            return data
        logger.warning("HANA semantics file missing 'tables' key: %s", semantics_path)
    except Exception as e:
        logger.warning("HANA semantics load failed: %s", e)
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIC TEMPLATE ENFORCEMENT — programmatic SQL from hana-semantics-hr.json
# ═════════════════════════════════════════════════════════════════════════════

def _match_semantic_pattern(user_query: str, hana_semantic_schema: Optional[Dict]) -> Optional[Dict]:
    """Match user query against hana-semantics-hr.json query_patterns.

    Returns pattern match dict with sql_template if matched, None otherwise.
    Matching is based on domain keyword overlap between user_query and
    the pattern's when_to_use description.
    """
    if not hana_semantic_schema or "query_patterns" not in hana_semantic_schema:
        return None

    q_lower = user_query.lower()
    patterns = hana_semantic_schema["query_patterns"]

    for pattern_name, pattern_def in patterns.items():
        when_to_use = pattern_def.get("when_to_use", "").lower()
        sql_template = pattern_def.get("sql_template", "")
        if not sql_template:
            continue

        # Extract domain keywords from when_to_use description
        domain_keywords = set()
        if "vendor" in when_to_use or "supplier" in when_to_use:
            domain_keywords.update(["vendor", "supplier", "ap ", "accounts payable", "expense", "cost", "payment"])
        if "customer" in when_to_use:
            domain_keywords.update(["customer", "client", "ar ", "accounts receivable", "revenue", "sales"])
        if "revenue" in when_to_use:
            domain_keywords.update(["revenue", "sales", "turnover"])
        if "expense" in when_to_use or "cost" in when_to_use:
            domain_keywords.update(["expense", "cost", "payment"])
        if "aging" in when_to_use:
            domain_keywords.update(["aging", "overdue", "open item", "open items"])
        if "period" in when_to_use or "month" in when_to_use or "quarter" in when_to_use:
            domain_keywords.update(["by month", "by quarter", "by period", "trend", "monthly", "quarterly"])

        # Check if query matches this pattern's domain
        if not any(kw in q_lower for kw in domain_keywords):
            continue

        # For top-N patterns, require ranking language
        if pattern_name.startswith("top_") or "top" in when_to_use:
            if not any(kw in q_lower for kw in ["top", "best", "highest", "largest", "rank"]):
                continue

        return {
            "pattern_name": pattern_name,
            "sql_template": sql_template,
            "correct_source": pattern_def.get("correct_source"),
            "incorrect_source": pattern_def.get("incorrect_source"),
        }

    return None


def _repair_sql_with_semantic_template(
    sql: str,
    user_query: str,
    hana_semantic_schema: Optional[Dict],
    tenant_id: str = "default",
) -> Optional[str]:
    """Attempt to repair invalid SQL using a semantic template from hana-semantics-hr.json.

    Returns repaired SQL if a matching template is found, None otherwise.
    """
    match = _match_semantic_pattern(user_query, hana_semantic_schema)
    if not match:
        return None

    sql_template = match["sql_template"]

    # Extract {n} from user query (e.g., "top 5", "top 10")
    n_match = re.search(r'\btop\s+(\d+)\b', user_query, re.IGNORECASE)
    n = int(n_match.group(1)) if n_match else 5

    # Extract {year}
    year = _extract_year_from_query(user_query)
    if year is None:
        year = datetime.now().year - 1

    try:
        repaired = sql_template.format(n=n, year=year)
        logger.info(
            "SEMANTIC_TEMPLATE_REPAIR: pattern=%s replaced invalid SQL with template SQL",
            match["pattern_name"]
        )
        return repaired
    except (KeyError, ValueError) as e:
        logger.warning("Semantic template substitution failed for %s: %s", match["pattern_name"], e)
        return None


def _validate_hana_sql(query: str) -> tuple[bool, Optional[str]]:
    """Validate HANA SQL for common correctness issues.

    Lightweight regex-based validation. No external dependencies.
    Returns (is_valid, error_message).
    """
    if not query or not isinstance(query, str):
        return False, "Empty or invalid SQL query"

    q = query.strip()
    q_upper = q.upper()

    # Check 1: Aggregate without GROUP BY when non-aggregated columns are selected
    has_aggregate = bool(re.search(r'\b(SUM|COUNT|AVG|MIN|MAX)\s*\(', q_upper))
    has_group_by = bool(re.search(r'\bGROUP\s+BY\b', q_upper))

    if has_aggregate and not has_group_by:
        select_part = re.search(r'SELECT\s+(.*?)\s+FROM', q_upper, re.DOTALL | re.IGNORECASE)
        if select_part:
            cols = select_part.group(1)
            # Remove aggregate expressions to check for non-aggregated columns
            cols_without_agg = re.sub(r'\b(SUM|COUNT|AVG|MIN|MAX)\s*\([^)]*\)', '', cols)
            # If there's still a column name, there are non-aggregated columns
            if re.search(r'\b[A-Z_][A-Z0-9_]*\b', cols_without_agg):
                return False, "SQL contains aggregate functions with non-aggregated columns but no GROUP BY clause"

    # Check 2: JOIN without ON clause
    joins = re.findall(r'\bJOIN\b', q_upper)
    ons = re.findall(r'\bON\b', q_upper)
    if joins and len(ons) < len(joins):
        return False, f"SQL contains {len(joins)} JOIN(s) but only {len(ons)} ON clause(s) — possible Cartesian product"

    # Check 3: CROSS JOIN
    if re.search(r'\bCROSS\s+JOIN\b', q_upper):
        return False, "SQL contains CROSS JOIN — verify this is intentional"

    return True, None


# ═════════════════════════════════════════════════════════════════════════════
# TEMPORAL REASONING — Intemporal context resolution for financial queries
# ═════════════════════════════════════════════════════════════════════════════

_FINANCIAL_METRIC_KEYWORDS = frozenset({
    "gross profit", "gross margin", "net profit", "net margin", "ebitda",
    "revenue", "sales", "income", "expense", "cost", "budget", "actual",
    "balance sheet", "cash flow", "cashflow", "profitability", "margin",
    "profit", "financial", "finance", "fiscal", "accounting", "gl ", "general ledger",
    "accounts payable", "accounts receivable", "ap ", "ar ", "invoice",
    "vendor", "customer", "payment", "receipt", "bank", "cash", "asset",
    "liability", "equity", "depreciation", "amortization", "tax",
})

_YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")


def _is_financial_metric_query(user_query: str) -> bool:
    """Heuristic check whether the query is about financial metrics."""
    q = user_query.lower()
    return any(kw in q for kw in _FINANCIAL_METRIC_KEYWORDS)


def _extract_year_from_query(user_query: str) -> Optional[int]:
    """Extract explicit year mention from the query, if any."""
    matches = _YEAR_PATTERN.findall(user_query)
    if not matches:
        return None
    # Prefer the last mentioned year (usually the relevant one)
    return int(matches[-1])


async def _get_available_fiscal_years(tenant_id: str = "default") -> List[int]:
    """Return available fiscal years from HANA financial tables, sorted descending."""
    try:
        from agent.integrations.hana_client import hana_manager
        client = hana_manager.get_client(tenant_id)
        years = set()
        for query in [
            "SELECT DISTINCT GJAHR FROM DBADMIN.BKPF ORDER BY GJAHR DESC",
            "SELECT DISTINCT RYEAR FROM DBADMIN.FAGLFLEXT ORDER BY RYEAR DESC",
            "SELECT DISTINCT RYEAR FROM DBADMIN.GLT0 ORDER BY RYEAR DESC",
        ]:
            try:
                result = client.call_tool("hana_execute_query", {"query": query, "maxRows": 50})
                normalized = normalize_hana_result(result, "hana_execute_query")
                if isinstance(normalized, dict) and "result" in normalized:
                    for row in normalized["result"]:
                        for value in row.values():
                            if isinstance(value, (int, str)):
                                try:
                                    years.add(int(str(value).strip()))
                                except (TypeError, ValueError):
                                    pass
            except Exception:
                continue
        return sorted(years, reverse=True)
    except Exception:
        return []


def _resolve_temporal_context(user_query: str, available_years: List[int]) -> Dict[str, Any]:
    """Resolve temporal context for financial queries.

    Returns a dict with:
      - resolved_year: int or None
      - is_inferred: bool
      - note: str
    """
    explicit_year = _extract_year_from_query(user_query)
    if explicit_year is not None:
        return {
            "resolved_year": explicit_year,
            "is_inferred": False,
            "note": f"Using explicitly requested year {explicit_year}.",
        }

    current_year = datetime.now().year
    if not available_years:
        fallback = current_year - 1 if current_year > 2000 else current_year
        return {
            "resolved_year": fallback,
            "is_inferred": True,
            "note": f"No available fiscal years detected; defaulting to most recent complete year {fallback}.",
        }

    if current_year in available_years:
        # Current year exists; treat it as potentially incomplete.
        complete_years = [y for y in available_years if y < current_year]
        if complete_years:
            resolved = complete_years[0]
            return {
                "resolved_year": resolved,
                "is_inferred": True,
                "note": f"Current year {current_year} data may be incomplete; defaulting to most recent complete year {resolved}.",
            }
        resolved = current_year
        return {
            "resolved_year": resolved,
            "is_inferred": True,
            "note": f"Only current year {current_year} is available; using it.",
        }

    resolved = available_years[0]
    return {
        "resolved_year": resolved,
        "is_inferred": True,
        "note": f"No data for current year {current_year}; defaulting to most recent available year {resolved}.",
    }


def _format_hana_semantics_for_prompt(
    semantic_schema: Dict,
    user_query: str = "",
    registry: Optional[Dict[str, List[str]]] = None,
    token_budget: int = 4000,
) -> str:
    """Format HANA semantic schema for LLM prompt with token budgeting."""
    if not semantic_schema or "tables" not in semantic_schema:
        return ""

    tables = semantic_schema.get("tables", {})
    if not tables:
        return ""

    available_tables = set()
    if registry:
        for schema_name, table_list in registry.items():
            available_tables.update(str(t).upper() for t in table_list)

    selected = []

    core_tables = {
        "BSEG", "BKPF", "FAGLFLEXA", "FAGLFLEXT", "SKA1", "SKAT", "GLT0",
        "T001", "T001W", "CEPC", "CEPCT", "ANLA", "ANEP", "EKKO", "MSEG",
        "TCURR", "T003T", "T052", "LFA1", "LFB1", "KNA1", "KNB1", "AUFK",
        "BSID", "BSAD", "BSIK", "BSAK", "SETLEAF", "SETHEADERT",
    }

    for table_name in sorted(tables.keys()):
        table_def = tables[table_name]
        desc = table_def.get("description", "")
        if not desc:
            continue

        if table_name.upper() in core_tables and (not available_tables or table_name.upper() in available_tables):
            selected.append((table_name, table_def))

    if user_query:
        q_upper = user_query.upper()
        for table_name, table_def in tables.items():
            if table_name.upper() in q_upper and (table_name, table_def) not in selected:
                if not available_tables or table_name.upper() in available_tables:
                    selected.append((table_name, table_def))

    if not selected:
        return ""

    lines = ["SAP HANA SEMANTIC SCHEMA — key financial tables and their business meanings:"]
    for table_name, table_def in selected:
        desc = table_def.get("description", "")
        lines.append(f"  {table_name}: {desc}")

        columns = table_def.get("columns", {})
        col_lines = []
        for col_name, col_def in columns.items():
            parts = [col_name]
            desc = col_def.get("description", "")
            meaning = col_def.get("meaning", "")
            note = col_def.get("business_note", "")

            if desc:
                parts.append(desc)
            if meaning and meaning != desc:
                parts.append(f"({meaning})")
            if note:
                parts.append(f"[{note}]")

            if len(parts) > 1:
                col_lines.append(": ".join(parts))

        for col_line in col_lines[:12]:
            lines.append(f"    - {col_line}")
        if len(col_lines) > 12:
            lines.append(f"    - ... and {len(col_lines) - 12} more columns")

    text = "\n".join(lines)

    # Append query patterns if available
    query_patterns = semantic_schema.get("query_patterns", {})
    if query_patterns:
        pattern_lines = ["\nQUERY PATTERNS — use these templates for common financial queries:"]
        for pattern_name, pattern_def in query_patterns.items():
            desc = pattern_def.get("description", "")
            correct = pattern_def.get("correct_source", "")
            incorrect = pattern_def.get("incorrect_source", "")
            sql = pattern_def.get("sql_template", "")
            notes = pattern_def.get("notes", [])
            if desc:
                pattern_lines.append(f"  [{pattern_name}] {desc}")
            if correct:
                pattern_lines.append(f"    Correct source: {correct}")
            if incorrect:
                pattern_lines.append(f"    WRONG: {incorrect}")
            if sql:
                pattern_lines.append(f"    Template: {sql}")
            for note in notes:
                pattern_lines.append(f"    - {note}")
        lines.extend(pattern_lines)
        text = "\n".join(lines)

    try:
        tokens = _count_tokens(text)
        if tokens > token_budget:
            while tokens > token_budget and lines:
                removed = False
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].startswith("    - "):
                        lines.pop(i)
                        text = "\n".join(lines)
                        tokens = _count_tokens(text)
                        removed = True
                        break
                if not removed:
                    break
                if not any(l.startswith("    - ") for l in lines):
                    break
    except Exception:
        pass

    return text


_PLANNER_TIMEOUT_SECONDS = 30


# ═════════════════════════════════════════════════════════════════════════════
# TOKENIZER — Industry standard: real tokenizer when available
# ═════════════════════════════════════════════════════════════════════════════

def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (OpenAI-compatible) with safe fallback.

    Uses cl100k_base encoding, which matches GPT-4/OpenAI tool-calling models.
    For Gemini OpenAI-compatible endpoint, this provides accurate token counts
    for prompt budget enforcement without impacting free tier limits.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text, disallowed_special=()))
    except Exception:
        # Fallback: conservative approximation (slightly overestimates)
        return int(len(text) / 3.0)


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
# HYBRID METRIC EXTRACTION — Fast gate + LLM fallback with TTL cache
# ═════════════════════════════════════════════════════════════════════════════
#
# Industry-pattern: fast deterministic gate for common cases, LLM for
# ambiguity. Reduces latency for repeated queries and maintains accuracy
# for paraphrased/multi-metric queries.
#
# Path:
#   1. Normalize query -> cache key
#   2. Cache hit -> return cached metrics
#   3. Fast gate (regex word-boundary, longest-match-first)
#   4. Decision: gate confident? -> cache & return
#   5. LLM extraction (json_mode=True) -> cache & return
#
_METRIC_EXTRACTION_CACHE = _TTLCache(ttl=300, maxsize=256)  # 5 min, 256 entries

_METRIC_FAST_GATE_KEYWORDS = [
    "gross profit", "gross_profit", "grossprofit",
    "net profit", "net_profit", "netprofit",
    "operating income",
    "revenue", "expense", "profit", "cost", "margin",
    "ebitda", "income", "loss", "budget", "actual",
    "profitability",
]

_METRIC_LLM_PROMPT = """\
Extract all financial metrics mentioned in this query. Return ONLY JSON:
{{"metrics": ["revenue", "expense", "profit"], "time_period": "month", "year": 2025}}

Allowed metrics (use these exact lowercase forms):
revenue, expense, profit, gross_profit, net_profit, ebitda, cost, margin, income, loss, budget, actual, profitability

Query: {query}
"""


def _metric_cache_key(query: str) -> str:
    normalized = query.lower().strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


def _extract_metrics_fast_gate(query: str) -> List[str]:
    """Regex word-boundary extraction with longest-match-first ordering.

    Returns deduplicated list preserving match order (most specific first).
    """
    q = query.lower()
    found: List[str] = []
    # Sort by length descending so "gross profit" matches before "profit"
    for kw in sorted(_METRIC_FAST_GATE_KEYWORDS, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        if pattern.search(q) and kw not in found:
            found.append(kw)
    return found


def _should_fallback_to_llm(fast_metrics: List[str], query: str) -> bool:
    """Heuristic: when is the fast gate likely wrong or incomplete?"""
    q = query.lower()
    if not fast_metrics:
        return True
    # If user mentions multiple distinct financial terms but gate found only one
    financial_terms = ["revenue", "expense", "profit", "cost", "margin", "income", "loss", "budget", "actual", "ebitda"]
    mentioned = [t for t in financial_terms if t in q]
    if len(mentioned) >= 2 and len(fast_metrics) < 2:
        return True
    return False


async def _extract_metrics_llm(query: str) -> List[str]:
    """LLM-based metric extraction using structured JSON response."""
    prompt = _METRIC_LLM_PROMPT.format(query=query)
    try:
        result = await asyncio.to_thread(
            call_llm,
            system_prompt="You are a financial query analyzer. Extract metrics precisely.",
            user_prompt=prompt,
            temperature=0.0,
            max_tokens=256,
            json_mode=True,
            tier="executor",
            estimated_tokens=500,
        )
        if not result or not isinstance(result, dict):
            return []
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            return []
        # Strip markdown code fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        parsed = json.loads(content.strip())
        metrics = parsed.get("metrics", [])
        return [m.lower().strip() for m in metrics if isinstance(m, str)]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
        logger.warning("CHART_DEBUG_HYBRID: LLM metric extraction failed: %s", e)
        return []


async def _extract_metrics_hybrid(user_query: str) -> List[str]:
    """Hybrid extraction: fast gate -> cache -> LLM fallback."""
    cache_key = _metric_cache_key(user_query)
    cached = _METRIC_EXTRACTION_CACHE.get(cache_key)
    if cached is not None:
        logger.info("CHART_DEBUG_HYBRID: cache hit for query key=%s metrics=%s", cache_key[:8], cached)
        return cached

    fast_metrics = _extract_metrics_fast_gate(user_query)
    logger.info("CHART_DEBUG_HYBRID: fast gate metrics=%s", fast_metrics)

    if not _should_fallback_to_llm(fast_metrics, user_query):
        _METRIC_EXTRACTION_CACHE.set(cache_key, fast_metrics)
        return fast_metrics

    logger.info("CHART_DEBUG_HYBRID: falling back to LLM extraction")
    llm_metrics = await _extract_metrics_llm(user_query)
    logger.info("CHART_DEBUG_HYBRID: LLM metrics=%s", llm_metrics)

    # Prefer LLM result if it found more metrics, otherwise use fast gate
    final_metrics = llm_metrics if len(llm_metrics) >= len(fast_metrics) else fast_metrics
    _METRIC_EXTRACTION_CACHE.set(cache_key, final_metrics)
    return final_metrics


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


# ═════════════════════════════════════════════════════════════════════════════
# BINNING CONFIGURATION — imported from agent/output/binning.py (single source)
# ═════════════════════════════════════════════════════════════════════════════

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


def build_hana_tool_schemas(tenant_id: Optional[str] = None) -> List[Dict]:
    """Build OpenAI function-calling schemas for SAP HANA MCP tools.

    Uses dynamically discovered tool schemas when available, otherwise
    falls back to the static schema list.
    """
    from agent.integrations.hana_client import get_cached_hana_tool_schemas
    return get_cached_hana_tool_schemas(tenant_id=tenant_id)


# ═════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMA VALIDATION — Fail fast on invalid tool arguments
# ═════════════════════════════════════════════════════════════════════════════

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


_ALL_DAB_TOOLS = {"read_records", "aggregate_records", "describe_entities"}


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
        from agent.integrations.hana_client import get_cached_hana_tool_schemas
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
            logger.warning("LLM tried to call unauthorized tool '%s' — skipping", tool_name)
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
                logger.warning("LLM tool arg validation failed for %s: %s — skipping", tool_name, validation_error)
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


# ═════════════════════════════════════════════════════════════════════════════
# METADATA INFERENCE — Heuristic chart/rag/binning from query + steps
# ═════════════════════════════════════════════════════════════════════════════

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
    unique_metrics = await _extract_metrics_hybrid(user_query)
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
        history_tokens = _count_tokens(conversation_history)
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
                _count_tokens(conversation_history)
            )
        full_query = "Previous conversation:\n" + conversation_history + "\n\nCurrent question: " + user_query

    # ═══════════════════════════════════════════════════════════════════════
    # Native tool calling
    # ═══════════════════════════════════════════════════════════════════════
    all_tools = build_all_tool_schemas(cached_schema, include_hana=True, tenant_id=tenant_id)
    tool_schema_map = _build_tool_schema_map(cached_schema, include_hana=True)

    hana_semantic_schema = _load_hana_semantic_schema()
    hana_semantic_block = _format_hana_semantics_for_prompt(
        hana_semantic_schema,
        user_query=user_query,
        registry=hana_schema_registry,
    )

    system_tc = build_tool_calling_system_prompt(
        schema_block, entity_list, user_context, rag_status, tone_guidance,
        hana_schema_registry=hana_schema_registry,
        hana_semantic_block=hana_semantic_block,
    )

    logger.info("Planner: native tool calling (system=%d tokens, tools=%d)", _count_tokens(system_tc), len(all_tools))
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

        # ── SQL validation and semantic-template repair ─────────────────────
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

        # ═══════════════════════════════════════════════════════════════════════
        # TEMPORAL REASONING — resolve year for financial queries
        # ═══════════════════════════════════════════════════════════════════════
        if _is_financial_metric_query(user_query):
            available_years = await _get_available_fiscal_years(tenant_id)
            temporal = _resolve_temporal_context(user_query, available_years)
            if temporal.get("resolved_year") is not None:
                plan["temporal_context"] = temporal
                plan["reasoning"] += " " + temporal.get("note", "")

        logger.info("Planner: SUCCESS — %d steps, chart=%s, rag=%s, binning=%s",
                   len(plan["steps"]),
                   "yes" if plan["chart"] else "no",
                   "yes" if plan["rag"] else "no",
                   "yes" if plan["client_side_binning"] else "no")
        return plan

    # No tool steps produced by the model. Treat as empty plan rather than silently degraded JSON parsing.
    logger.warning("Planner: native tool calling produced no steps. Returning empty plan.")
    return _empty_plan()