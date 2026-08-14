"""agent/integrations/hana_sql.py

HANA SQL validation and semantic-template repair.

Validates HANA SQL for common correctness issues and can repair invalid
queries using templates from hana-semantics-hr.json.
"""
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("hr_agent")

_YEAR_PATTERN = re.compile(r"\b(20\d{2}|19\d{2})\b")


def _extract_year_from_query(user_query: str) -> Optional[int]:
    """Extract explicit year mention from the query, if any."""
    matches = _YEAR_PATTERN.findall(user_query)
    if not matches:
        return None
    # Prefer the last mentioned year (usually the relevant one)
    return int(matches[-1])


def _load_hana_semantic_schema() -> Optional[Dict]:
    """Load HANA semantic descriptions from hana-semantics-hr.json."""
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


def _validate_hana_sql(query: str) -> Tuple[bool, Optional[str]]:
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
