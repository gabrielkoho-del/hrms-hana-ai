"""agent/hana/semantic.py

HANA semantic schema formatting for LLM prompts.

Formats hana-semantics-hr.json table descriptions and query patterns
into a token-budgeted prompt block for the planner.
"""
import logging
from typing import Any, Dict, List, Optional

from agent.core.utils import count_tokens

logger = logging.getLogger("hr_agent")

_CORE_TABLES = {
    "BSEG", "BKPF", "FAGLFLEXA", "FAGLFLEXT", "SKA1", "SKAT", "GLT0",
    "T001", "T001W", "CEPC", "CEPCT", "ANLA", "ANEP", "EKKO", "MSEG",
    "TCURR", "T003T", "T052", "LFA1", "LFB1", "KNA1", "KNB1", "AUFK",
    "BSID", "BSAD", "BSIK", "BSAK", "SETLEAF", "SETHEADERT",
}


def format_hana_semantics_for_prompt(
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

    for table_name in sorted(tables.keys()):
        table_def = tables[table_name]
        desc = table_def.get("description", "")
        if not desc:
            continue

        if table_name.upper() in _CORE_TABLES and (not available_tables or table_name.upper() in available_tables):
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
        tokens = count_tokens(text)
        if tokens > token_budget:
            while tokens > token_budget and lines:
                removed = False
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].startswith("    - "):
                        lines.pop(i)
                        text = "\n".join(lines)
                        tokens = count_tokens(text)
                        removed = True
                        break
                if not removed:
                    break
                if not any(l.startswith("    - ") for l in lines):
                    break
    except Exception:
        pass

    return text
