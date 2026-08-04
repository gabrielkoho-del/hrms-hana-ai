"""agent/dab/odata_normalizer.py

Pre-call OData request normalization for DAB/OData strictness.

DAB (Azure Data API Builder) enforces OData protocol rules strictly:
  - Entity and field names are CASE-SENSITIVE
  - Filter values must match the declared OData type (no quoting numeric fields)

This module provides schema-driven normalization that corrects these issues
before the request reaches DAB, acting as a defense-in-depth layer alongside
prompt-level instructions in tool_planner.py.

Usage:
    from agent.dab.odata_normalizer import normalize_odata_args

    # Inside the DAB tool call executor, before sending the request:
    normalize_odata_args(args, cached_schema)
"""

import logging
import re
from typing import Any, Dict, List, Optional

from agent.dab.metrics import record_entity_normalized

logger = logging.getLogger("hr_agent")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_EDM_TYPES: frozenset = frozenset((
    "Edm.Int64", "Edm.Int32", "Edm.Int16",
    "Edm.Double", "Edm.Single", "Edm.Decimal",
    "Edm.Guid",
    "int64", "int32", "int16",
    "double", "float", "decimal",
    "int", "integer",  # Also strip quotes from DAB 'Int' type
))

# Fields that should ALWAYS be treated as string (even if schema says numeric)
# EMPLOYEE_NO is an alphanumeric employee identifier that looks like '000024'
# Key-fields in V_EMP so it must preserve leading zeros for exact match
STRING_EXCEPTIONS: frozenset = frozenset({"employee_no"})

STRING_EDM_TYPES: frozenset = frozenset((
    "Edm.String", "Edm.String", "string", "nvarchar", "varchar",
))

# OData keywords that must NOT be treated as field names during filter parsing
_ODATA_KEYWORDS = frozenset((
    "eq", "ne", "gt", "lt", "ge", "le",
    "and", "or", "not", "in", "has",
    "contains", "startswith", "endswith",
    "null", "true", "false",
    "tolower", "toupper", "trim", "substring",
    "indexof", "length", "concat", "replace",
    "year", "month", "day", "hour", "minute", "second",
    "date", "now", "datetimeoffset",
))

# ─────────────────────────────────────────────────────────────────────────────
# Schema Lookups
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_entity_data(entity_name: str, cached_schema: Any) -> Optional[Dict]:
    """Locate an entity's data dict within the cached schema (dict or list format)."""
    if not entity_name or not cached_schema:
        return None

    if isinstance(cached_schema, dict):
        return cached_schema.get(entity_name) or next(
            (v for k, v in cached_schema.items() if k.lower() == entity_name.lower()),
            None,
        )
    if isinstance(cached_schema, list):
        return next(
            (e for e in cached_schema if isinstance(e, dict) and e.get("name", "").lower() == entity_name.lower()),
            None,
        )
    return None


def get_field_map(entity_name: str, cached_schema: Any) -> Dict[str, str]:
    """Build a case-insensitive field name map for an entity.

    Returns:
        ``{lowercase_name: exact_schema_name}`` for every field in the entity.
    """
    entity_data = _resolve_entity_data(entity_name, cached_schema)
    field_map: Dict[str, str] = {}
    if not isinstance(entity_data, dict):
        return field_map

    for f in entity_data.get("fields", entity_data.get("columns", [])):
        if isinstance(f, dict):
            name = f.get("name", "")
            if name:
                field_map[name.lower()] = name
    return field_map


def get_field_type_map(entity_name: str, cached_schema: Any) -> Dict[str, str]:
    """Build a map of ``{lowercase_field_name: edm_type}`` for an entity.

    Returns types like ``'Edm.Int64'``, ``'Edm.Double'``, ``'string'``, ``'boolean'``.
    Used to detect type mismatches in OData filter expressions.
    """
    entity_data = _resolve_entity_data(entity_name, cached_schema)
    type_map: Dict[str, str] = {}
    if not isinstance(entity_data, dict):
        return type_map

    for f in entity_data.get("fields", entity_data.get("columns", [])):
        if isinstance(f, dict):
            name = f.get("name", "")
            ftype = f.get("type", "")
            if name and ftype:
                type_map[name.lower()] = ftype
    return type_map


# ─────────────────────────────────────────────────────────────────────────────
# Entity Name Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_entity_name(args: Dict, cached_schema: Any) -> None:
    """Correct the ``entity`` value in *args* to match the schema exactly.

    Resolution order:
      1. Exact match  →  no change
      2. Case-insensitive match  →  correct casing
      3. Fuzzy prefix/suffix match  →  correct (e.g. ``employee`` → ``Employee``)

    Mutates *args* in place.
    """
    entity = args.get("entity", "")
    if not entity or not isinstance(entity, str):
        return

    # Build schema name list (handles dict or list)
    if isinstance(cached_schema, dict):
        schema_names = list(cached_schema.keys())
    elif isinstance(cached_schema, list):
        schema_names = [e.get("name", "") for e in cached_schema if isinstance(e, dict)]
    else:
        return

    # 1. Exact match — nothing to do
    if entity in schema_names:
        return

    # 2. Case-insensitive match
    matched = next((k for k in schema_names if k.lower() == entity.lower()), None)
    if matched:
        logger.warning("Entity name case-corrected: %s -> %s", entity, matched)
        record_entity_normalized("entity_case", entity, matched)
        args["entity"] = matched
        return

    # 3. Fuzzy prefix/suffix match (e.g. "employee" vs "Employees")
    for name in schema_names:
        if name.lower().startswith(entity.lower()) or name.lower().endswith(entity.lower()):
            logger.warning("Entity name fuzzy-corrected: %s -> %s", entity, name)
            record_entity_normalized("entity_fuzzy", entity, name)
            args["entity"] = name
            return

    logger.error("Entity '%s' not found in schema. Available: %s", entity, schema_names)


# ─────────────────────────────────────────────────────────────────────────────
# Field Name + Type Normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_field_names(args: Dict, cached_schema: Any) -> None:
    """Normalize field names across all OData args to match exact schema casing.

    Corrects field names in: ``select``, ``filter``, ``orderby``, ``groupby``,
    ``field``, ``having``.

    Also fixes type mismatches in filter expressions: strips quotes from values
    compared against numeric fields (e.g. ``EMPLOYEE_NO eq '000024'`` →
    ``EMPLOYEE_NO eq 000024``) and adds quotes to string fields with unquoted
    numeric values (e.g. ``employee_no eq 24`` → ``employee_no eq '24'``).

    Mutates *args* in place.
    """
    entity_name = args.get("entity", "")
    field_map = get_field_map(entity_name, cached_schema)
    field_types = get_field_type_map(entity_name, cached_schema)
    if not field_map:
        return

    def _correct(raw: str) -> str:
        """Case-correct a single field name token."""
        stripped = raw.strip()
        if not stripped or stripped == "*":
            return stripped
        exact = field_map.get(stripped.lower())
        if exact and exact != stripped:
            logger.info("Field name case-corrected: %s -> %s", stripped, exact)
            record_entity_normalized("field_case", stripped, exact)
            return exact
        return stripped

    def _correct_orderby(token: str) -> str:
        """Case-correct the field portion of an ``orderby`` token (e.g. ``salary desc``)."""
        parts = token.strip().rsplit(None, 1)
        field = _correct(parts[0])
        if len(parts) == 2:
            return f"{field} {parts[1].lower()}"
        return field

    def _correct_filter(filter_str: str) -> str:
        """Case-correct field names inside an OData ``$filter`` expression,
        then strip quotes from numeric field comparisons, then add quotes
        to string fields with unquoted numeric values."""
        # Step 1: token-aware field name correction
        tokens = re.split(r"(\s+|[()',])", filter_str)
        result: List[str] = []
        in_quote = False

        for token in tokens:
            if token == "'":
                in_quote = not in_quote
                result.append(token)
                continue
            if in_quote or not token.strip():
                result.append(token)
                continue
            if token.strip().lower() in _ODATA_KEYWORDS:
                result.append(token)
                continue
            result.append(_correct(token))

        joined = "".join(result)

        # Step 2: type-aware quote stripping on numeric fields
        joined = _strip_quotes_on_numeric_fields(joined, field_types)

        # Step 3: add quotes to string fields with unquoted numeric values
        joined = _add_quotes_on_string_fields(joined, field_types)

        return joined

    # ── Apply to each OData arg key ──

    select_val = args.get("select")
    if select_val:
        if isinstance(select_val, str):
            args["select"] = ",".join(_correct(f) for f in select_val.split(",") if f.strip())
        elif isinstance(select_val, list):
            args["select"] = ",".join(_correct(str(f)) for f in select_val if str(f).strip())

    filter_val = args.get("filter")
    if filter_val and isinstance(filter_val, str):
        args["filter"] = _correct_filter(filter_val)

    orderby_val = args.get("orderby")
    if orderby_val:
        if isinstance(orderby_val, str):
            args["orderby"] = ",".join(
                _correct_orderby(f) for f in orderby_val.split(",") if f.strip()
            )
        elif isinstance(orderby_val, list):
            args["orderby"] = [_correct_orderby(str(f)) for f in orderby_val if str(f).strip()]

    groupby_val = args.get("groupby")
    if groupby_val:
        if isinstance(groupby_val, str):
            args["groupby"] = [_correct(f) for f in groupby_val.split(",") if f.strip()]
        elif isinstance(groupby_val, list):
            args["groupby"] = [_correct(str(f)) for f in groupby_val if str(f).strip()]

    field_val = args.get("field")
    if field_val and isinstance(field_val, str) and field_val != "*":
        args["field"] = _correct(field_val)

    having_val = args.get("having")
    if having_val and isinstance(having_val, str):
        args["having"] = _correct_filter(having_val)


# ─────────────────────────────────────────────────────────────────────────────
# Filter Type Correction (internal)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_quotes_on_numeric_fields(filter_str: str, field_types: Dict[str, str]) -> str:
    """Post-process a filter string to remove quotes around numeric field values.

    Corrects patterns like ``EMPLOYEE_NO eq '000024'`` → ``EMPLOYEE_NO eq 000024``
    when ``EMPLOYEE_NO`` is declared as ``Edm.Int64`` in the schema.
    
    EXCEPTION: Fields in STRING_EXCEPTIONS are always kept as strings to preserve
    leading zeros (e.g., '000024' stays '000024', not 24).
    """
    def _replace(match: re.Match) -> str:
        field_name = match.group(1)
        operator = match.group(2)
        quoted_val = match.group(3)
        field_lower = field_name.lower()

        # Preserve string quotes for STRING_EXCEPTIONS (like EMPLOYEE_NO)
        if field_lower in STRING_EXCEPTIONS:
            return match.group(0)  # Leave as-is, keep the quotes

        if field_lower in field_types and field_types[field_lower] in NUMERIC_EDM_TYPES:
            inner = quoted_val.strip("'\"")
            try:
                float(inner)
                logger.info(
                    "Filter type-fix: %s %s '%s' -> %s %s %s (numeric field)",
                    field_name, operator, quoted_val, field_name, operator, inner,
                )
                record_entity_normalized("filter_quote_strip", quoted_val, inner)
                return f"{field_name} {operator} {inner}"
            except ValueError:
                pass
        return match.group(0)

    return re.sub(
        r"(\w+)\s+(eq|ne|gt|lt|ge|le)\s+(['\"][^'\"]*['\"])",
        _replace,
        filter_str,
        flags=re.IGNORECASE,
    )


def _add_quotes_on_string_fields(filter_str: str, field_types: Dict[str, str]) -> str:
    """Post-process a filter string to add quotes around values for string fields.

    Corrects patterns like ``employee_no eq 24`` → ``employee_no eq '24'``
    when ``employee_no`` is a string field. This handles cases where the LLM
    generates unquoted numeric-looking values for string-ID fields.
    """
    def _replace(match: re.Match) -> str:
        field_name = match.group(1)
        operator = match.group(2)
        unquoted_val = match.group(3)
        field_lower = field_name.lower()

        # Only quote for fields known to be strings
        is_string_exception = field_lower in STRING_EXCEPTIONS
        is_string_type = (
            field_lower in field_types
            and field_types[field_lower] in STRING_EDM_TYPES
        )
        if not is_string_exception and not is_string_type:
            return match.group(0)

        # If the value looks numeric, add quotes
        try:
            float(unquoted_val)
            new_filter = f"{field_name} {operator} '{unquoted_val}'"
            logger.info(
                "Filter type-fix: %s %s %s -> %s %s '%s' (string field)",
                field_name, operator, unquoted_val, field_name, operator, unquoted_val,
            )
            record_entity_normalized("filter_quote_add", unquoted_val, f"'{unquoted_val}'")
            return new_filter
        except ValueError:
            pass
        return match.group(0)

    return re.sub(
        r"(\w+)\s+(eq|ne|gt|lt|ge|le)\s+([^'\"\s,)]+)",
        _replace,
        filter_str,
        flags=re.IGNORECASE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def normalize_odata_args(args: Dict, cached_schema: Any) -> None:
    """Run all OData normalizations on *args* in place.

    Applies in order:
      1. Entity name case correction
      2. Field name case correction across all OData parameters
      3. Type-aware filter value correction (quote stripping for numeric fields)

    This is the single entry point to call from the DAB tool executor.
    """
    normalize_entity_name(args, cached_schema)
    normalize_field_names(args, cached_schema)