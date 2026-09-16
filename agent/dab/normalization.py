"""DAB tool-call normalization: entity names and dimension fields."""
import logging
import re
from typing import Any, Dict, List

from agent.core.config_loader import (
    get_all_dimension_fields,
    get_dimension_mappings,
)

logger = logging.getLogger("hr_agent")


def normalize_entity_names(steps: List[Dict], cached_schema: Dict) -> List[Dict]:
    """Post-processor: normalize entity names in DAB tool-call steps to match schema exactly.

    HANA tools use schema_name/table_name or raw SQL; they are skipped here.
    """
    if not steps or not cached_schema:
        return steps

    schema_names = list(cached_schema.keys()) if isinstance(cached_schema, dict) else []
    lower_to_exact = {k.lower(): k for k in schema_names}

    for step in steps:
        tool = step.get("tool", "")
        # Import here to avoid circular imports
        from agent.core.tool_planner import _DAB_TOOL_NAMES
        if tool not in _DAB_TOOL_NAMES:
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


def normalize_dimension_fields(steps: List[Dict], user_query: str, cached_schema: Dict) -> List[Dict]:
    """Post-processor: correct dimension fields in DAB tool-call steps based on user query keywords.

    Prevents the LLM from choosing semantically wrong dimension fields
    (e.g., COST_CENTER when the user asked for department).
    """
    if not steps or not cached_schema or not user_query:
        return steps

    dimension_map = get_dimension_mappings()
    all_dimension_fields = get_all_dimension_fields()
    query_lower = user_query.lower()

    # Find which dimension the user is asking for
    target_dim = None
    target_cols = []
    for kw, cols in dimension_map.items():
        if kw in query_lower:
            target_dim = kw
            target_cols = [c.lower() for c in cols]
            break

    if not target_dim:
        return steps

    for step in steps:
        tool = step.get("tool", "")
        # Import here to avoid circular imports
        from agent.core.tool_planner import _DAB_TOOL_NAMES
        if tool not in _DAB_TOOL_NAMES:
            continue

        args = step.get("args", {})
        entity = args.get("entity", "")
        if not entity:
            continue

        entity_schema = cached_schema.get(entity, {})
        if isinstance(entity_schema, dict):
            # cached_schema stores full entity dicts; field names are in entity_schema["fields"]
            fields_list = entity_schema.get("fields", entity_schema.get("columns", []))
            if isinstance(fields_list, list) and fields_list:
                col_names = [f.get("name") or f.get("column_name") or str(f) for f in fields_list]
            else:
                # Fallback: entity_schema might be a dict of field names (test structure)
                col_names = list(entity_schema.keys())
        elif isinstance(entity_schema, list):
            col_names = [c.get("name") or c.get("column_name") or str(c) for c in entity_schema]
        else:
            continue

        col_set = {c.lower(): c for c in col_names}

        # Find the preferred column that exists in this entity
        matched_col = None
        for tc in target_cols:
            if tc in col_set:
                matched_col = col_set[tc]
                break

        if not matched_col:
            continue

        # Correct groupby in aggregate_records
        if tool == "aggregate_records":
            groupby = args.get("groupby", [])
            if groupby:
                for i, gb in enumerate(groupby):
                    gb_lower = gb.lower()
                    if gb_lower not in target_cols and gb_lower in all_dimension_fields and gb_lower in col_set:
                        logger.warning(
                            "Planner dimension corrected: %s -> %s (query asked for %s)",
                            gb, matched_col, target_dim
                        )
                        groupby[i] = matched_col
                        args["groupby"] = groupby
                        break

        # Correct select in read_records
        elif tool == "read_records":
            select_val = args.get("select")
            if select_val:
                if isinstance(select_val, str):
                    fields = [f.strip() for f in select_val.split(",") if f.strip()]
                    modified = False
                    for i, field in enumerate(fields):
                        field_lower = field.lower()
                        if field_lower not in target_cols and field_lower in all_dimension_fields and field_lower in col_set:
                            logger.warning(
                                "Planner dimension corrected: %s -> %s (query asked for %s)",
                                field, matched_col, target_dim
                            )
                            fields[i] = matched_col
                            modified = True
                    if modified:
                        args["select"] = ",".join(fields)

    return steps
