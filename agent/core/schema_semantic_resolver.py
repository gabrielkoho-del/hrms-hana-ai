"""Schema-driven semantic dimension resolution.

Builds a concept-to-column mapping from schema field metadata (name, description,
type) so dimension normalization can rely on schema semantics instead of only
hardcoded keyword maps.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hr_agent")

# Heuristic concept hints extracted from field names/descriptions.
# Each entry is (concept_keywords, preferred_column_patterns).
_CONCEPT_HINTS: List[Tuple[List[str], List[str]]] = [
    (["department", "dept"], ["department_code", "department", "dept_code"]),
    (["branch"], ["branch_code", "branch"]),
    (["company"], ["company_code", "company"]),
    (["division"], ["division_code", "division"]),
    (["section"], ["section_code", "section"]),
    (["position", "job", "role"], ["position_description", "position_code", "position"]),
    (["gender"], ["gender"]),
    (["nationality", "nation"], ["nationality_code", "nationality"]),
    (["marital", "marriage"], ["marital_status", "marital"]),
    (["status", "employment status", "employee status"], ["employee_status", "status"]),
    (["location"], ["location_code", "location"]),
    (["grade", "pay grade", "job grade"], ["grade_code", "grade"]),
    (["category"], ["category_code", "category"]),
    (["pay country", "country", "payroll country"], ["pay_country", "country_code"]),
    (["cost center", "costcentre"], ["cost_center", "cost_center_code", "costcentre"]),
    (["profit center", "profitcentre"], ["profit_center", "profit_center_code", "profitcentre"]),
    (["superior", "manager", "reporting manager"], ["superior_no", "superior", "manager_no"]),
    (["level"], ["employee_level", "level_code", "level"]),
    (["confirmation", "probation"], ["confirmation_status", "confirmation"]),
    (["resigned", "resignation", "termination"], ["date_resigned", "resigned_date", "resigned"]),
    (["joined", "join date", "hire date", "date joined"], ["date_joined", "join_date", "hire_date"]),
    (["hire", "hire date"], ["date_joined", "hire_date", "hire"]),
    (["tenure"], ["date_joined", "tenure_band", "tenure"]),
]


def _build_field_text(field: Dict[str, Any]) -> str:
    """Build searchable text from field metadata."""
    parts = [str(field.get("name", "")), str(field.get("column_name", ""))]
    parts.append(str(field.get("type", "")))
    if field.get("description"):
        parts.append(str(field.get("description")))
    return " ".join(parts).lower()


def _match_concept(field_text: str, concept_keywords: List[str]) -> bool:
    """Check if field text matches any concept keyword."""
    for kw in concept_keywords:
        if kw in field_text:
            return True
    return False


def _matches_column(field_name: str, patterns: List[str]) -> bool:
    """Check if field name matches any preferred column pattern."""
    fn = field_name.lower()
    for pat in patterns:
        if fn == pat.lower() or fn.endswith("_" + pat.lower()) or fn.startswith(pat.lower() + "_"):
            return True
    return False


def build_semantic_dimension_map(cached_schema: Dict[str, Any]) -> Dict[str, List[str]]:
    """Build a dimension concept -> preferred column list mapping from schema metadata.

    Scans all entity fields and infers semantic concepts from field names and
    descriptions. Returns a mapping compatible with the existing YAML
    `dimension_map` format.
    """
    if not cached_schema:
        return {}

    # Collect all fields across all entities
    all_fields: List[Tuple[str, Dict[str, Any]]] = []
    for entity_name, entity_data in cached_schema.items():
        if not isinstance(entity_data, dict):
            continue
        fields = entity_data.get("fields", entity_data.get("columns", []))
        if not isinstance(fields, list):
            continue
        for f in fields:
            if isinstance(f, dict):
                all_fields.append((entity_name, f))

    if not all_fields:
        return {}

    # Build concept -> column mapping
    concept_map: Dict[str, List[str]] = {}
    for concept_keywords, preferred_patterns in _CONCEPT_HINTS:
        matched_cols: List[str] = []
        for entity_name, field in all_fields:
            field_text = _build_field_text(field)
            if _match_concept(field_text, concept_keywords):
                col_name = field.get("name") or field.get("column_name") or str(field)
                if col_name and col_name not in matched_cols:
                    matched_cols.append(col_name)

        if matched_cols:
            primary_concept = concept_keywords[0]
            concept_map[primary_concept] = matched_cols

    logger.info("SchemaSemanticResolver: built %d concept mappings from %d fields", len(concept_map), len(all_fields))
    return concept_map


def resolve_dimension_fields(
    user_query: str,
    cached_schema: Dict[str, Any],
    yaml_dimension_map: Optional[Dict[str, List[str]]] = None,
) -> Optional[Tuple[str, List[str]]]:
    """Resolve the target dimension concept and preferred columns from the schema.

    Args:
        user_query: The user's query text.
        cached_schema: The cached DAB schema.
        yaml_dimension_map: Optional YAML-based dimension map as fallback.

    Returns:
        Tuple of (concept_key, preferred_columns) or None if no match.
    """
    query_lower = user_query.lower()
    if not query_lower:
        return None

    # Build schema-driven dimension map
    schema_map = build_semantic_dimension_map(cached_schema)

    # Merge with YAML map (YAML takes precedence for known concepts)
    merged_map = {**schema_map}
    if yaml_dimension_map:
        for kw, cols in yaml_dimension_map.items():
            if kw in merged_map:
                # Prefer YAML columns but keep schema columns as fallback
                merged_map[kw] = cols + [c for c in merged_map[kw] if c not in cols]
            else:
                merged_map[kw] = cols

    # Find matching concept
    for kw, cols in merged_map.items():
        if kw in query_lower:
            return kw, cols

    return None
